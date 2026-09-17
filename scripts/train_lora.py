# -*- coding: utf-8 -*-
"""
D2: Qwen2.5-0.5B-Instruct CPU LoRA 微调 —— 考勤规则判定

要点：
  1. 只在 assistant 回答段计算 loss（prompt 段 label 置 -100）——避免模型把算力浪费在背 prompt 上
  2. CPU fp32 训练，LoRA 只训练 ~0.6% 参数
  3. 训练前/后端到端对比，量化微调收益
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8')

LAB = os.environ.get('LLM_LAB_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(LAB, 'models', 'Qwen2.5-0.5B-Instruct')
DATA = os.path.join(LAB, 'data')
OUT = os.path.join(LAB, 'out', 'lora-attendance')
MAX_LEN = 384

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)
from peft import LoraConfig, get_peft_model, TaskType

torch.manual_seed(42)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def encode_sample(rec):
    """返回 labels 已掩码的样本。messages = [system, user, assistant]"""
    msgs = rec['messages']
    prompt_msgs = [m for m in msgs if m['role'] != 'assistant']
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    pfx_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full = tok(full_text, add_special_tokens=False)['input_ids'][:MAX_LEN]
    plen = len(tok(pfx_text, add_special_tokens=False)['input_ids'])
    labels = list(full)
    for i in range(min(plen, len(labels))):
        labels[i] = -100          # 只在 assistant 段算 loss
    return {'input_ids': full, 'labels': labels, 'attention_mask': [1] * len(full)}


def load_jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        L = max(len(f['input_ids']) for f in feats)
        out = {'input_ids': [], 'labels': [], 'attention_mask': []}
        for f in feats:
            n = L - len(f['input_ids'])
            out['input_ids'].append(f['input_ids'] + [self.pad_id] * n)
            out['labels'].append(f['labels'] + [-100] * n)
            out['attention_mask'].append(f['attention_mask'] + [0] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


train_ds = [encode_sample(r) for r in load_jsonl(os.path.join(DATA, 'train.jsonl'))]
valid_ds = [encode_sample(r) for r in load_jsonl(os.path.join(DATA, 'valid.jsonl'))]
los = [sum(1 for x in d['labels'] if x != -100) for d in train_ds]
print('训练集 %d 条 / 验证集 %d 条' % (len(train_ds), len(valid_ds)))
print('序列长度 max %d，回答段 token 数 中位 %d 均值 %.0f' % (
    max(len(d['input_ids']) for d in train_ds),
    sorted(los)[len(los) // 2], sum(los) / len(los)))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
model.config.use_cache = False

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    bias='none',
)
model = get_peft_model(model, lora_cfg)
tr, tot = model.get_nb_trainable_parameters()
print('可训练参数 %s / 总参数 %s = %.3f%%' % (f'{tr:,}', f'{tot:,}', 100 * tr / tot))

args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type='cosine',
    warmup_steps=20,
    logging_steps=10,
    eval_strategy='epoch',
    # 关键：不做中间保存。save_total_limit 会触发「删除旧 checkpoint」，
    # 而这个删除在受限沙箱里会被安全守卫拦截并中断训练进程。
    save_strategy='no',
    save_total_limit=None,
    fp16=False, bf16=False,
    optim='adamw_torch',
    report_to=[],
    seed=42,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=valid_ds,
    data_collator=PadCollator(tok.pad_token_id),
)

t0 = time.time()
trainer.train()
print('\n训练完成，耗时 %.1f 分钟' % ((time.time() - t0) / 60))

trainer.save_model(OUT)
tok.save_pretrained(OUT)
print('LoRA 适配器已保存 ->', OUT)
for f in sorted(os.listdir(OUT)):
    fp = os.path.join(OUT, f)
    if os.path.isfile(fp):
        print('   %-24s %8.2f MB' % (f, os.path.getsize(fp) / 2**20))
