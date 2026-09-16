# -*- coding: utf-8 -*-
"""前向冒烟测试：加载 Qwen2.5-0.5B-Instruct，跑一次 CPU 生成，确认权重可用。"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')

LAB = r'C:\Users\intpj\WorkBuddy\2026-09-16-11-35-28\llm-lab'
MODEL_DIR = os.path.join(LAB, 'models', 'Qwen2.5-0.5B-Instruct')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '| threads', torch.get_num_threads())

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
print('tokenizer 就绪 %.1fs | vocab %d' % (time.time() - t0, tok.vocab_size))

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
model.eval()
print('模型就绪 %.1fs | 参数量 %.3f B' % (
    time.time() - t0, sum(p.numel() for p in model.parameters()) / 1e9))

# 用真实业务 prompt 探一下基线能力（未微调时表现）
msg = [
    {'role': 'system', 'content': '你是考勤核算助手。只输出 JSON，不要任何解释。'},
    {'role': 'user', 'content':
        '请判定以下考勤记录：\n张伟 2026-09-20(周三) 打卡记录：08:47、18:31，共 2 次。\n'
        '规则：上班 08:30 前打卡为正常，下班 17:30 后打卡为正常；'
        '输出字段 date/weekday/status/late_minutes/early_leave_minutes/overtime_minutes/issues/note'},
]
text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors='pt')

t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                         pad_token_id=tok.eos_token_id)
gen = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
dt = time.time() - t0
n_new = out.shape[1] - inputs['input_ids'].shape[1]

print('\n--- 生成结果（%.1fs, %d tokens, %.1f tok/s）---' % (dt, n_new, n_new / dt))
print(gen)
print('--- 基线能力探针结束 ---')
print('结论：权重可加载、可推理。基线输出 %s。' % (
    '已接近规范 JSON' if gen.strip().startswith('{') else '未按 JSON 格式输出（正是微调要解决的点）'))
