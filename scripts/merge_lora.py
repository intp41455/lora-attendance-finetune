# -*- coding: utf-8 -*-
"""
D4-1: 合并 LoRA 适配器进基座权重，产出独立可部署模型。
（对应简历中「LoRA 微调后合并权重、脱离 PEFT 依赖独立推理」）
"""
import os, sys, time, json
sys.stdout.reconfigure(encoding='utf-8')

LAB = os.environ.get('LLM_LAB_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(LAB, 'models', 'Qwen2.5-0.5B-Instruct')
LORA_DIR = os.path.join(LAB, 'out', 'lora-attendance')
MERGED = os.path.join(LAB, 'out', 'qwen2.5-0.5b-attendance-merged')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)

print('加载基座 + LoRA 适配器...')
base = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
model = PeftModel.from_pretrained(base, LORA_DIR)

t0 = time.time()
print('合并权重（merge_and_unload）...')
model = model.merge_and_unload()
print('合并完成 %.1fs' % (time.time() - t0))

os.makedirs(MERGED, exist_ok=True)
model.save_pretrained(MERGED, safe_serialization=True)
tok.save_pretrained(MERGED)

tot = sum(os.path.getsize(os.path.join(r, f))
          for r, _, fs in os.walk(MERGED) for f in fs)
print('合并模型已保存 -> %s  (%.2f GB)' % (MERGED, tot / 2**30))

# 冒烟验证：合并后独立加载（不依赖 LoRA）能否复现微调行为
del model, base
import gc
gc.collect()
print('\n独立加载合并模型做冒烟验证...')
m2 = AutoModelForCausalLM.from_pretrained(
    MERGED, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
m2.eval()
msg = [{'role': 'system', 'content': '你是考勤核算助手。只输出 JSON，不要任何解释。'},
       {'role': 'user', 'content':
        '请判定以下考勤记录：\n张伟 2026-09-20(周三) 打卡记录：08:47、18:31，共 2 次。'}]
text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
inp = tok(text, return_tensors='pt')
with torch.no_grad():
    out = m2.generate(**inp, max_new_tokens=160, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True))
print('\n（基线此处会答 late_minutes=2；正确应为 17）')
