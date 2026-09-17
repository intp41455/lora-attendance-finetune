# -*- coding: utf-8 -*-
"""
微调前后对比评测：基座模型 vs 基座+LoRA

指标：
  1. JSON 可解析率        —— 格式遵从度
  2. status 字段完全正确率 —— 业务判定准确率
  3. 分钟数（迟到/早退/加班）字段精确命中率
  4. 全字段整体正确率（6 个业务字段全对才计入）
"""
import os, sys, json, time, re
sys.stdout.reconfigure(encoding='utf-8')

LAB = os.environ.get('LLM_LAB_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(LAB, 'models', 'Qwen2.5-0.5B-Instruct')
LORA_DIR = os.path.join(LAB, 'out', 'lora-attendance')
VALID = os.path.join(LAB, 'data', 'valid.jsonl')
RESULT = os.path.join(LAB, 'out', 'eval_compare.json')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

FIELD_KEYS = ['status', 'late_minutes', 'early_leave_minutes', 'overtime_minutes']


def parse_json(txt):
    m = re.search(r'\{.*\}', txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_inputs(rec, tok):
    msgs = [m for m in rec['messages'] if m['role'] != 'assistant']
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors='pt')


def run_eval(model, tok, rows, tag):
    model.eval()
    n = len(rows)
    ok_parse = ok_status = ok_nums = ok_all = 0
    details, t0 = [], time.time()
    for rec in rows:
        gold = json.loads(rec['messages'][-1]['content'])
        inp = build_inputs(rec, tok)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = parse_json(txt)
        s_ok = n_ok = a_ok = False
        if pred:
            ok_parse += 1
            s_ok = str(pred.get('status', '')).strip() == str(gold['status']).strip()
            n_ok = all(pred.get(k) is not None and int(pred.get(k, -1)) == int(gold[k])
                       for k in FIELD_KEYS[1:])
            ok_status += s_ok
            ok_nums += n_ok
            a_ok = s_ok and n_ok
            ok_all += a_ok
        details.append({'gold': gold, 'pred': pred, 'raw': txt.strip()[:300],
                        'status_ok': s_ok, 'nums_ok': n_ok})
    dt = time.time() - t0
    print('\n[%s] %d 条  耗时 %.1f 分钟 (%.1f tok/s 估)' % (tag, n, dt / 60, 0))
    print('  JSON 可解析      %2d/%d  %5.1f%%' % (ok_parse, n, 100 * ok_parse / n))
    print('  status 判定正确  %2d/%d  %5.1f%%' % (ok_status, n, 100 * ok_status / n))
    print('  分钟数字段正确   %2d/%d  %5.1f%%' % (ok_nums, n, 100 * ok_nums / n))
    print('  全字段整体正确   %2d/%d  %5.1f%%' % (ok_all, n, 100 * ok_all / n))
    return {'tag': tag, 'n': n, 'parse': ok_parse, 'status': ok_status,
            'nums': ok_nums, 'all': ok_all, 'details': details}


rows = [json.loads(l) for l in open(VALID, encoding='utf-8') if l.strip()]
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

print('=' * 62)
print('加载基座模型...')
base = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
r_base = run_eval(base, tok, rows, '基座 Qwen2.5-0.5B-Instruct')

print('=' * 62)
print('挂载 LoRA 适配器...')
ft = PeftModel.from_pretrained(base, LORA_DIR)
ft.eval()
r_ft = run_eval(ft, tok, rows, '基座 + LoRA(考勤规则)')

print('\n' + '=' * 62)
print('%-26s %10s %10s %10s' % ('指标', '微调前', '微调后', '提升'))
print('-' * 62)
for key, name in [('parse', 'JSON 可解析率'), ('status', 'status 判定正确率'),
                  ('nums', '分钟数字段正确率'), ('all', '全字段整体正确率')]:
    b = 100 * r_base[key] / r_base['n']
    a = 100 * r_ft[key] / r_ft['n']
    print('%-26s %9.1f%% %9.1f%% %+9.1f%%' % (name, b, a, a - b))
print('=' * 62)

# 抽 3 条看具体变化
print('\n--- 逐条对比抽样 ---')
shown = 0
for i, (d0, d1) in enumerate(zip(r_base['details'], r_ft['details'])):
    if d0['status_ok'] and d0['nums_ok'] and d1['status_ok'] and d1['nums_ok']:
        continue
    if shown >= 3:
        break
    shown += 1
    g = d0['gold']
    print('\n[%d] 真值: status=%s late=%s early=%s ot=%s' % (
        i + 1, g['status'], g['late_minutes'], g['early_leave_minutes'], g['overtime_minutes']))
    print('    微调前: %s' % json.dumps(d0['pred'], ensure_ascii=False) if d0['pred'] else '    微调前: 解析失败')
    print('    微调后: %s' % json.dumps(d1['pred'], ensure_ascii=False) if d1['pred'] else '    微调后: 解析失败')

os.makedirs(os.path.dirname(RESULT), exist_ok=True)
with open(RESULT, 'w', encoding='utf-8') as f:
    json.dump({'base': r_base, 'finetuned': r_ft}, f, ensure_ascii=False, indent=2)
print('\n明细已存 ->', RESULT)
