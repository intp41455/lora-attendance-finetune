# -*- coding: utf-8 -*-
"""
D4-2: 本地推理 API —— 把微调后的考勤判定模型包成 HTTP 服务。

启动:  python scripts/serve.py            # 默认 :8100
自测:  python scripts/serve.py --selftest
"""
import os, sys, json, time, argparse
sys.stdout.reconfigure(encoding='utf-8')

LAB = os.environ.get('LLM_LAB_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(LAB, 'out', 'qwen2.5-0.5b-attendance-merged')
LORA_DIR = os.path.join(LAB, 'out', 'lora-attendance')
BASE = os.path.join(LAB, 'models', 'Qwen2.5-0.5B-Instruct')

SYSTEM = ('你是考勤核算助手。根据打卡流水判定考勤状态，只输出 JSON，不要任何解释。\n'
          '字段：date, weekday, status(达标/迟到/早退/迟到且早退/加班/缺卡/异常), '
          'late_minutes, early_leave_minutes, overtime_minutes, issues, note')

_cache = {}


def get_model(use_lora=True):
    key = 'lora' if use_lora else 'base'
    if key in _cache:
        return _cache[key]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if os.path.isdir(MERGED):
        path = MERGED
        tok = AutoTokenizer.from_pretrained(MERGED, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
        if use_lora and os.path.isdir(LORA_DIR):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, LORA_DIR).merge_and_unload()
    model.eval()
    _cache[key] = (model, tok)
    return _cache[key]


def judge(record_text):
    import torch
    model, tok = get_model()
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': record_text}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors='pt')
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    import re
    m = re.search(r'\{.*\}', raw, re.S)
    parsed = None
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            pass
    return {'result': parsed, 'raw': raw.strip(), 'latency_ms': int((time.time() - t0) * 1000)}


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    class Req(BaseModel):
        record: str

    app = FastAPI(title='Attendance Judge API', version='1.0.0')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': 'qwen2.5-0.5b-attendance'}

    @app.post('/v1/judge')
    def api_judge(req: Req):
        return JSONResponse(judge(req.record))

    return app


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8100)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    if a.selftest:
        r = judge('请判定以下考勤记录：\n张伟 2026-09-20(周三) 打卡记录：08:47、18:31，共 2 次。')
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        import uvicorn
        uvicorn.run(build_app(), host='127.0.0.1', port=a.port)
