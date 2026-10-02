# CLAUDE.md

本仓库是「考勤规则指令微调」实验：用硬编码考勤规则引擎当 teacher 自动生成指令数据，对 Qwen2.5-0.5B-Instruct 做 LoRA 微调，把确定性规则判定蒸馏成小模型的泛化判断，并在无 GPU（CPU fp32）环境下跑通「数据生成 → 编码掩码 → 微调 → 前后评测 → 权重合并 → FastAPI 服务化」全链路。

## 技术栈

- Python，无依赖清单文件（requirements/pyproject 均不在版本控制内）
- `torch`（本机为 `torch 2.13.0+cpu`，无 CUDA）、`transformers`、`peft`（v0.21.0）、`fastapi`、`uvicorn`
- 基座模型：`models/Qwen2.5-0.5B-Instruct/`（本地目录，由 HuggingFace 下载得到，**未纳入 git**）
- 注意：`trl` 出现在 README 依赖列表中，但仓库脚本实际未 import，只用 transformers Trainer

## 目录结构

```
llm-lab/
├── scripts/
│   ├── gen_dataset.py    # 规则引擎 → 指令数据集（9 类场景共 175 条，随机种子 20260916）
│   ├── smoke_test.py     # 基座模型冒烟：加载耗时、参数量、CPU 吞吐
│   ├── train_lora.py     # LoRA 微调（含 prompt 段 label 掩码），输出 out/lora-attendance/
│   ├── eval_lora.py      # 基座 vs 基座+LoRA 四指标对比评测，明细写 out/eval_compare.json
│   ├── merge_lora.py     # merge_and_unload() 合并权重 → out/qwen2.5-0.5b-attendance-merged/
│   └── serve.py          # FastAPI /v1/judge 推理服务（默认 :8100，--selftest 走本地判定）
├── data/
│   ├── train.jsonl       # 158 条（chat 格式：system/user/assistant，assistant 为 JSON 字符串）
│   └── valid.jsonl      # 17 条留出集
├── docs/                 # 链路架构图 lora-pipeline.png / .workflow.html / .workflow.json
├── models/               # 基座模型权重（gitignore，体积过大）
└── out/                  # 训练产物：lora-attendance/（适配器 33.6 MB + checkpoint-20/40）、merged 模型（gitignore）
```

## 安装 / 运行

模型 `Qwen/Qwen2.5-0.5B-Instruct` 需自行下载到 `models/Qwen2.5-0.5B-Instruct/`（未包含在仓库，`models/` 与 `out/` 均在 .gitignore 中）。

```bash
# 1. 生成指令数据集（幂等，固定种子）
python scripts/gen_dataset.py

# 2. 基座冒烟测试
python scripts/smoke_test.py

# 3. LoRA 微调（CPU fp32，3 epoch 约 10 分钟）
python scripts/train_lora.py

# 4. 前后对比评测
python scripts/eval_lora.py

# 5. 合并权重（脱离 PEFT 独立部署）
python scripts/merge_lora.py

# 6. 起推理服务 / 自测
python scripts/serve.py --selftest
python scripts/serve.py            # uvicorn 监听 127.0.0.1:8100，POST /v1/judge {"record": "..."}
```

仓库无 pytest 等测试框架；验证手段即 `smoke_test.py`（加载+生成冒烟）与 `serve.py --selftest`（端到端判定自测）。

## 关键约定与坑点

- **路径全部硬编码**：各脚本顶部 `LAB = r'C:\Users\intpj\WorkBuddy\2026-09-16-11-35-28\llm-lab'`（Windows 绝对路径），换机器/目录需同步改所有脚本；`gen_dataset.py` 是例外，用相对 `__file__` 定位 data 目录。
- **无 GPU 约束是本实验设计前提**：0.5B 全量微调需 ~8 GB 内存（权重 2 + 梯度 2 + Adam 状态 4），可用内存 9.2 GB 跑不动；LoRA r=16 覆盖 7 个线性层（q/k/v/o + gate/up/down），可训练参数 8.79M ≈ 1.75%。
- **label 掩码**：`train_lora.py` 的 `encode_sample()` 把 prompt 段 label 置 -100，只在 assistant 段算 loss（避免学「怎么问」并防止梯度信号被输入 token 稀释）；padding 也用 -100 填充 labels。
- **必须 `save_strategy='no'`**：`save_total_limit` 触发的旧 checkpoint 删除在受限沙箱里会被安全守卫拦截、中断训练进程（train_lora.py 第 99-102 行注释）。out/lora-attendance/ 里的 checkpoint-20/40 是历史产物，非当前训练配置产生。
- **GQA 注意**：Qwen2.5-0.5B 有 14 query 头 : 2 KV 头（7:1），k_proj/v_proj 输出维度 128 而非 896，算 LoRA 参数量时别套错。
- **评测结论**（17 条留出集，见 README）：基座 JSON 可解析率 100% 但业务判定 0%；微调后 status 正确率 58.8%、全字段 35.3%。格式遵从靠 prompt 即可，业务规则必须微调；数值回归（分钟数）对 0.5B 模型仍困难，方向是下沉回规则引擎。
- **数据集随机生成但种子固定**（20260916），重新跑 `gen_dataset.py` 会覆盖 `data/` 里已版本控制的文件，内容应一致。
- LICENSE 为 MIT。
