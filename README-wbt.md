# Current extension status — 2026-10-25
本项目按用户新指令使用物理 GPU 2；以 config/gpu.env 为准。旧文档中 GPU 3 是历史配置。

新流程支持 --problem / --paper，运行方式见 research_extension/README.md。
电池热失控预警端到端测试使用 runs/battery-thermal-warning-v4/；此前 v1/v2/v3 为保留的调试记录。
所有代码、依赖、模型、缓存、附件和输出仍限制在 wbt 根目录及虚拟环境。

本地模型建议与系统预设评测规范分开保存：model_final_plan.json 是模型一次修订的原文；final_plan.json 是应用明确评测契约后的交付；protocol_overrides.json 记录改动和来源。事件标签、误报分母、按电芯/试验划分及统计单位由协议约束，不冒充模型自发生成的科学发现。

# wbt local ResearchAgent reproduction

All project data, environment packages, models, caches, logs, and outputs must stay under `/home/think/mnt/cyh/610/wbt/`. Use the `wbt` Python environment for every command. System drivers, global packages, and other project folders must not be modified.

## Enter

```bash
source /home/think/mnt/cyh/610/wbt/activate.sh
```

Python 3.10 virtual environment: `envs/wbt/`.

## GPU and Ollama

- GPU selection: `config/gpu.env`; initially the UUID of physical GPU 3 (RTX 4090).
- Active runtime: `runtime/ollama-v0.9.6/`, selected in `config/runtime.env`.
- Ollama 0.40.10 was downloaded but rejected the installed NVIDIA 535 driver. Version 0.9.6 supports the existing driver and has passed actual GPU inference validation. No system driver changes were made.
- Endpoint: `127.0.0.1:11435`; OpenAI-compatible URL: `http://127.0.0.1:11435/v1`.
- Original DeepSeek-R1-0528-Qwen3-8B Q4_K_M weights remain in `models/ollama/`.
- To change GPUs later: edit the UUID in `config/gpu.env`, then run `scripts/restart-ollama.sh` while this project has no active inference request. Do not stop other services.
- LLM parameters: `config/llm.env`. Context: 32768, parallel requests: 1, KV cache: q8_0, flash attention enabled.

## One paper, one round

```bash
source /home/think/mnt/cyh/610/wbt/activate.sh
./scripts/start-ollama.sh
./scripts/reproduce-one.sh /home/think/mnt/cyh/610/wbt/runs/up5-one-round-repeat
```

Use a new output directory for each run. Existing completed results are protected. The script prepares the cached input, records dependencies and code hashes, runs all three stages once, and validates the output before creating a report.

The prepared paper is UP5: Unbiased Foundation Model for Fairness-aware Recommendation (Corpus ID 258832356). Its real Semantic Scholar reference metadata and SPECTER v2 embeddings are cached in `cache/s2/`. Retrieval selects 10 of 48 candidates with abstracts and embeddings by the original cosine-similarity method, and selects 30 knowledge entities from the uploaded knowledge store.

## Outputs

`runs/up5-one-round/` contains `result.json`, `ideas.jsonl`, `report.md`, `input_context.json`, `stage_records.jsonl`, `llm_calls.jsonl`, `status.json`, `validation.json`, `gpu-evidence.json`, dependencies, and a code snapshot. The earlier incomplete format-debugging attempt is retained separately in `runs/up5-one-round-attempt1/`.

## Adaptations

Original generation prompts and five evaluation criteria per stage are preserved. The local client has configurable generation parameters, explicit errors and raw request/response logs. Markdown section labels are normalized. Reviews use Ollama JSON Schema output for the same review, feedback and 1–5 rating fields. Missing fields are retried and rejected if still incomplete. The original sentence-transformers cosine utility is implemented with the same CPU PyTorch normalized dot product; no replacement embedding model was introduced. CPU-only PyTorch is used, so the project Python code does not allocate other GPUs.

This run validates the full workflow with a substituted local 8B quantized model. Model self-ratings are not independent evaluations or evidence that original paper benchmark results were reproduced.
