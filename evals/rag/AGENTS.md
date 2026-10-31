# RAG Evaluation Assets AGENTS.md

本文件适用于 `evals/rag/`，继承根目录 `AGENTS.md`。评测实现位于 `backend/evaluation/` 和 `scripts/evaluate_rag.py`；RAG 实现变更还需读取 `../../backend/rag/AGENTS.md`。

## 资产角色

- `rag_smoke_v1.json`：人工标注的版本化 Dataset。
- `corpus/`：隔离测试知识库使用的受控语料。
- `offline_smoke_observations_v1.json`：验证评分 Contract 的脱敏静态 Observation。
- `gates_v1.json`：只接受 `contract_smoke` provenance 的离线门禁。
- `live_gates_v1.json`：只接受生产图 `live_rag` provenance 的门禁。
- `baseline_v1.json`：由 Dataset + offline Observation + Gate 确定性生成的 baseline。
- `schema/`：从后端 Pydantic 模型生成的 Dataset/Observation/Gate/Report JSON Schema。
- `crud_rag_100_v1.manifest.json`：固定外部来源和 100 题抽样；下载/语料/真实运行产物不提交，命令见 `../../docs/runbooks/crud-rag-benchmark.md`。
- `multihop_rag_100_v2.manifest.json`：默认外部基准，固定 609 篇原文、四类各 25 题，分层开发 40 / 测试 60；v1 保留历史，命令和共享语料边界见 `../../docs/runbooks/multihop-rag-benchmark.md`。

MultiHop 默认使用 `--design optimization` 的 A/B/C0/C1 四组：A/B/C0 冻结原始拼接策略，C1 使用融合/原问题重排/来源选择。原 `retrieval` 三组标识含义保持不变。不能混合两种设计，不能用开发集分数充当测试集优化前基线。

离线 baseline 只证明数据契约、指标和 Gate 可重建，不代表真实模型、Milvus、Document Version 或生产知识库质量。

## 不变量

- Dataset 内容变化会改变 fingerprint；旧 Observation 和 baseline 必须被拒绝，不能通过 override 默认混用。
- baseline 比较要求相同 Dataset fingerprint、RAG source fingerprint、corpus/profile/index 等可比 provenance；显式 mismatch override 只能用于调查，不能成为发布门禁。
- 不手工修改 Observation 或 baseline 来提高指标。静态 Observation 是受控测试夹具；live Observation 必须由 Adapter 运行正式 RAG/HITL 产生。
- Report/Observation 不保存 chunk 正文、endpoint、Secret、完整 provider payload、原始异常或私有推理。
- 离线测试不得联网、下载模型或要求 Provider credential；live 运行必须在独立进程和隔离索引执行。
- 评测 Adapter 观察生产 Interface，不复制另一套检索、答案或 HITL 逻辑。

## 常用命令

```bash
uv run python scripts/evaluate_rag.py validate \
  --dataset evals/rag/rag_smoke_v1.json

uv run python scripts/generate_rag_eval_schemas.py --check

uv run python scripts/evaluate_rag.py score \
  --dataset evals/rag/rag_smoke_v1.json \
  --observations evals/rag/offline_smoke_observations_v1.json \
  --gates evals/rag/gates_v1.json \
  --baseline evals/rag/baseline_v1.json \
  --report /tmp/rag-report.json \
  --markdown /tmp/rag-report.md \
  --fail-on-regression
```

重建 committed baseline：

```bash
uv run --no-sync python scripts/evaluate_rag.py score \
  --dataset evals/rag/rag_smoke_v1.json \
  --observations evals/rag/offline_smoke_observations_v1.json \
  --gates evals/rag/gates_v1.json \
  --report /tmp/rebuilt-rag-baseline.json \
  --fail-on-regression
cmp /tmp/rebuilt-rag-baseline.json evals/rag/baseline_v1.json
```

真实发布检查使用 `run`、显式 `profile-id`/`index-id`、受控模型和隔离 Milvus/Document Version。该 CLI 一次进程只运行一个 profile；配置变化后启动新进程，不在运行中热切换。持久化三组对比使用 Job 冻结的实验配置和 request context，不修改全局 Provider 或检索配置。

## 修改 Dataset

- Case ID 稳定且唯一；问题、期望 route/outcome、HITL、Evidence/coverage 期望保持结构化。
- 新 Case 应补足缺失切片，而不是为当前实现量身定制答案措辞。
- 测试 prompt injection 时区分“检索到安全说明”和“端到端没有泄密”；不要以 retrieval-only case 宣称完整 Guardrail。
- corpus 使用可提交、无 Secret、许可明确的受控文本；真实外部数据固定来源/revision/hash，不隐式下载。
- 规模扩大时保持确定性排序，避免时间、随机数和机器路径进入 fingerprint。

## 修改 Gate 或指标

- Gate 是发布政策，调低阈值需要可审查的质量理由，不能只因 PR 失败。
- 指标方向、聚合、slice 和 unsupported/conflict 语义变化属于评测 Contract 变化，应更新 ADR、schema、测试和 baseline。
- 评分器测试必须覆盖边界值、缺失字段、fingerprint mismatch、provenance mismatch 和 deterministic rendering。

## Code Review Rules

### 手工美化评测结果

- 阻止直接编辑 Observation/baseline 数值、删除失败 Case 或降低 Gate 只为变绿。
  安全路径：修复实现或标注事实；通过命令确定性再生工件并解释政策变化。

### Offline 冒充 Live

- 阻止把 `contract_smoke` 报告写成真实 RAG/模型质量结论。
  安全路径：保留 provenance，发布前运行隔离 `live_rag` profile。

### 不可比 baseline

- 阻止跨 Dataset/source/corpus/model/index fingerprint 比较后宣称回归或提升。
  安全路径：只比较冻结身份相同的报告，或明确作为非门禁实验。

### 评测复制生产逻辑

- 阻止在 Adapter 中另写检索/重写/HITL 实现。
  安全路径：调用正式生产 seam，投影为安全 Observation。
