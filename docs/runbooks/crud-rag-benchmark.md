# CRUD-RAG：100 题三组对比

当前默认基准已切换到 [MultiHop-RAG](multihop-rag-benchmark.md)，以减少全量索引成本。本流程保留供研究使用；运行前应接受下述 91,135 篇的索引规模。

所有命令从仓库根目录执行。数据准备可离线重建；导入、索引和真实评测需要单独启动基础设施。三组复用正式的 Document Publication、Index Job、Dataset、Evaluation Job、Worker 和生产 RAG Interface。

## 固定数据与抽样边界

来源是 [IAAR-Shanghai/CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG/tree/1aace383994e1f68efa12cf2a8e2dadfb4102ceb)，revision 固定为 `1aace383994e1f68efa12cf2a8e2dadfb4102ceb`。保留上游的使用和许可要求。下载包含一个 QA 文件和 48 个语料分片；先重组字节分片再按 UTF-8 解码，每条原始文本生成一个 HTML Document。问答、参考答案和 `thoughts` 不进入知识库。

当前固定版本实际读出 91,965 条原始记录，按规范化正文去重后为 **91,135 篇**，不要把论文中的语料数量当作本次索引数量。QA 三类共 2,394 题；仅使用参考新闻能严格、唯一映射到原始语料的 477 题，另外 1,917 题写入 `alignment_audit.json`。不使用模糊匹配补齐 gold，不根据模型分数选题。

默认固定 seed `20261030`，最终 **100 题**：单文档 25、双文档 38、三文档 37。先按事件和共享来源分组，再划分 dev 30 / test 70，两个集合不共享事件或 gold 文档。每类容量不足时由其余类别补足。这个来源可对齐子集适合项目内可重复对比，不能宣称代表 CRUD-RAG 全量成绩。修改系统时使用 dev；`--split all` 是一次 100 题的整体对比，不是盲测。

版本化的 [manifest](../../evals/rag/crud_rag_100_v1.manifest.json) 保存源文件 SHA-256、抽样 ID、语料和 Dataset fingerprint。大体积原始数据、语料和运行产物保留在被忽略的 `.artifacts/crud-rag/`，不提交到仓库。

## 下载和准备

使用已经安装的 Python 环境；CPU/GPU extras 的安装方法见 [README](../../README.md)。这里用 `--no-sync`，避免运行脚本时改变当前环境。

```bash
uv run --no-sync python scripts/crud_rag.py download
uv run --no-sync python scripts/crud_rag.py prepare --count 100 --dev-count 30
uv run --no-sync python scripts/evaluate_rag.py validate --dataset .artifacts/crud-rag/prepared/dataset.json
```

只有 `download` 显式联网；`prepare` 不调用模型。默认参数必须与提交的 manifest 完全一致；已有 `manifest.json` 的输出目录会拒绝覆盖，重建时使用新的 `--output` 目录。自定义抽样可设置 `--count`、`--dev-count`、`--seed`，或用 `--expected-manifest` 验证自定义的冻结版本。

准备目录包含 `corpus.jsonl`、`dataset.json`、`dataset_dev.json`、`dataset_test.json`、`alignment_audit.json` 和 `manifest.json`。后续步骤会验证语料和题集校验和，拒绝未同步的修改。

## 在线前置条件

在独立评测环境中配置 PostgreSQL、Redis、Milvus、上传目录、缓存目录、Tenant 和模型；所有进程使用同一组配置。避免把评测语料混入业务知识库。准备可用且已预热的 Embedding，并在本地 `.env` 配置 Reranker 的 `RERANK_MODEL`、`RERANK_BINDING_HOST` 和服务需要的密钥。四个模型角色从持久化 Model Control 读取，环境变量只负责首次种子。

```bash
uv run --no-sync python scripts/crud_rag.py preflight
```

该命令只检查配置、端口、数据库迁移和 Model Snapshot 就绪状态，输出 `.artifacts/crud-rag/preflight.json`；**模型调用数为 0**，未就绪返回退出码 1。它不证明 Provider 协议、Embedding 缓存或 Reranker 实际可用，这些需要后续真实执行验证。

只对独立评测数据库运行迁移，并按 [持久化 worker Runbook](persistent-indexing-worker.md) 启动 API、Indexing Worker 和 Evaluation Worker：

```bash
uv run --no-sync alembic upgrade head
```

本功能新增 `0019_rag_experiments`，为 Evaluation Job 增加可空的实验配置；已有普通评测保持原语义。迁移后需有已存在的管理员用户和四角色 Assignment。

## 导入与索引

```bash
uv run --no-sync python scripts/crud_rag.py ingest --username admin --knowledge-base-name "CRUD-RAG benchmark"
```

这会把全部 91,135 篇语料转成转义 HTML，经 `DocumentPublication.submit_batch` 每批最多 100 篇提交持久 Index Job。`ingestion.json` 的 `complete=true` 只代表提交完成；实际解析、分块、Embedding 和版本原子发布由现有 Indexing Worker 执行。中断后可用同一条命令重新提交，复用正式的内容/构建幂等机制。知识库的名称、语料或配置变更应使用新目录和新评测知识库。

100 题限制控制问答和 Judge 次数，不会减少全语料索引的 Embedding 成本。索引完成后保留该知识库的只读状态，直到三组执行结束。

## 创建三组任务

```bash
uv run --no-sync python scripts/crud_rag.py enqueue --username admin --split all
```

默认一份 100 题 Dataset、三组 Evaluation Job，共 300 次 Case 执行。需要单独开发或盲测时，改为 `--split dev`（30 × 3）或 `--split test`（70 × 3），不要在调参后把同一 test 反复使用的结果称为盲测。

| 组 | 召回 | Reranker | 候选数 / 返回数 |
| --- | --- | --- | --- |
| dense | Dense | 关闭 | 30 / 10 |
| hybrid | Dense + BM25 + RRF | 关闭 | 30 / 10 |
| hybrid_rerank | Dense + BM25 + RRF | 开启 | 30 / 10 |

三组保留相同的 Auto-merge、复杂问题分解、Grader、重写策略、Answer/Evaluator 模型及证据预算，只改变召回模式和 Reranker。这里的 Dense 指召回方式，整体仍经过生产 RAG 图。实验配置随 Job 冻结并由 Case 的 request context 持有，不通过修改进程全局变量切组。

入队前验证全量文档内容哈希与就绪状态，三组共享一次冻结的 Model Snapshot、Dataset、corpus/index identity 和固定参数，并在同一事务创建；缺少重排配置不会遗留前两组任务。Worker 开始/结束及重领时检查索引、配置漂移，拒绝继续不可比的实验。

任务 ID 写入 `jobs_all.json`（其他 split 同理）。已有收据会拒绝再次提交；可在工作台查询进度、失败 Case 或取消，继续使用现有 lease/fencing 和恢复流程。失败后的重跑应保留旧报告，并使用新的准备目录/任务收据。

## 报告

```bash
uv run --no-sync python scripts/crud_rag.py compare --jobs .artifacts/crud-rag/prepared/jobs_all.json --output .artifacts/crud-rag/comparison
```

三组全部结束后导出 JSON 和 Markdown。不同题集、源码、模型、语料、索引或固定参数会拒绝比较；出现 Hybrid/Rerank 降级、Provider 故障时 `valid=false`，同时保留全部分数和失败计数，不能删除失败题来提高结果。该报告是受控对比，`release_gate=false`；它不替代现有发布 Gate，不能与普通 baseline 混用。

主要观察 Document Recall@5/10、All sources hit@5/10、answer correctness、groundedness、completeness、RAG P50/P95、Judge 耗时、Provider failure 和 retrieval fallback rate。文档召回是在最终 top-K chunk 结果中匹配、去重 gold 文档；不是 top-K 独立文档检索。无精确 gold chunk 时 chunk ranking 指标保持 unavailable。

答案指标复用项目现有结构化 Evaluator Judge，不是 Ragas 官方指标；仍需要人工抽检。Answer 与 Judge 使用相同预算下的最终证据。RAG 耗时包含图执行和回答生成，另报 Judge 耗时，整体耗时继续保留。当前不计算 token 金额或推算费用，不以请求数量冒充计费数据。

## 离线验证范围

```bash
uv run --no-sync pytest -q tests/test_crud_rag_benchmark.py tests/test_rag_benchmark_jobs.py
uv run --no-sync python scripts/generate_rag_eval_schemas.py --check
```

测试使用本地合成语料、SQLite 和受控 Provider，验证分片重组、确定性抽样、来源隔离、篡改拒绝、原生 Index Job 幂等提交、三组原子创建、100 × 3 个持久 Case、并发 context 隔离和漂移拒绝。通过这些检查只证明流程和契约；真实 PostgreSQL/Milvus、Embedding、Reranker、答案质量和延迟需要单独在线验证。

当前 Skill Registry 的安全文件读取要求 `O_DIRECTORY/O_NOFOLLOW`，原生 Windows 不支持这些标志。涉及整个应用启动或真实 Agent/HITL 的检查应在安装项目依赖的 Linux/WSL 环境执行；不要为了本机测试绕过该安全校验。
