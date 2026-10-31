# MultiHop-RAG：609 篇原文、100 题受控对比

所有命令从仓库根目录执行，使用已经安装的环境，`--no-sync` 保留当前 CPU/GPU 依赖。当前默认外部基准采用此数据集。

## 数据和划分边界

来源为 [MultiHop-RAG 官方仓库](https://github.com/yixuantt/MultiHop-RAG)和 [Hugging Face 数据集](https://huggingface.co/datasets/yixuantt/MultiHopRAG)，固定 revision `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`。保留 ODC-BY 许可及来源归属。

完整语料含 **609 篇**新闻，`corpus.json` 为 6,785,567 字节；上游有 **2,556 题**。默认 seed `20261030`，从 inference、comparison、temporal、null 四类各抽 25 题，共 **100 题**，划分 dev 40 / test 60（每类分别 10 / 15 题）。两个集合的 Case ID 不重叠，但使用同一完整语料，gold 文档可以重叠；不能将其称为来源隔离或全量官方成绩。

文章 URL 严格、唯一映射为 Document canonical name；非 null 题保留全部 2–4 篇 gold 来源，null 题预期拒答。索引内容只有原文和来源元数据，问题、参考答案、gold fact 不进入索引。提交的 [manifest](../../evals/rag/multihop_rag_100_v2.manifest.json)冻结源文件哈希、抽样、划分和 corpus/Dataset identity。

## 下载、准备和导入

```bash
uv run --no-sync python scripts/multihop_rag.py download
uv run --no-sync python scripts/multihop_rag.py prepare
uv run --no-sync python scripts/evaluate_rag.py validate --dataset .artifacts/multihop-rag/prepared-v2/dataset.json
uv run --no-sync python scripts/multihop_rag.py ingest --username admin
```

只有 `download` 显式联网，从固定 revision 下载两份 JSON；`prepare` 不调用模型。默认抽样必须匹配提交的 manifest；重建需使用新的 `--output`，已有 manifest 不覆盖。数据与真实运行产物保存在忽略目录 `.artifacts/multihop-rag/`。

导入复用 `DocumentPublication.submit_batch`、Catalog 内容/构建幂等及持久 Index Job，每批最多 100 篇。`ingestion.json` 的 `complete=true` 只表示全部提交。按[索引 Worker Runbook](persistent-indexing-worker.md)启动 GPU 环境的 worker，等待 609 篇都原子发布为 ready，再创建评测任务。中断后可重新执行同一导入命令；不要复制一套直接写 Milvus 的导入器。

## A/B/C0/C1 四组对比

配置、迁移和四角色 Model Assignment 的前提与 [CRUD-RAG Runbook](crud-rag-benchmark.md#在线前置条件)一致。API、Indexing Worker、Evaluation Worker 必须使用同一数据库、Milvus、共享上传目录和固定 Embedding revision。C0/C1 需要可用 Reranker。

```bash
uv run --no-sync python scripts/multihop_rag.py preflight
uv run --no-sync python scripts/multihop_rag.py enqueue --username admin --design optimization --split dev
uv run --no-sync python scripts/multihop_rag.py compare --jobs .artifacts/multihop-rag/prepared-v2/jobs_dev.json --output .artifacts/multihop-rag/comparison
```

preflight 只检查就绪条件，模型调用数为 0。缺少 Reranker 时不得把回退结果当作 C0/C1；整批入队失败且不创建部分任务。真实评测会产生模型费用，入队后由独立 Evaluation Worker 执行。

默认 `--design optimization` 使用下列四个冻结实验标识。A/B/C0 保持原始检索汇总策略，只有 C1 使用优化策略；模型、corpus/index、故障修复、每个子问题候选 30 / TopK 10 和提示词预算一致。

| 组别 | 实验标识 | 检索 | Synthesis 策略 |
| --- | --- | --- | --- |
| A | `dense_baseline` | Dense | `concatenate_v1` |
| B | `hybrid_baseline` | Dense + BM25/RRF | `concatenate_v1` |
| C0 | `hybrid_rerank_baseline` | Hybrid + 子问题 Reranker | `concatenate_v1` |
| C1 | `hybrid_rerank_optimized` | Hybrid + 子问题及原问题 Reranker | `source_priority_v1` |

`concatenate_v1` 按规划顺序拼接并去重，不做全局融合、二次重排或来源配额，最终列表可能超过 10；指标在前 10 内计算。`source_priority_v1` 对有界候选执行全局 RRF、原问题重排、来源优先选择，最终至多 10；裁减后按需复评原问题一次。两者的实际返回量和调用量不同，不能声称推理成本完全相同，答案/Judge 的字符预算保持不变。

开发阶段为 40 × 4 次 Case，最终测试为 60 × 4 = 240 次 Case（题目仍只有 60 个），每个 Case 可包含多次 LLM 调用。主要比较 C0−A 与 C1−C0；报告同时保留 B−A、C0−B、C1−A，正负差值如实输出。`--split all` 不保留测试盲测边界。

历史 `--design retrieval` 三组使用原 `dense/hybrid/hybrid_rerank` 标识及其既有共享优化策略，不重新解释历史 Job。新四组不能混入旧三组报告；它们通过同一 Service/Worker/生产图运行，策略由冻结实验名确定并绑定 request context，不进行全局切换。

enqueue 会逐项验证全部 609 篇当前内容哈希和 gold 覆盖，随后通过现有 Dataset/Evaluation Job service 原子创建整批任务。报告复用同一指标和漂移/降级拒绝规则，详见 [报告说明](crud-rag-benchmark.md#报告)。其中 Document Recall、All sources hit 检查多文档覆盖；null 切片同时观察拒答表现。答案指标使用项目现有 Judge，不是 Ragas 官方指标。

## 本地 GPU Reranker

可选的 `scripts/local_reranker.py` 提供与现有 HTTP Adapter 一致的 `/v1/rerank`。固定使用 `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，通过 Transformers 在 CUDA 上以 FP16 推理；只加载本地 safetensors，不执行远程模型代码、不退回 CPU。首次运行前显式下载该 revision 的配置、tokenizer 和 `model.safetensors`（约 2.3 GB），设置 `.env`：

```dotenv
RERANK_MODEL=BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
RERANK_BINDING_HOST=http://127.0.0.1:8091
RERANK_API_KEY=<本地生成的随机密钥>
RERANK_TIMEOUT_SECONDS=30
RERANK_MAX_CONCURRENCY=1
LOCAL_RERANK_MODEL_PATH=<该 revision 的本地缓存目录>
LOCAL_RERANK_BATCH_SIZE=4
LOCAL_RERANK_MAX_LENGTH=1024
```

```bash
uv run --no-sync python -m uvicorn scripts.local_reranker:create_app --factory --host 127.0.0.1 --port 8091 --no-access-log
uv run --no-sync python -m backend.workers.evaluation --concurrency 3
```

`GET /health` 返回实际模型标识、CUDA/FP16、batch、截断长度和已完成请求数。服务单次最多 30 篇、单篇 8,000 字符、合计 60,000 字符，默认每批 4 对、最多 1,024 tokens；分数经过 sigmoid，原始输入索引保持不变。失败返回 503，重试仍由调用端 ProviderExecutor 单独拥有。模型 revision、长度和精度在三组运行期间必须固定，不能中途改服务配置。

8 GB 显卡应避免多个进程重复加载 Embedding。可以对 API/Indexing/Evaluation 统一设置 `EMBEDDING_WARMUP_ON_START=false`，保留 `EMBEDDING_DEVICE=cuda`；模型仍在首次查询时加载 GPU。同一 Evaluation 进程的三个 claimer 共享 Provider Runtime/GPU Embedding，分别持有独立 Job lease/fence/request context。每组内部顺序处理 Case。

默认使用新目录 `prepared-v2` 和 v2 manifest，固定同一组 100 个问题、分层 40/60；v1 manifest 与旧目录保留历史记录。既有 609 篇语料无需重新向量化，同一知识库与语料 fingerprint 核对一致后可复用导入收据。开发完成后显式执行 `enqueue --split test`；不要在开发时运行或查看测试集结果。

## 开发集调优与运行留档

四组设计仅 C1 使用全局 RRF 融合各子问题的可用证据，默认候选 30、最终 10，并对原问题统一重排一次，保留有界候选的全部分数后再选择。最终优先不同文档，各取一个片段，剩余位置按相关性填充，同一文档仍可提供多个片段。A/B/C0 使用原始拼接去重；分支的召回、Auto-merging 和 Grader 语义保持一致。

选择使用实际 Evidence，不能仅调整评测 Observation 的排名。如果 C1 全局选择丢弃了候选且各分支都有可用证据，合并 Grader 检查原问题一次；这可能增加一次 Grader 调用。C1 另增加一次最多 30 对的本地 Rerank，需将新增延迟计入结果。空分支和故障仍保留覆盖缺口。

每轮使用新的 prepared 目录：将已验证的 `manifest.json`、`corpus.jsonl`、`dataset.json`、`dataset_dev.json`、`dataset_test.json` 和 `ingestion.json` 原样复制到新目录（保留相同 fingerprint），对新目录执行 `enqueue --prepared <新目录> --username admin --design optimization --split dev`。不要覆盖旧 `jobs_dev.json` 或重导语料。用同一版本运行整批任务，完成后以 `compare --jobs <新目录>/jobs_dev.json --output <本轮报告前缀>` 导出。保存分支、源码 fingerprint、Job IDs、固定配置、失败和分母；失败不得删除或补分。

先前报告保留为开发基准记录；跨源码实验不能当作同源码发布 Gate。独立 LLM 规划会带来方差，不能将所有 B/C 差异直接归因于 Reranker。开发集结果确定后再固定实现并运行一次测试集，60 题测试集不用于挑选参数。

## 离线验证

```bash
uv run --no-sync pytest -q tests/test_local_reranker.py tests/test_evaluation_worker_cli.py tests/test_multihop_rag_benchmark.py tests/test_crud_rag_benchmark.py tests/test_rag_benchmark_jobs.py
uv run --no-sync python scripts/generate_rag_eval_schemas.py --check
```

离线测试覆盖确定性抽样、严格 URL gold 映射、答案隔离、checksum/manifest 漂移拒绝，以及三组/四组持久 Case、request context 策略隔离、批次原子性和受控报告。它们不证明真实答案质量；缺少 Reranker 时只完成导入和离线流程验证。
