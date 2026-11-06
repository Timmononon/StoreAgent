# StoreAgent

知识库优先的 AI Agent 平台，前端名称为「知识助手」。支持文档问答、可恢复执行、
人工确认（HITL）和 RAG 效果评测。

## 核心能力

- **知识库问答**：文档版本化入库，Dense + BM25 混合检索、Rerank 与来源引用。
- **可恢复执行**：Thread / Run / Event 持久化，支持流式重连、取消和 HITL 暂停恢复。
- **Skill / Tool**：知识检索、Web Research、管理员只读 SQL 与自定义 HTTP Tool。
- **模型与评测**：管理模型角色分配，为 Run 和 Evaluation Job 冻结模型快照，查看评测报告与趋势。
- **权限与审计**：用户隔离、角色权限与工具审批；可选能力需要单独配置。

**技术栈**：Python 3.12+ · FastAPI · LangGraph · PostgreSQL · Redis · Milvus · Vue 3 · TypeScript

## 快速启动

需要 Python 3.12+、uv、Node.js 22.12+、npm 10+、Docker Compose。
以下命令在仓库根目录执行，使用 Bash（Windows 建议 WSL）。

### 1. 配置环境

```bash
cp .env.example .env
```

编辑 `.env`：

- 配置 `ARK_API_KEY`、`BASE_URL`、`MODEL`、`FAST_MODEL`、`GRADE_MODEL`、`EVALUATION_MODEL`。
- 将 `JWT_SECRET_KEY` 替换为至少 32 字符的随机值。
- 首次注册管理员时设置 `ADMIN_INVITE_CODE`；留空将禁用公开管理员注册。

模型配置用于首次初始化，之后通过管理界面的「模型中心」维护。
其余配置见 [.env.example](.env.example)。

### 2. 安装与初始化

```bash
docker compose up -d
uv sync --frozen

cd frontend
npm ci
npm run build
cd ..

uv run --frozen alembic upgrade head
uv run --frozen python -m backend.tools.registry_cli validate
```

### 3. 启动应用

```bash
bash scripts/start.sh
```

启动器统一管理 API、索引 worker 和 RAG 评估 worker；Compose 只启动基础设施。
API 与索引 worker 需共享持久化的 `UPLOAD_DIR`。

- 应用：[http://127.0.0.1:8000](http://127.0.0.1:8000/)
- API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 就绪检查：[http://127.0.0.1:8000/health/ready](http://127.0.0.1:8000/health/ready)

前端开发可另开终端执行 `cd frontend && npm run dev`，访问端口 3000。
生产配置、迁移和三进程托管见[部署指南](docs/runbooks/deployment.md)。

## 项目结构

| 目录                      | 内容                                   |
| ------------------------- | -------------------------------------- |
| `backend/`                | API、Agent、RAG、持久运行与后台 worker |
| `frontend/`               | Vue 前端与管理控制面                   |
| `contracts/`              | 跨端协议与生成类型的 Schema 真源       |
| `evals/rag/`              | 评测数据集、baseline 与质量 Gate       |
| `tests/`、`frontend/e2e/` | 后端测试与浏览器回归                   |
| `docs/`                   | 架构决策与运维指南                     |

## 详细文档

- **开发与部署**：[部署指南](docs/runbooks/deployment.md) · [仓库质量门禁](docs/runbooks/repository-quality-gates.md) · [前端质量门禁](docs/runbooks/frontend-quality-gates.md)
- **知识库与评测**：[索引 worker](docs/runbooks/persistent-indexing-worker.md) · [RAG 评测](evals/rag/README.md)
- **能力配置**：[Skill / Tool](docs/runbooks/skill-tool-registry.md) · [SQL Assistant](docs/runbooks/sql-assistant.md)
- **架构**：[领域术语](CONTEXT.md) · [架构决策](docs/adr/)
