# 📚 文献阅读助手

上传 PDF 文献（可多篇），让大模型帮你**精读摘要、提取陌生概念并绘制概念图谱、围绕概念自测出题、多轮问答**。知识库持久化保存，重启后无需重新上传。

## 🚀 快速开始

### 第 0 步：配置你的 API 密钥（必做！）

> **⚠️ 本项目不包含 API 密钥**——模型能力由 [智谱AI开放平台](https://open.bigmodel.cn) 提供，
> 每位使用者需要自己申请（新用户有免费额度）。没配密钥的话，启动应用后会显示配置引导页。

1. **申请密钥**：打开 [open.bigmodel.cn](https://open.bigmodel.cn) → 注册/登录 → 「控制台」→「API keys」→ 复制密钥
2. **填入密钥**：在项目根目录，把模板文件 **`.env.example`** 复制一份并重命名为 **`.env`**，然后把你的密钥粘贴进去：

   ```bash
   # macOS / Linux
   cp .env.example .env
   # Windows（cmd）
   copy .env.example .env
   ```

   打开 `.env`，改成这样（等号右边换成你自己的密钥）：

   ```ini
   ZHIPU_API_KEY=粘贴你刚复制的密钥
   ```

3. `.env` 已被 `.gitignore` 排除，**你的密钥只留在自己电脑上，不会被提交到 GitHub**。

### 第 1 步：安装依赖

```bash
pip install -r requirements.txt
```

Python 3.9+；推荐用虚拟环境（`uv venv` 或 `python -m venv .venv`）。

### 第 2 步：启动

```bash
python run_app.py run app.py --server.port 8501
```

> 💡 请用 `run_app.py` 而不是直接 `streamlit run app.py`：启动器会给 Streamlit 的静态服务
> 打个补丁，让自托管的 mermaid/ELK 图谱引擎（`static/` 目录）能被正确加载，概念图谱秒出图；
> 直接启动时图谱会回退走 CDN，网络不好会很慢。

浏览器打开 http://localhost:8501 ，在左侧上传 PDF 即可开始。

## ✨ 功能

| 功能 | 说明 |
| --- | --- |
| 📋 论文速读 | 等距采样全文，生成四段式结构化摘要：逻辑关系彩色标签 + 关键术语同色下划线 |
| 📖 问答 | 基于文献内容的多轮对话（BM25 + 向量混合检索），支持按文献范围过滤 |
| 🧠 概念图谱 | 自动提取对初学者陌生的概念及相互关系，渲染成可缩放/平移/悬停看定义的交互式图谱，支持导出 PNG |
| 📝 自测出题 | 选一个概念，一键生成一组 5 道单选题（含相邻概念与关系的考察），交卷后统一判分 + 逐题解析 |
| 💾 结果持久化 | 摘要/概念图谱按文献范围自动保存，重启应用后直接查看；文献重新上传时自动失效 |

## 🗂 项目结构

```
├── app.py               # Streamlit 界面
├── rag_core.py          # 核心流水线：入库/检索/提取/摘要/出题
├── run_app.py           # 启动器（静态资源补丁）
├── .env.example         # API 密钥模板 ← 复制成 .env 并填入你的密钥
├── requirements.txt
├── static/mermaid/      # 自托管的 mermaid + ELK 布局引擎
├── chroma_db/           # 向量库（运行后自动生成，不入库）
└── artifacts/           # 摘要/图谱存档（运行后自动生成，不入库）
```

## ❓ 常见问题

- **启动后一直显示配置引导页**：说明 `.env` 没建好或 `ZHIPU_API_KEY` 还是模板占位文字，回到「第 0 步」检查。
- **改了 `.env` 还是提示没密钥**：改完需要重启应用（Ctrl+C 后重新运行启动命令）。
- **图谱加载慢**：确认是用 `python run_app.py run app.py` 启动的（见第 2 步说明）。
