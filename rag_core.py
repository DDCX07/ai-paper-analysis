# rag_core.py
# 核心RAG流水线：PDF加载 -> 分块 -> 向量化(智谱embedding-3) -> Chroma(持久化) -> 检索问答(GLM)
# 扩展模块：口语化问答、概念与关系提取、自测出题、多文献管理与按来源过滤检索
import json
import hashlib
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pdfplumber
import jieba
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- 配置 ---
# 厂商无关：接口地址/密钥/模型名全部从.env读取，不写死任何一家。
# 默认预设为智谱GLM；换DeepSeek/Kimi/通义/OpenAI等只需改.env（示例见.env.example）。
load_dotenv()  # 读取项目根目录的 .env 文件

# 密钥：LLM_API_KEY优先，兼容旧名ZHIPU_API_KEY
API_KEY = os.environ.get("LLM_API_KEY", "").strip() or os.environ.get("ZHIPU_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError(
        "未找到API密钥：请在项目根目录创建 .env 文件，填入 LLM_API_KEY=你的密钥"
        "（接口地址/模型名等完整示例见 .env.example）"
    )

# 对话模型三要素。LLM_API_TYPE决定接口协议：
#   anthropic（默认，智谱Anthropic兼容端点）/ openai（OpenAI兼容端点，多数厂商用这个）
LLM_API_TYPE = os.environ.get("LLM_API_TYPE", "anthropic").strip().lower()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://open.bigmodel.cn/api/anthropic").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-5.3-flash").strip()
# 输出上限。智谱GLM强制思考，thinking块要占两三千token，必须给足余量；
# 其他厂商若报max_tokens超限（如部分模型上限8k），可在.env里调小
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "16384"))

# Embedding模型（走OpenAI兼容接口，允许用另一家厂商）。
# EMBEDDING_API_KEY不填则复用LLM_API_KEY
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "embedding-3").strip()
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "").strip() or API_KEY

# 学术文献用较大分块效果更好（参考DeepSeek方案中的调优建议）
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4
# 概念提取时最多送入LLM的文本块数（控制耗时与token成本）
EXTRACT_MAX_CHUNKS = 6
# 概念提取的并行线程数
EXTRACT_WORKERS = 3

# 向量库持久化目录：应用重启后知识库仍在，无需重新上传解析
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# 调试日志：排查GLM结构化输出失败时，打印具体是哪个文本块、失败原因。
# 环境变量 RAG_LOG=DEBUG 可看全部细节；默认 WARNING。
logger = logging.getLogger("rag_core")


def _get_embeddings() -> OpenAIEmbeddings:
    """Embedding模型，走OpenAI兼容端点（默认智谱embedding-3，可换厂商）。

    check_embedding_ctx_length=False 很关键：关闭后不做tiktoken本地分词
    （那是OpenAI专用逻辑），直接把整段文本发给接口。
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        check_embedding_ctx_length=False,
    )


def _get_llm():
    """对话模型。默认智谱GLM走Anthropic兼容端点；.env里 LLM_API_TYPE=openai
    时改走OpenAI兼容端点（DeepSeek/Kimi/通义/OpenAI等）。

    智谱专用逻辑（anthropic分支）：glm-5.3-flash始终思考，传disabled会报1210错误，
    只能开思考并给最小budget（1024）压低延迟。
    注意：thinking块本身能占到两三千token，max_tokens必须给足余量，
    否则思考没结束就截断，text块为空。
    返回内容里可能多一个thinking块，各调用方统一用_msg_text取纯文本，不受影响。
    """
    common = dict(
        model=LLM_MODEL,
        api_key=API_KEY,
        base_url=LLM_BASE_URL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.2,
        timeout=120,
    )
    if LLM_API_TYPE == "openai":
        return ChatOpenAI(**common)
    return ChatAnthropic(**common, thinking={"type": "enabled", "budget_tokens": 1024})


def _source_filter(sources):
    """把文献名列表转成Chroma的where过滤条件；None表示不过滤（检索全部）。"""
    if not sources:
        return None
    return {"source": {"$in": list(sources)}}


# --- PDF加载（pdfplumber：表格转文本 + 扫描版检测） ---

def _table_to_text(table) -> str:
    """把pdfplumber提取的表格（二维列表）转成竖线分隔的文本行，供向量化与LLM阅读。"""
    rows = []
    for row in table:
        cells = [re.sub(r"\s+", " ", str(c)).strip() if c is not None else "" for c in row]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def load_pdf_pages(pdf_path: str, progress_callback=None) -> list[Document]:
    """用pdfplumber逐页提取文本和表格。

    表格会被转成 | 分隔的文本行（LLM可读），公式在PDF文本层中通常已含unicode符号，
    保留原样即可。整篇几乎没有文本层时判定为扫描版，抛出明确错误。
    """
    def report(frac, msg):
        if progress_callback:
            progress_callback(frac, msg)

    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            report(i / n_pages * 0.9, f"解析第 {i + 1}/{n_pages} 页...")
            text = page.extract_text() or ""
            # 表格检测：pdfplumber的lines/rects策略对学术文献表格效果好
            for table in page.extract_tables():
                t = _table_to_text(table)
                if t:
                    text += "\n\n[表格]\n" + t
            docs.append(Document(
                page_content=text.strip(),
                metadata={"page": i},
            ))

        # 扫描版检测：所有页加起来几乎没有文本层
        total_chars = sum(len(d.page_content) for d in docs)
        if total_chars < 100 * n_pages:
            raise RuntimeError(
                "检测到扫描版PDF（没有可提取的文本层），当前暂不支持图片型文献。"
                "请下载可复制文字的电子版，或使用OCR工具（如 Adobe Acrobat 的识别功能）转换后再上传。"
            )
    return docs


# --- 向量库管理（持久化 + 多文献） ---

def get_vectorstore() -> Chroma:
    """获取持久化的Chroma向量库。同一目录反复打开即可，重启不丢数据。"""
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_get_embeddings())


def list_sources(vectorstore: Chroma) -> list[str]:
    """列出库中所有文献（按文件名去重排序）。"""
    data = vectorstore.get(include=["metadatas"])
    sources = {m.get("source") for m in data.get("metadatas", []) if m and m.get("source")}
    return sorted(sources)


def add_pdf(vectorstore: Chroma, pdf_file, source_name: str, progress_callback=None) -> int:
    """处理一个上传的PDF并入库（source_name=文件名，作为检索过滤的标签）。

    同名文献重复上传会先删除旧块再入库（相当于替换为新版本）。
    返回本次入库的文本块数。
    """
    def report(fraction, message):
        if progress_callback:
            progress_callback(min(fraction, 1.0), message)

    report(0.05, "保存上传文件...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        temp_pdf_path = tmp_file.name

    try:
        report(0.08, "解析PDF（含表格提取）...")
        documents = load_pdf_pages(temp_pdf_path, progress_callback=lambda f, m: report(0.08 + f * 0.22, m))
        report(0.30, f"加载完成，共 {len(documents)} 页")

        report(0.30, "切分文本块...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        # 给每块打上文献标签，检索时可按文献过滤
        for c in chunks:
            c.metadata["source"] = source_name
        report(0.35, f"切分完成，共 {len(chunks)} 块")

        # 替换同名文献的旧数据
        old = vectorstore.get(where={"source": source_name}, include=[])
        if old.get("ids"):
            vectorstore.delete(ids=old["ids"])

        # 分批向量化入库，逐批回报真实进度
        total = len(chunks)
        batch_size = 16
        for i in range(0, total, batch_size):
            vectorstore.add_documents(chunks[i:i + batch_size])
            done = min(i + batch_size, total)
            report(0.35 + 0.60 * done / total, f"向量化进度 {done}/{total}")
        report(1.0, f"《{source_name}》入库完成，共 {total} 块")
        return total
    finally:
        os.unlink(temp_pdf_path)


def remove_source(vectorstore: Chroma, source_name: str) -> None:
    """从库中删除一篇文献的全部文本块。"""
    old = vectorstore.get(where={"source": source_name}, include=[])
    if old.get("ids"):
        vectorstore.delete(ids=old["ids"])


# --- 生成结果的磁盘缓存（摘要 / 概念图谱，按检索范围存取） ---
# 摘要和概念图谱生成很慢（GLM强制思考后单次要1-2分钟），而session_state一重启
# 或切换检索范围就没了。这里把生成结果按检索范围落盘：同一范围再次打开时直接
# 加载，不必重新花钱花时间生成。

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _artifact_key(sources) -> str:
    """检索范围 -> 稳定缓存键（排序后取md5；空列表表示'全部文献'范围）。"""
    raw = "\n".join(sorted(s or "" for s in (sources or [])))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def save_artifact(kind: str, sources, data) -> None:
    """持久化一份生成结果。kind: 'summary'（结构化摘要）或 'graph'（概念+关系）。"""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    payload = {"sources": sorted(sources or []), "kind": kind, "data": data}
    path = os.path.join(ARTIFACT_DIR, f"{kind}_{_artifact_key(sources)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_artifact(kind: str, sources):
    """读取对应检索范围的缓存结果；没有或文件损坏返回None。"""
    path = os.path.join(ARTIFACT_DIR, f"{kind}_{_artifact_key(sources)}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("data")
    except (OSError, ValueError) as e:
        logger.warning("缓存文件读取失败 %s: %s", path, e)
        return None


def list_artifacts(kind: str):
    """列出某类已保存结果的来源范围（供UI提示"哪些文献已有图谱/摘要可看"）。"""
    out = []
    if not os.path.isdir(ARTIFACT_DIR):
        return out
    for fn in os.listdir(ARTIFACT_DIR):
        if not fn.startswith(kind + "_") or not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(ARTIFACT_DIR, fn), encoding="utf-8") as f:
                out.append(sorted(json.load(f).get("sources", [])))
        except (OSError, ValueError):
            continue
    return out


_SCOPE_FILE = os.path.join(ARTIFACT_DIR, "_last_scope.json")


def save_last_scope(sources) -> None:
    """记住本次使用的检索范围，让新开的浏览器界面能自动回到同一篇文献。"""
    try:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(_SCOPE_FILE, "w", encoding="utf-8") as f:
            json.dump({"sources": sorted(sources or [])}, f, ensure_ascii=False)
    except OSError as e:
        logger.warning("保存检索范围失败: %s", e)


def load_last_scope():
    """读取上次使用的检索范围；没有记录返回None。"""
    try:
        with open(_SCOPE_FILE, encoding="utf-8") as f:
            return list(json.load(f).get("sources", []))
    except (OSError, ValueError):
        return None


def delete_artifacts_for(sources) -> None:
    """文献被删除/重新上传后，所有引用它的缓存全部作废删除（防止新旧内容错配）。"""
    if not os.path.isdir(ARTIFACT_DIR):
        return
    names = set(sources or [])
    for fn in os.listdir(ARTIFACT_DIR):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(ARTIFACT_DIR, fn)
        try:
            with open(path, encoding="utf-8") as f:
                stale = names & set(json.load(f).get("sources", []))
        except (OSError, ValueError):
            continue
        if stale:
            os.unlink(path)


# --- 问答 ---

# 口语化系统提示词（架构图中的 StylePrompt 节点）
STYLE_PROMPT = """你是一位耐心的文献阅读助手，帮助初学者理解学术文献。回答要求：
1. 用口语化的中文大白话解释，像给朋友讲课一样，少用长句和生僻词；
2. 遇到专业术语时先用大白话说清楚，再在括号里给出原文术语；
3. 适当举贴近生活的例子帮助理解；
4. 只基于提供的文献内容回答，不要编造；文献里没有的信息要明确说明。
5. 回答末尾不需要客套话。"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", STYLE_PROMPT),
    ("human", "请根据以下文献内容回答问题。\n\n文献内容：\n{context}\n\n问题：{question}"),
])


# --- 问答（混合检索：BM25关键词 + 向量语义，EnsembleRetriever加权融合） ---

def _jieba_tokenize(text: str) -> list[str]:
    """BM25的中文分词器：纯空格分词对中文无效，必须用jieba。"""
    return jieba.lcut(text)


def get_hybrid_retriever(vectorstore: Chroma, sources=None, k: int = RETRIEVER_K):
    """关键词(BM25) + 语义(Chroma向量) 加权融合检索。

    BM25索引从向量库中同一批分块构建（与向量检索共享数据源），
    sources非空时两路检索都限定在该文献范围内。
    """
    flt = _source_filter(sources)
    data = vectorstore.get(where=flt, include=["documents", "metadatas"])
    docs = [
        Document(page_content=d, metadata=m or {})
        for d, m in zip(data.get("documents", []), data.get("metadatas", []))
        if d
    ]
    if not docs:
        return None

    bm25 = BM25Retriever.from_documents(
        docs, preprocess_func=_jieba_tokenize, k=k
    )
    search_kwargs = {"k": k}
    if flt:
        search_kwargs["filter"] = flt
    vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    # 关键词路与语义路各占一半：公式符号、专有名词类问题BM25占优，
    # 口语化改写类问题向量检索占优，融合后互为补充
    return EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.5, 0.5])


def get_qa_chain(vectorstore: Chroma, sources=None) -> RetrievalQA:
    """创建RAG问答链（口语化风格）。sources非空时只在该文献范围内检索。"""
    retriever = get_hybrid_retriever(vectorstore, sources=sources) or vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=_get_llm(),
        chain_type="stuff",  # 检索块较少时直接拼进prompt即可
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )


CONV_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", STYLE_PROMPT),
    ("human", """请根据文献内容回答用户的最新问题。如果是追问（例如"刚才那个公式怎么推导"），
请结合对话历史理解用户指的是什么。

对话历史（可能为空）：
{history}

文献内容：
{context}

最新问题：{question}"""),
])


def answer_question(vectorstore: Chroma, question: str, history: list = None,
                    sources=None, retriever=None) -> dict:
    """多轮对话式问答：混合检索 + 对话历史拼入prompt。

    history: [{"role": "user"|"assistant", "content": str}]，只取最近几轮防止prompt过长。
    retriever: 可选，外部缓存的混合检索器（避免每次问答重建BM25索引）。
    返回 {"answer": str, "source_documents": [Document]}，结构与旧QA链兼容。
    """
    if retriever is None:
        retriever = get_hybrid_retriever(vectorstore, sources=sources) or vectorstore.as_retriever()
    docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[{d.metadata.get('source', '?')} · 第{(d.metadata.get('page', 0) or 0) + 1}页]\n{d.page_content}"
        for d in docs
    )
    turns = (history or [])[-4:]  # 最近4条消息（2轮），防止历史把prompt撑爆
    hist_text = "".join(
        f"{'用户' if t['role'] == 'user' else '助手'}：{t['content'][:300]}\n" for t in turns
    ) or "（无）"

    answer = _invoke_text(_get_llm(),
                          CONV_QA_PROMPT.format(history=hist_text, context=context, question=question))
    return {"answer": answer or "（模型未返回内容，请重试。）", "source_documents": docs}


def extract_text(result: dict) -> str:
    """从QA链的返回中提取纯文本答案（GLM会返回thinking块，需过滤）。"""
    raw = result.get("result", "")
    if raw.strip().startswith("["):
        # 旧版链路可能把整个content块列表str()进去，此时回退解析
        import ast

        try:
            blocks = ast.literal_eval(raw)
            return "".join(
                b.get("text", "") for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        except (ValueError, SyntaxError):
            pass
    return raw.strip()


# --- 概念提取与出题 ---

class ExtractedConcept(BaseModel):
    """从文献中提取的单个概念。"""
    name: str = Field(description="概念/术语的名称，保留文献原文写法")
    definition: str = Field(description="基于文献内容的初学者友好解释，不超过50字")
    category: str = Field(description="概念所属主题，从这些类别中选一个最贴切的：模型架构、算法机制、训练方法、数学工具、评价指标、应用任务、其他")


class ExtractedRelation(BaseModel):
    """两个概念之间的关系，形如 主体-谓语-客体。"""
    subject: str = Field(description="关系的主 Concept 名称")
    predicate: str = Field(description="关系类型，必须是简短词语（不超过6个字），如：依赖、包含、用于、组成、对比")
    object: str = Field(description="关系的客体概念名称")


class ExtractionResult(BaseModel):
    """单次结构化抽取的完整结果。"""
    concepts: list[ExtractedConcept] = Field(description="文献中出现的对初学者陌生的概念，没有则为空列表")
    relations: list[ExtractedRelation] = Field(description="概念之间的关系，没有则为空列表")


class QuizQuestion(BaseModel):
    """基于文献内容生成的单选题。"""
    question: str = Field(description="题干，考查对该概念的理解")
    options: list[str] = Field(description="恰好4个选项A-D，干扰项要有迷惑性但明显错误")
    answer_index: int = Field(description="正确选项的下标，取值0-3")
    explanation: str = Field(description="答案解析，口语化，说明为什么对、其他选项错在哪")


class QuizBatch(BaseModel):
    """一次生成的一组单选题。"""
    questions: list[QuizQuestion] = Field(description="一组单选题")


EXTRACT_PROMPT = """你是一位耐心的学术阅读导师。请从下面的文献片段中，提取出对初学者来说陌生的核心概念，
以及这些概念之间的关系。要求：
1. 只提取文献中真正出现的概念，不要自己编造；
2. 概念解释要基于文献内容、面向初学者；
3. 关系要明确、有信息量（例如"自注意力 依赖 缩放点积注意力"），关系词必须简短（不超过6个字）；
4. 必须通过工具提交结果，不要用纯文本回答；
5. 单个片段最多提取10个概念，选最重要的；
6. 如果片段是参考文献列表、纯公式推导、致谢等没有实质概念的内容，必须返回空的concepts和relations列表，仍然要调用工具提交，绝不许省略。

文献片段：
{text}"""

QUIZ_PROMPT = """你是一位出题老师。请基于下面的文献内容，围绕概念「{concept}」出一道中文单项选择题，
考查对该概念的理解。要求：
1. 题干清晰，可以用大白话问；
2. 恰好4个选项，正确答案只有一个，干扰项要有迷惑性但不能模棱两可；
3. 给出口语化的答案解析。

概念解释：{definition}

文献内容：
{context}"""

# 一组出题（一次调用生成多道，比逐题调用快数倍）：一半考概念本身，
# 其余考与它直接相关的概念、以及它们与该概念的关系（关系来自概念图谱提取结果）
QUIZ_BATCH_PROMPT = """你是一位出题老师。请基于下面的文献内容，围绕核心概念「{concept}」出{n}道中文单项选择题，
组成一份小测验。要求：
1. 题目构成：约一半考查「{concept}」本身的理解，其余考查"相关概念"里列出的概念、
   以及它们与「{concept}」之间的关系（比如谁依赖谁、谁是谁的组成部分、效果对比）；
2. 题干清晰，可以用大白话问；题目之间角度尽量多样（定义、作用、因果关系、对比），不要重复；
3. 每题恰好4个选项，正确答案只有一个，干扰项要有迷惑性但不能模棱两可；
4. 每题给出口语化的答案解析；
5. 正确答案的位置要打散，不要连续落在同一选项；
6. 必须通过工具提交结果，不要用纯文本回答。

核心概念解释：{definition}

相关概念及其定义：
{related}

与「{concept}」相关的关系：
{relations_text}

文献内容：
{context}"""

# 结构化摘要（"读论文模式"）
# 输出约定（app.py的render_summary_markdown会把这些标记渲染成图形样式）：
#   - 每条要点以「逻辑词」开头 -> 渲染成彩色徽章
#   - 关键术语用[[术语]]包裹 -> 渲染成同色下划线高亮
SUMMARY_PROMPT = """你是一位帮初学者精读论文的导师。请基于下面的文献内容，输出一份Markdown格式的结构化摘要。

格式要求（必须严格遵守）：
1. 必须包含以下四个二级标题。每部分先用一句话概括，然后换行用列表展开3-6条要点（每条以"- "开头）：
## 研究背景
## 核心创新点
## 方法论
## 实验结论
2. 每条要点的最开头用「」标出这条要点承担的逻辑角色，只能从下面这些词里选最贴切的一个：
   「问题」「方案」「因果」「转折」「并列」「递进」「对比」「目的」「方法」「结果」「补充」
   示例：- 「因果」因为RNN要逐帧计算，所以训练无法并行，速度很慢。
3. 关键概念/术语/模型名/评价指标，第一次出现时用双方括号包起来，如 [[自注意力]]、[[Transformer]]；
   同一个术语全文只在第一次出现时标注一次，不要整句标注，普通词不要标。
4. 语言通俗但不失准确，可以用大白话解释；关键公式用 $...$ 包裹。
5. 只基于文献内容，不要编造；文献中缺失的部分直接写"文献中未明确提及"。
6. 直接输出Markdown，不要输出任何解释或前言。

文献内容：
{context}"""


def generate_paper_summary(vectorstore: Chroma, sources=None, max_chunks: int = 12,
                           progress_callback=None) -> str:
    """读论文模式：按页序等距采样全文文本块，生成四段式结构化摘要。

    等距采样而不是只用检索命中的块，保证背景（开头）、方法（中段）、结论（结尾）都覆盖到。
    """
    def report(frac, msg):
        if progress_callback:
            progress_callback(min(max(frac, 0.0), 1.0), msg)

    report(0.1, "收集全文文本块...")
    flt = _source_filter(sources)
    data = vectorstore.get(where=flt, include=["documents", "metadatas"])
    items = [
        (m.get("page", 0) or 0, d)
        for d, m in zip(data.get("documents", []), data.get("metadatas", []))
        if d
    ]
    if not items:
        raise RuntimeError("文献库为空，无法生成摘要。")
    items.sort(key=lambda x: x[0])

    # 等距采样：开头/中间/结尾都取到
    step = max(len(items) / max_chunks, 1)
    sampled = [items[int(i * step)] for i in range(min(max_chunks, len(items)))]
    context = "\n\n".join(d for _, d in sampled)
    report(0.2, f"已采样 {len(sampled)} 块，正在精读...")

    llm = _get_llm()
    summary = _invoke_text(llm, SUMMARY_PROMPT.format(context=context))
    report(1.0, "摘要生成完成！")
    return summary or "（模型未返回内容，请重试。）"


def _clean_chunk_text(text: str) -> str:
    """方案B：送入LLM前剔除对概念提取毫无价值的段落。

    PyPDFLoader切出的块里常见：参考文献列表、纯公式/数字行、页眉页脚。
    这些内容既浪费token，又是GLM"拒答/不调工具"的高发区。
    清洗后为空则返回None，调用方可直接跳过，连LLM都不用调。
    """
    lines = []
    for line in text.splitlines():
        s = line.strip()
        # 参考文献条目：[1]xxx / [12]xxx
        if re.match(r"^\[\d+\]", s):
            continue
        # 作者-年份式引用行：Vaswani, A., et al. (2017)...
        if re.match(r"^[A-Z][a-zA-Z,\.\- ]+et al\.?", s):
            continue
        # 几乎全是数学符号/数字/标点的行（保留少量字母，如"where x is..."这类有文字的行）
        letters = len(re.findall(r"[A-Za-z一-鿿]", s))
        if len(s) > 0 and letters / max(len(s), 1) < 0.15 and not s.endswith(("。", ".", "？", "？", "!")):
            continue
        lines.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned or None


def _msg_text(message) -> str:
    """从AIMessage中取纯文本（content可能是str，也可能是块列表）。"""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "".join(parts)


def _invoke_text(llm, prompt) -> str:
    """流式调用并拼接纯文本回复。

    GLM强制思考后，长请求在非流式下会长时间一个字节都不返回，
    实测超过若干分钟会被网关/中间链路以ReadTimeout掐断（小请求秒回，无此问题）。
    流式传输持续有增量到达，连接不会被判定空闲；对调用方仍然是拿到完整文本。
    """
    parts = []
    for chunk in llm.stream(prompt):
        t = _msg_text(chunk)
        if t:
            parts.append(t)
    return "".join(parts).strip()


def _parse_extraction(message):
    """方案C：手动解析结构化结果，不再依赖with_structured_output的"失败即None"。

    三级递进：
    1) 模型正确调用了工具 -> 直接用tool_calls参数构造；
    2) 模型没调工具而是吐了纯文本 -> 从文本里抢救出JSON（剥掉```json围栏）再构造；
    3) 都不行 -> 返回None，由调用方记录日志并重试。
    """
    # 1) 标准路径：tool_calls
    for tc in getattr(message, "tool_calls", None) or []:
        args = tc.get("args") or {}
        if args.get("concepts") is not None or args.get("relations") is not None:
            try:
                return ExtractionResult.model_validate(args)
            except Exception as e:
                # 参数被max_tokens截断等情况：JSON不完整，无法挽救，需重试
                logger.warning("tool_calls参数校验失败（疑似截断）: %s", str(e)[:200])
                return None

    # 2) 退化路径：从纯文本中抢救JSON
    raw = _msg_text(message).strip()
    if raw:
        # 去掉markdown代码围栏，截取首尾大括号之间的内容
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                data = json.loads(raw[start:end + 1])
                return ExtractionResult.model_validate(data)
            except Exception as e:
                logger.debug("文本JSON抢救失败: %s", str(e)[:200])
    logger.warning("模型既未调用工具，也无法从返回文本中解析出JSON（前200字: %r）", raw[:200])
    return None


def extract_concepts(vectorstore: Chroma, sources=None, max_chunks: int = EXTRACT_MAX_CHUNKS,
                     progress_callback=None) -> dict:
    """从向量库的文本块中提取概念与关系，跨块去重。

    sources非空时只提取这些文献的内容。
    progress_callback: 可选，签名 callback(fraction, message)，按文本块回报真实进度。
    返回 {"concepts": [{name, definition, category}], "relations": [{subject, predicate, object}],
          "skipped": [{chunk_snippet, reason}]}   # skipped记录提取失败的块，便于人工排查
    """
    def report(fraction, message):
        if progress_callback:
            progress_callback(min(fraction, 1.0), message)

    data = vectorstore.get(where=_source_filter(sources), limit=max_chunks, include=["documents"])
    # 方案B：先清洗，参考文献/纯公式块清洗后为空的直接跳过（省token也避开模型拒答区）
    texts, dropped = [], 0
    for t in data.get("documents", []):
        if not t or not t.strip():
            continue
        cleaned = _clean_chunk_text(t)
        if cleaned is None:
            dropped += 1
            logger.info("跳过无实质内容的块（清洗后为空），前100字: %r", t[:100])
        else:
            texts.append(cleaned)
    if dropped:
        logger.info("预处理跳过 %d 个公式/参考文献类空块", dropped)

    llm = _get_llm().bind_tools([ExtractionResult])  # 方案C：绑工具+手动解析，不用with_structured_output

    def extract_one(text):
        """提取单个文本块。失败时记日志（含块前200字），最多重试2次。"""
        prompt = EXTRACT_PROMPT.format(text=text)
        for attempt in range(2):
            try:
                msg = llm.invoke(prompt)
                res = _parse_extraction(msg)
                if res is not None:
                    return res
                logger.warning("第%d次提取未得到结构化结果，块前200字: %r", attempt + 1, text[:200])
            except Exception as e:
                logger.warning("第%d次提取异常: %s；块前200字: %r", attempt + 1, str(e)[:200], text[:200])
        return None

    skipped: list[dict] = []

    # 并行提取：多块同时送LLM，比串行快数倍；结果顺序无关（后面会去重合并）
    concept_map: dict[str, str] = {}
    relations: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        futures = {pool.submit(extract_one, t): t for t in texts}
        for future in as_completed(futures):
            done += 1
            report(done / max(len(texts), 1), f"已完成 {done}/{len(texts)} 块文本分析...")
            res = future.result()
            if res is None:
                # 记录被跳过的块（不静默），前200字供人工确认块的类型
                skipped.append({"chunk_snippet": futures[future][:200], "reason": "两次尝试均未得到结构化结果"})
                continue  # 单块失败不影响整体提取
            for c in res.concepts:
                key = c.name.strip().lower()
                if key and key not in concept_map:
                    concept_map[key] = {
                        "name": c.name.strip(),
                        "definition": c.definition.strip(),
                        "category": (c.category or "").strip() or "其他",
                    }
            for r in res.relations:
                rel = {"subject": r.subject.strip(), "predicate": r.predicate.strip(), "object": r.object.strip()}
                if rel not in relations:
                    relations.append(rel)

    report(1.0, f"提取完成：{len(concept_map)} 个概念，{len(relations)} 条关系"
                + (f"（{len(skipped)} 块解析失败已跳过）" if skipped else ""))
    return {"concepts": list(concept_map.values()), "relations": relations, "skipped": skipped}


def _invoke_stream(llm, prompt):
    """流式调用并聚合为完整消息。长请求非流式会被网关掐断（见_invoke_text注释）；
    聚合后的AIMessageChunk保留tool_calls，供绑工具的调用解析。"""
    agg = None
    for chunk in llm.stream(prompt):
        agg = chunk if agg is None else agg + chunk
    return agg


def _parse_quiz_batch(message):
    """解析出题结果，与_parse_extraction同思路：先工具调用，再从纯文本抢救JSON。"""
    for tc in getattr(message, "tool_calls", None) or []:
        args = tc.get("args") or {}
        if args.get("questions"):
            try:
                return QuizBatch.model_validate(args)
            except Exception:
                logger.warning("工具参数无法解析为QuizBatch，转纯文本抢救")
                break
    raw = _msg_text(message).strip()
    if not raw:
        return None
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1)
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"(\[.*\]|\{.*\})", raw, re.S)  # 抓第一个JSON数组/对象
        if not m:
            return None
        try:
            data = json.loads(m.group(1))
        except Exception:
            return None
    if isinstance(data, list):
        data = {"questions": data}
    try:
        return QuizBatch.model_validate(data)
    except Exception:
        return None


def generate_quiz_batch(vectorstore: Chroma, concept_name: str, definition: str = "",
                        concepts=None, relations=None, sources=None,
                        num_questions: int = 5) -> list[QuizQuestion]:
    """围绕指定概念出一组单选题（默认5道，单次调用生成）。

    约一半考概念本身，其余考与它直接相连的概念及它们与该概念的关系
    （来自概念图谱提取结果）。sources非空时限定文献范围。
    """
    # 从图谱里找出与该概念直接相连的关系与相邻概念，连同定义一起喂给出题prompt
    rel_lines: list[str] = []
    related_defs: dict[str, str] = {}
    for r in relations or []:
        if concept_name in (r.get("subject"), r.get("object")):
            other = r["object"] if r["subject"] == concept_name else r["subject"]
            line = f"- {r['subject']} —{r['predicate']}→ {r['object']}"
            if line not in rel_lines:
                rel_lines.append(line)
            related_defs.setdefault(other, "")
    if concepts:
        cmap = {c["name"]: c["definition"] for c in concepts}
        for name in related_defs:
            related_defs[name] = cmap.get(name, "")
    related_text = "\n".join(
        f"- {n}：{d or '（文献中未给出明确定义）'}" for n, d in list(related_defs.items())[:5]
    ) or "（无）"
    relations_text = "\n".join(rel_lines[:8]) or "（未提取到与该概念直接相关的关系，题目全部围绕该概念本身出）"

    # 检索范围 = 概念本身 + 相邻概念（各取少量），让关系类题目也有文献依据
    flt = _source_filter(sources)
    base_kwargs = {"k": 3}
    if flt:
        base_kwargs["filter"] = flt
    retriever = vectorstore.as_retriever(search_kwargs=base_kwargs)
    docs = retriever.invoke(concept_name)
    if related_defs:
        rel_kwargs = {"k": 1}
        if flt:
            rel_kwargs["filter"] = flt
        rel_retriever = vectorstore.as_retriever(search_kwargs=rel_kwargs)
        seen = {d.page_content for d in docs}
        for name in list(related_defs)[:4]:
            for d in rel_retriever.invoke(name):
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    docs.append(d)
    context = "\n\n".join(d.page_content for d in docs)

    llm = _get_llm().bind_tools([QuizBatch])
    prompt = QUIZ_BATCH_PROMPT.format(
        concept=concept_name,
        n=num_questions,
        definition=definition or "（文献中未给出明确定义）",
        related=related_text,
        relations_text=relations_text,
        context=context,
    )
    # 流式保活出一次题；聚合解析失败（个别情况下流式工具调用聚合不完整）退回非流式重试
    batch = _parse_quiz_batch(_invoke_stream(llm, prompt))
    if batch is None or not batch.questions:
        logger.warning("流式出题未解析出结果，退回非流式重试一次")
        batch = _parse_quiz_batch(llm.invoke(prompt))
    if batch is None or not batch.questions:
        raise RuntimeError("模型没有成功生成题目，请点击\"生成题目\"再试一次。")
    return batch.questions[:num_questions]


# --- 测试代码 ---
if __name__ == "__main__":
    print("RAG Core module loaded.")
    print("LLM:", LLM_MODEL, f"({LLM_API_TYPE} @ {LLM_BASE_URL})",
          "| Embedding:", EMBEDDING_MODEL, f"(@ {EMBEDDING_BASE_URL})")
    vs = get_vectorstore()
    print("已入库文献:", list_sources(vs) or "（空）")
