# app.py
# 文献阅读助手界面（Streamlit）：问答 / 概念图谱 / 自测出题 三个Tab
# 支持多文献管理：可同时上传多篇PDF入库，检索/提取/出题均可按文献来源过滤
import json
import logging
import mimetypes
import threading
import time

import streamlit as st

# 自托管mermaid/ELK需要静态服务能以text/javascript返回.js/.mjs。
# Streamlit的AppStaticFileHandler默认只给白名单扩展名真实Content-Type，
# 其余强制text/plain并带nosniff（浏览器会拒载）。这里运行时把.js/.mjs
# 加进白名单，并注册.mjs的MIME类型（Tornado按mimetypes猜测）。
try:
    import streamlit.web.server.app_static_file_handler as _asfh

    if ".mjs" not in _asfh.SAFE_APP_STATIC_FILE_EXTENSIONS:
        _asfh.SAFE_APP_STATIC_FILE_EXTENSIONS = tuple(
            _asfh.SAFE_APP_STATIC_FILE_EXTENSIONS
        ) + (".js", ".mjs")
except Exception:
    pass  # 未来版本结构变化时放弃补丁，图谱走CDN回退
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")
from streamlit.components.v1 import html as render_html

# rag_core的调试日志（提取失败块的信息）输出到终端，方便排查
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
try:
    from rag_core import (
        get_vectorstore,
        list_sources,
        add_pdf,
        remove_source,
        get_hybrid_retriever,
        answer_question,
        extract_concepts,
        enrich_concept_definitions,
        generate_quiz_batch,
        generate_paper_summary,
        save_artifact,
        load_artifact,
        delete_artifacts_for,
        list_artifacts,
        save_last_scope,
        load_last_scope,
    )
except RuntimeError:
    # 首次使用者没配密钥时，rag_core会在导入时抛RuntimeError。
    # 与其让用户看到一堆报错堆栈，这里直接渲染成一步步的配置引导。
    st.set_page_config(page_title="📚 文献阅读助手", layout="wide")
    st.error("## 🔑 首次使用：请先在 `.env` 中配置你的模型服务", icon="🚨")
    st.markdown(
        "本项目默认使用智谱GLM，也支持 **DeepSeek、Kimi、通义千问、OpenAI 等任何兼容接口的厂商**。"
        "**项目里不包含密钥**，每位使用者需要自己申请。要填的就是三样东西："
        "**接口地址、API密钥、模型名**，跟着下面三步走，一分钟搞定："
    )
    st.markdown(
        "### 第 1 步：去厂商开放平台申请密钥\n"
        "注册/登录后，在控制台里创建 API key 并复制，同时记下它的**接口地址（base_url）**"
        "和**模型名**（都在厂商文档的\"接口说明\"里）。常见厂商：\n\n"
        "| 厂商 | 申请/文档地址 | 协议 | 模型名示例 |\n"
        "| --- | --- | --- | --- |\n"
        "| 智谱GLM（默认） | [open.bigmodel.cn](https://open.bigmodel.cn) | anthropic | `glm-5.3-flash` |\n"
        "| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | openai | `deepseek-chat` |\n"
        "| Kimi | [platform.moonshot.cn](https://platform.moonshot.cn) | openai | `moonshot-v1-32k` |\n"
        "| 通义千问 | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) | openai | `qwen-plus` |\n"
        "| OpenAI | [platform.openai.com](https://platform.openai.com) | openai | `gpt-4o-mini` |"
    )
    st.markdown("### 第 2 步：把三项配置填进项目根目录的 `.env` 文件")
    st.code(
        "# 在项目根目录执行（Windows 把 cp 换成 copy）：\n"
        "cp .env.example .env",
        language="bash",
    )
    st.markdown("然后用任意文本编辑器打开 `.env`，把开头的四行改成你选的厂商：")
    st.code(
        "LLM_BASE_URL=https://open.bigmodel.cn/api/anthropic   # 接口地址\n"
        "LLM_API_TYPE=anthropic                                # 接口协议：anthropic 或 openai\n"
        "LLM_API_KEY=粘贴你刚复制的密钥                          # API密钥\n"
        "LLM_MODEL=glm-5.3-flash                               # 模型名",
        language="ini",
    )
    st.markdown("### 第 3 步：重启应用\n按 Ctrl+C 停掉当前进程，重新运行启动命令，刷新本页面即可。")
    st.info(
        "💡 `.env.example` 里有全部常见厂商的地址/模型速查表，复制对应几行即可切换厂商；"
        "`.env` 已被 .gitignore 排除，你的密钥不会被提交到 GitHub。"
    )
    st.stop()

# --- 页面配置 ---
st.set_page_config(page_title="📚 文献阅读助手", layout="wide")  # 铺满浏览器宽度，去掉两侧大留白
st.title("📚 文献阅读助手")
st.markdown("上传PDF文献（可多篇），提问、看概念、做自测。知识库持久化保存，重启后无需重新上传。")

# --- 初始化session state ---
DEFAULT_STATE = {
    "vectorstore": None,
    "sources": [],            # 库中全部文献名
    "selected_sources": [],   # 当前勾选的文献过滤范围（空=全部）
    "retriever": None,        # 混合检索器缓存（随过滤范围重建）
    "retriever_key": None,    # 构建当前检索器时用的过滤范围
    "messages": [],           # 多轮对话历史 [{"role", "content", "docs"?}]
    "concepts": [],           # [{name, definition, category}]
    "relations": [],          # [{subject, predicate, object}]
    "quiz": None,             # QuizQuestion对象
    "summary": None,          # 读论文模式的结构化摘要缓存
    "pending_question": None, # 概念图谱联动：待送入问答的问题
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 持久化向量库：启动时直接打开磁盘上的chroma_db目录，之前入库的文献都在
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = get_vectorstore()
    st.session_state.sources = list_sources(st.session_state.vectorstore)


@st.cache_data(show_spinner=False)
def _load_source_pages(source_name: str):
    """从向量库读取某篇文献的全部文本块并按页归组。

    Streamlit每次交互都会重跑整个脚本，读库结果按文献名缓存，
    避免翻页/切换时反复读整个向量库；文献重传或删除时手动clear()。
    """
    vs = get_vectorstore()
    data = vs.get(where={"source": source_name}, include=["documents", "metadatas"])
    pages = {}
    for d, m in zip(data.get("documents", []), data.get("metadatas", [])):
        if d:
            pages.setdefault(m.get("page", 0) or 0, []).append(d.strip())
    return pages


def clear_cached_results():
    """清空全部派生结果（对话/摘要/概念/题目），文献库内容变化时用。"""
    st.session_state.retriever = None
    st.session_state.retriever_key = None
    st.session_state.messages = []  # 检索范围变了，旧对话历史不再成立
    st.session_state.concepts = []
    st.session_state.relations = []
    st.session_state.quiz = None
    st.session_state.summary = None


def invalidate_cached_results():
    """检索范围变化后的结果切换：对话作废；摘要和概念图谱按范围落盘缓存，
    有缓存就直接恢复（生成一次要1-2分钟，重启/切范围不该重来）。"""
    clear_cached_results()
    srcs = list(st.session_state.selected_sources)
    save_last_scope(srcs)  # 记住范围，新开浏览器界面时自动回到这里
    # 范围同时写进URL：F5刷新后浏览器原样带着参数发回来，比服务端文件更可靠
    # （多标签页/旧会话的文件写入可能互相覆盖），刷新后图谱就不会"丢"
    try:
        if srcs:
            st.query_params["scope"] = srcs
        elif "scope" in st.query_params:
            del st.query_params["scope"]
    except Exception:
        pass  # 个别嵌入式环境不允许改URL，退回仅用文件记忆
    graph = load_artifact("graph", srcs)
    if graph is None and not srcs and len(st.session_state.sources) == 1:
        # “全部文献”范围没有自己的缓存、但库里只有一篇时，那篇的缓存就是全部内容的缓存
        graph = load_artifact("graph", [st.session_state.sources[0]])
    st.session_state.concepts = graph["concepts"] if graph else []
    st.session_state.relations = graph["relations"] if graph else []
    summary = load_artifact("summary", srcs)
    if summary is None and not srcs and len(st.session_state.sources) == 1:
        summary = load_artifact("summary", [st.session_state.sources[0]])
    st.session_state.summary = summary


def make_progress_bar(label="准备中..."):
    """创建进度条和真实进度回调（适用于有分步/分块结构的长任务）。"""
    bar = st.progress(0.0, text=label)

    def callback(fraction, message):
        bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

    return bar, callback


def fade_out_progress(bar, text="✅ 完成！"):
    """进度条走完后的淡化效果：进度条消失，完成提示以动画淡到低亮度，
    不再有大块进度条一直杵在页面上（CSS动画随下次rerun自然清除）。"""
    bar.empty()
    st.markdown(
        f'<div style="color:#1a7f37;font-size:0.9em;'
        f'animation:cg_progress_fade 2.5s ease 0.5s forwards;">{text}</div>'
        f'<style>@keyframes cg_progress_fade {{ to {{ opacity: 0.35; }} }}</style>',
        unsafe_allow_html=True,
    )


def run_with_progress_bar(fn, label):
    """执行单次LLM调用类任务（没有可回报的中间进度）。

    在后台线程跑任务，主线程让进度条缓步推进，跑完立即拉满并返回结果；
    出错时把异常抛回调用方，由外层 st.error 展示。
    """
    bar = st.progress(0.0, text=label)
    holder = {}

    def worker():
        try:
            holder["value"] = fn()
        except Exception as e:
            holder["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    fraction = 0.05
    while thread.is_alive():
        bar.progress(min(fraction, 0.95), text=label)
        fraction += 0.03
        time.sleep(1.0)

    thread.join()
    if "error" in holder:
        bar.progress(1.0, text=f"{label} — 失败")
        raise holder["error"]
    fade_out_progress(bar, f"✅ {label} — 完成！")
    return holder["value"]


def format_node_label(name):
    """节点标签排版：英文术语和中文译名拆成两行，如
    'self-attention（自注意力）' -> 'self-attention<br/>自注意力'。"""
    safe = name.replace('"', "'")
    if "（" in safe and safe.endswith("）"):
        en, _, rest = safe.partition("（")
        return f"{en}<br/>{rest[:-1]}"
    return safe


# 主题分类配色（浅色填充 + 深色描边），循环使用
CATEGORY_PALETTE = [
    ("#eaf3fc", "#5b8fc9"),  # 蓝
    ("#e8f6ec", "#5ea575"),  # 绿
    ("#fdf3e0", "#d29a3a"),  # 橙黄
    ("#f3ecfa", "#9a72c4"),  # 紫
    ("#fdeaea", "#cd6a6a"),  # 红
    ("#eaf7f6", "#4fa8a2"),  # 青
    ("#f5f0e6", "#a68b5b"),  # 棕
    ("#eef0f4", "#7d8aa0"),  # 灰
]


def shorten_predicate(pred):
    """边标签显示用的短关系词：去掉括号补充说明，超长截断。"""
    import re

    p = re.sub(r"（[^）]*）", "", pred).strip() or pred
    if len(p) > 6:
        p = p[:6] + "…"
    return p


def build_mermaid_chart(concepts, relations):
    """把概念与关系转成mermaid flowchart代码。

    返回 {
      "chart": mermaid代码,
      "cat_colors": {主题: (填充色, 描边色)},   # 供图例使用
      "edge_defs": {短关系词: 完整关系词},      # 供边悬停提示使用
    }
    """
    known = {c["name"]: c for c in concepts}
    known_lower = {name.lower(): c for name, c in known.items()}

    def resolve(name):
        # 优先精确匹配，退而求其次大小写不敏感匹配
        return known.get(name) or known_lower.get(name.lower()) or {
            "name": name, "category": "未分类", "definition": ""
        }

    # 连接度：连接越多的概念越核心
    degree = {}
    for r in relations:
        for n in (r["subject"], r["object"]):
            degree[n] = degree.get(n, 0) + 1

    # 主题 -> 配色序号
    cat_colors = {}
    def cat_class(category):
        if category not in cat_colors:
            fill, border = CATEGORY_PALETTE[len(cat_colors) % len(CATEGORY_PALETTE)]
            cat_colors[category] = (fill, border)
        return f"cat{list(cat_colors).index(category)}"

    node_ids = {}
    node_class = {}
    lines = ["flowchart LR"]  # 横向布局：概念从左往右展开，长术语标签不吃垂直空间
    for r in relations:
        for name in (r["subject"], r["object"]):
            if name not in node_ids:
                node_ids[name] = f"N{len(node_ids)}"
                concept = resolve(name)
                label = format_node_label(concept["name"])
                lines.append(f'    {node_ids[name]}("{label}")')
                classes = [cat_class(concept.get("category", "未分类"))]
                if degree.get(name, 0) >= 3:
                    classes.append("core")  # 核心概念：加粗放大
                elif degree.get(name, 0) <= 1:
                    classes.append("minor")  # 边缘概念：缩小淡化
                node_class[node_ids[name]] = classes
    for r in relations:
        pred = r["predicate"].replace('"', "'")
        # 用引号包裹的管道形式，容错性最好（关系名带括号等特殊字符也能解析）
        lines.append(f'    {node_ids[r["subject"]]} -->|"{shorten_predicate(pred)}"| {node_ids[r["object"]]}')

    # 样式与类分配
    for i, (category, (fill, border)) in enumerate(cat_colors.items()):
        lines.append(f'    classDef cat{i} fill:{fill},stroke:{border},stroke-width:1.5px,color:#1f2d3d;')
    lines.append('    classDef core font-size:17px,font-weight:bold,stroke-width:3.5px;')
    lines.append('    classDef minor font-size:13px,opacity:0.85;')
    for nid, classes in node_class.items():
        lines.append(f'    class {nid} {",".join(classes)};')

    # 完整关系词 -> 悬停提示映射（显示的是截断后的短词）
    edge_defs = {}
    for r in relations:
        short = shorten_predicate(r["predicate"])
        if short != r["predicate"]:
            import re
            edge_defs[re.sub(r"\s+", "", short)] = f"完整关系：{r['predicate']}"

    return {"chart": "\n".join(lines), "cat_colors": cat_colors, "edge_defs": edge_defs}


def render_mermaid(chart, concept_defs=None, edge_defs=None):
    """在页面中渲染mermaid图，带缩放/全屏工具栏和悬停讲解。

    concept_defs: {概念名: 定义}。鼠标悬停在节点上时弹出对应定义卡片。
    edge_defs: {短关系词: 完整关系说明}。鼠标悬停在边上时弹出完整关系。
    布局优先用ELK引擎（边交叉远少于默认dagre，参考GitDiagram），CDN不可达时自动退回dagre。
    mermaid本体CDN用npmmirror（国内实测比jsdelivr快一个量级）。
    """
    n_nodes = max(chart.count('("'), 1)
    height = min(320 + 110 * n_nodes, 1500)

    # 深色模式适配的实际判断在JS里做（读宿主页面真实背景亮度，见下方脚本），
    # 这里的浅色值只作为CSS变量的兜底默认值。
    wrap_bg, wrap_border = "#fafbfc", "#eee"
    tb_bg, tb_border, tb_text, tb_hover, tb_zoom = (
        "#ffffff", "#d9e2ec", "#24425c", "#eef3f8", "#667")

    # JS里用 textContent 去掉空白后匹配节点/边，这里预先算好每个概念的key
    import re

    defs = {}
    for name, definition in (concept_defs or {}).items():
        label = format_node_label(name)
        key = re.sub(r"\s+", "", label.replace("<br/>", ""))
        defs[key] = definition

    # JSON在f-string外先算好：f-string表达式里写{}会被误解析为集合字面量
    defs_json = json.dumps(defs, ensure_ascii=False)
    edge_defs_json = json.dumps(edge_defs or {}, ensure_ascii=False)

    render_html(
        f"""
<style>
  :root {{
    --cg-wrap-bg: {wrap_bg}; --cg-wrap-border: {wrap_border};
    --cg-tb-bg: {tb_bg}; --cg-tb-border: {tb_border}; --cg-tb-text: {tb_text};
    --cg-tb-hover: {tb_hover}; --cg-tb-zoom: {tb_zoom};
  }}
  #cgraph-wrap {{ height: 100%; overflow: auto; border: 1px solid var(--cg-wrap-border); border-radius: 8px;
                  background: var(--cg-wrap-bg); cursor: grab; }}
  #cgraph-wrap.dragging {{ cursor: grabbing; }}
  #cgraph {{ transform-origin: top left; padding: 12px; }}
  #cgraph svg {{ width: 100%; height: 100%; }}
  .cg-toolbar {{ position: sticky; top: 6px; margin-left: 6px; z-index: 10; width: fit-content;
                background: var(--cg-tb-bg); border: 1px solid var(--cg-tb-border); border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,.12); display: flex; }}
  .cg-toolbar button {{ border: none; background: none; padding: 4px 10px; cursor: pointer;
                       font-size: 14px; color: var(--cg-tb-text); }}
  .cg-toolbar button:hover {{ background: var(--cg-tb-hover); }}
  .cg-toolbar span {{ padding: 4px 6px; font-size: 12px; color: var(--cg-tb-zoom); min-width: 38px; text-align: center; }}
  #cgraph-tip {{ position: fixed; display: none; width: 380px; max-width: 90vw; padding: 10px 14px;
                 background: #24425c; color: #fff; font-size: 13px; line-height: 1.7;
                 border-radius: 8px; box-shadow: 0 4px 14px rgba(0,0,0,.25); z-index: 99;
                 pointer-events: none; }}
  .cgraph-tip-name {{ font-size: 14px; font-weight: 600; margin-bottom: 4px;
                      color: #a8d4ff; }}
</style>
<div class="cg-toolbar">
  <button id="zout" title="缩小">－</button>
  <span id="zlevel">100%</span>
  <button id="zin" title="放大">＋</button>
  <button id="zreset" title="重置">重置</button>
  <button id="zfull" title="全屏查看">⛶ 全屏</button>
  <button id="zpng" title="导出PNG图片">📷 PNG</button>
</div>
<div id="cgraph-wrap"><div id="cgraph"></div>
  <!-- 提示卡片必须放在wrap内部：全屏时浏览器只渲染全屏元素的子树，放在外面会消失 -->
  <div id="cgraph-tip"><div class="cgraph-tip-name"></div><div class="cgraph-tip-body"></div></div>
</div>
<script type="module">
  // 优先加载本应用自托管的mermaid/ELK（/app/static，无外网依赖、秒加载），
  // 取不到宿主origin或文件缺失时回退CDN，保证图谱始终能渲染
  const origin = (() => {{
    try {{ return window.parent.location.origin; }} catch (e) {{ return ''; }}
  }})();

  async function loadMermaid() {{
    if (origin) {{
      try {{
        await new Promise((res, rej) => {{
          const s = document.createElement('script');
          s.src = origin + '/app/static/mermaid/mermaid.min.js';
          s.onload = res;
          s.onerror = () => rej(new Error('local failed'));
          document.head.appendChild(s);
        }});
        if (window.mermaid) return window.mermaid;
      }} catch (e) {{ /* 回退CDN */ }}
    }}
    return (await import(
      'https://registry.npmmirror.com/mermaid/11.4.1/files/dist/mermaid.esm.min.mjs'
    )).default;
  }}

  async function loadElkLayouts() {{
    const urls = [];
    if (origin) urls.push(origin + '/app/static/mermaid/elk.mjs');
    urls.push('https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0.2.1/dist/mermaid-layout-elk.esm.min.mjs');
    for (const u of urls) {{
      try {{ return (await import(u)).default; }} catch (e) {{ /* 试下一个来源 */ }}
    }}
    return null;
  }}

  const mermaid = await loadMermaid();

  // ---- 深色模式：读宿主页面(Streamlit)实际渲染的背景亮度来判断 ----
  // 不用Streamlit的st.context.theme：它在首次加载/切换主题的瞬间可能返回旧值
  function detectDark() {{
    try {{
      const doc = window.parent.document;
      const el = doc.querySelector('[data-testid="stApp"]') || doc.body || doc.documentElement;
      const m = getComputedStyle(el).backgroundColor.match(/\\d+/g);
      if (!m) return false;
      const [r, g, b] = m.map(Number);
      return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.5;
    }} catch (e) {{ return false; }}
  }}
  let curDark = detectDark();

  function cgPalette() {{
    return curDark ? {{
      wrapBg: '#1b2130', nodeFill: '#2b3547', nodeBorder: '#7ea8d8', nodeText: '#e6edf6',
      lineColor: '#5f7288', edgeLabelBg: '#232a3a',
    }} : {{
      wrapBg: '#fafbfc', nodeFill: '#eaf3fc', nodeBorder: '#5b8fc9', nodeText: '#1f2d3d',
      lineColor: '#8aa2b8', edgeLabelBg: '#ffffff',
    }};
  }}
  // 画布/工具栏配色通过CSS变量下发（mermaid节点配色在重渲染时生效）
  function applyUiPalette() {{
    const s = document.documentElement.style;
    const dark = curDark;
    s.setProperty('--cg-wrap-bg', dark ? '#1b2130' : '#fafbfc');
    s.setProperty('--cg-wrap-border', dark ? '#2c3547' : '#eee');
    s.setProperty('--cg-tb-bg', dark ? '#232a3a' : '#ffffff');
    s.setProperty('--cg-tb-border', dark ? '#3a4459' : '#d9e2ec');
    s.setProperty('--cg-tb-text', dark ? '#cdd6e4' : '#24425c');
    s.setProperty('--cg-tb-hover', dark ? '#2e374c' : '#eef3f8');
    s.setProperty('--cg-tb-zoom', dark ? '#8fa0b8' : '#667');
  }}
  applyUiPalette();

  function baseConfig() {{
    const p = cgPalette();
    return {{
      startOnLoad: false,
      suppressErrorRendering: true,   // 解析失败时不让mermaid往页面里塞自带的报错图
      securityLevel: 'antiscript',    // 概念名来自LLM输出，禁掉标签里可能夹带的脚本
      theme: 'base',
      flowchart: {{ nodeSpacing: 110, rankSpacing: 140, padding: 20, curve: 'linear',
                    htmlLabels: false }},  // SVG原生text标签：PNG导出时才不会变空白
      themeVariables: {{
        fontSize: '30px',
        primaryColor: p.nodeFill,
        primaryBorderColor: p.nodeBorder,
        primaryTextColor: p.nodeText,
        lineColor: p.lineColor,
        edgeLabelBackground: p.edgeLabelBg,
      }},
      themeCSS: `
        /* 悬停反馈：只缩放节点形状（绕形状自身中心），文字不动——
           之前对g.node全体子元素缩放时，文字的transform-origin解析不稳，
           会出现文字向右下漂移的问题 */
        g.node rect, g.node polygon {{ transition: transform .16s ease, filter .16s ease;
                      transform-box: fill-box; transform-origin: center; }}
        @media (hover: hover) and (pointer: fine) {{
          g.node:hover rect, g.node:hover polygon {{ transform: scale(1.08); filter: brightness(0.9); }}
        }}
        @media (prefers-reduced-motion: reduce) {{ g.node rect, g.node polygon {{ transition: none; }} }}
      `,
    }};
  }}

  // ELK布局引擎（GitDiagram同款）：比默认dagre的边交叉和绕线少得多，
  // 概念多、关系密的图谱可读性差距明显。本地/CDN都加载失败则退回dagre。
  let useElk = false;
  const elkLayouts = await loadElkLayouts();
  if (elkLayouts) {{
    mermaid.registerLayoutLoaders(elkLayouts);
    useElk = true;
  }} else {{
    console.warn('ELK布局加载失败，使用默认dagre布局');
  }}

  function mermaidConfig(withElk) {{
    const base = baseConfig();
    return withElk
      ? {{ ...base, flowchart: {{ ...base.flowchart, defaultRenderer: 'elk' }} }}
      : {{ ...base, flowchart: {{ ...base.flowchart, defaultRenderer: 'dagre' }} }};
  }}
  const defs = {defs_json};
  const edgeDefs = {edge_defs_json};
  const tip = document.getElementById('cgraph-tip');
  const wrap = document.getElementById('cgraph-wrap');
  const inner = document.getElementById('cgraph');

  function showTip(title, text, e) {{
    tip.querySelector('.cgraph-tip-name').textContent = title || '';
    tip.querySelector('.cgraph-tip-name').style.display = title ? 'block' : 'none';
    tip.querySelector('.cgraph-tip-body').textContent = text;
    tip.style.display = 'block';
    const w = tip.offsetWidth;
    tip.style.left = Math.max(8, Math.min(e.clientX + 16, window.innerWidth - w - 10)) + 'px';
    // 卡片变高了：靠下时翻转到鼠标上方，避免探出屏幕看不到后半段
    const h = tip.offsetHeight;
    tip.style.top = (e.clientY + 16 + h > window.innerHeight ? e.clientY - h - 10 : e.clientY + 16) + 'px';
  }}
  function hideTip() {{ tip.style.display = 'none'; }}

  // 渲染入口独立成函数：主题切换时用当前配色整图重画
  let renderSeq = 0;
  async function renderChart() {{
    const seq = ++renderSeq;
    try {{
      let svg;
      try {{
        // 优先ELK；个别图ELK布局器可能溢出报错，退回dagre重画一次
        mermaid.initialize(mermaidConfig(useElk));
        ({{ svg }} = await mermaid.render('cgraph' + seq + 'a', {json.dumps(chart)}));
      }} catch (err) {{
        if (!useElk) throw err;
        console.warn('ELK渲染失败，回退dagre', err);
        mermaid.initialize(mermaidConfig(false));
        ({{ svg }} = await mermaid.render('cgraph' + seq + 'b', {json.dumps(chart)}));
      }}
      if (seq !== renderSeq) return;  // 渲染期间又触发了重绘，丢弃过期结果
      inner.innerHTML = svg;
      const svgEl = inner.querySelector('svg');

      // ---- 缩放与平移 ----
      const bb = svgEl.getBBox();
      let scale = 1;
      function apply() {{
        inner.style.width = bb.width * scale + 'px';
        inner.style.height = bb.height * scale + 'px';
        document.getElementById('zlevel').textContent = Math.round(scale * 100) + '%';
      }}
      // 自适应铺满容器宽度（等同fit模式），窗口尺寸变化时重新适配
      function fitWidth() {{
        scale = Math.min(3, Math.max(0.4, (wrap.clientWidth - 26) / bb.width));
        apply();
      }}
      fitWidth();
      // 以下全部用 on* 属性绑定（重复渲染时自动覆盖，不会叠加监听器）
      window.onresize = fitWidth;
      function zoomBy(f) {{ scale = Math.min(3, Math.max(0.5, scale * f)); apply(); }}
      document.getElementById('zin').onclick = () => zoomBy(1.25);
      document.getElementById('zout').onclick = () => zoomBy(0.8);
      document.getElementById('zreset').onclick = () => {{ wrap.scrollTo(0, 0); fitWidth(); }};
      wrap.onwheel = e => {{
        e.preventDefault();
        zoomBy(e.deltaY < 0 ? 1.15 : 0.87);
      }};
      // 拖拽平移：用Pointer事件，鼠标和平板触屏都能拖
      let down = null;
      wrap.onpointerdown = e => {{
        down = {{ x: e.clientX, y: e.clientY, l: wrap.scrollLeft, t: wrap.scrollTop }};
        wrap.classList.add('dragging');
      }};
      window.onpointermove = e => {{
        if (!down) return;
        wrap.scrollLeft = down.l - (e.clientX - down.x);
        wrap.scrollTop = down.t - (e.clientY - down.y);
      }};
      window.onpointerup = () => {{ down = null; wrap.classList.remove('dragging'); }};
      // 触屏上接管手势（否则拖拽会被浏览器原生滚动抢走），鼠标端无感
      wrap.style.touchAction = 'none';

      // ---- 全屏 ----
      document.getElementById('zfull').onclick = async () => {{
        if (document.fullscreenElement) {{ document.exitFullscreen(); return; }}
        try {{ await wrap.requestFullscreen(); }}
        catch (err) {{
          // iframe不允许真全屏时退化为页面内铺满
          wrap.style.position = 'fixed'; wrap.style.inset = '0';
          wrap.style.zIndex = 999; wrap.style.height = '100vh';
        }}
      }};
      document.onfullscreenchange = () => fitWidth();

    // ---- 导出PNG：SVG序列化 -> Image -> 2倍分辨率canvas下载 ----
    document.getElementById('zpng').onclick = async () => {{
      const btn = document.getElementById('zpng');
      const old = btn.textContent;
      btn.textContent = '生成中...';
      try {{
        const clone = svgEl.cloneNode(true);
        clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        clone.setAttribute('width', bb.width);
        clone.setAttribute('height', bb.height);
        const xml = new XMLSerializer().serializeToString(clone);
        const img = new Image();
        await new Promise((res, rej) => {{
          img.onload = res; img.onerror = () => rej(new Error('SVG转换失败'));
          img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(xml)));
        }});
        const pad = 24, scale = 2;  // 2倍分辨率保证清晰
        const canvas = document.createElement('canvas');
        canvas.width = (bb.width + pad * 2) * scale;
        canvas.height = (bb.height + pad * 2) * scale;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = cgPalette().wrapBg;  // 与页面主题同底色，深色模式下浅色节点文字才可见
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, pad, pad, bb.width * scale, bb.height * scale);
        const a = document.createElement('a');
        a.download = '概念图谱.png';
        a.href = canvas.toDataURL('image/png');
        a.click();
      }} catch (err) {{ alert('导出失败: ' + err.message); }}
      btn.textContent = old;
    }};

    // ---- 悬停讲解：节点显示概念名+详细定义，边显示完整关系词 ----
    svgEl.querySelectorAll('g.node').forEach(n => {{
      const def = defs[n.textContent.replace(/\\s+/g, '')];
      if (!def) return;
      n.style.cursor = 'help';
      n.addEventListener('mousemove', e => showTip(n.textContent.replace(/\\s+/g, ' '), def, e));
      n.addEventListener('mouseleave', hideTip);
    }});
    svgEl.querySelectorAll('g.edgeLabel').forEach(el => {{
      const txt = edgeDefs[el.textContent.replace(/\\s+/g, '')];
      if (!txt) return;
      el.style.cursor = 'help';
      el.addEventListener('mousemove', e => showTip('', txt, e));
      el.addEventListener('mouseleave', hideTip);
    }});
    }} catch (e) {{
      inner.textContent = '图谱渲染失败: ' + e.message;
    }}
  }}
  await renderChart();

  // 深浅色实时切换：监听宿主页面根节点的属性变化（Streamlit切主题改class/变量），
  // 亮度反转时用新配色整图重画；跨域受限时静默放弃（仅首载时判断主题）
  try {{
    const pdoc = window.parent.document;
    let deb = null;
    const mo = new MutationObserver(() => {{
      clearTimeout(deb);
      deb = setTimeout(() => {{
        const d = detectDark();
        if (d !== curDark) {{ curDark = d; applyUiPalette(); renderChart(); }}
      }}, 200);
    }});
    const opt = {{ attributes: true, attributeFilter: ['class', 'style', 'data-theme'] }};
    mo.observe(pdoc.documentElement, opt);
    mo.observe(pdoc.body, opt);
  }} catch (e) {{ /* 忽略 */ }}
</script>
""",
        height=height,
    )


# --- 摘要富文本渲染：逻辑关系徽章 + 术语同色下划线 ---
# 与rag_core.SUMMARY_PROMPT的输出约定对应：每条要点以「逻辑词」开头，
# 关键术语用[[术语]]标记；旧版无标记的摘要也能正常渲染（正则不匹配就原样通过）

# 逻辑词 -> (文字/边框色, 背景色)，语义近似的意思色系相近：问题/转折偏红，方案/结果偏绿
# 深色模式单独一套：底色换成深色调、文字提亮，否则浅色徽章在深底上刺眼
LOGIC_STYLES = {
    "问题": ("#cd6a6a", "#fdeaea"),
    "转折": ("#cd6a6a", "#fdeaea"),
    "因果": ("#d29a3a", "#fdf3e0"),
    "对比": ("#4fa8a2", "#eaf7f6"),
    "并列": ("#5b8fc9", "#eaf3fc"),
    "递进": ("#9a72c4", "#f3ecfa"),
    "目的": ("#5ea575", "#e8f6ec"),
    "方法": ("#5b8fc9", "#eaf3fc"),
    "方案": ("#5ea575", "#e8f6ec"),
    "结果": ("#5ea575", "#e8f6ec"),
    "补充": ("#7d8aa0", "#eef0f4"),
}
LOGIC_STYLES_DARK = {
    "问题": ("#e28b8b", "#3a2528"),
    "转折": ("#e28b8b", "#3a2528"),
    "因果": ("#e0b06a", "#39301d"),
    "对比": ("#6cc4be", "#1c3331"),
    "并列": ("#7ea9dc", "#1f2c3e"),
    "递进": ("#b58fe0", "#2f2440"),
    "目的": ("#79c393", "#1d3327"),
    "方法": ("#7ea9dc", "#1f2c3e"),
    "方案": ("#79c393", "#1d3327"),
    "结果": ("#79c393", "#1d3327"),
    "补充": ("#9aa7bd", "#2a3140"),
}
# 术语下划线轮换色：同一术语固定同一颜色，同色即同词，扫一眼就能追踪指代
TERM_COLORS = ["#5b8fc9", "#d29a3a", "#9a72c4", "#cd6a6a", "#4fa8a2", "#5ea575"]
TERM_COLORS_DARK = ["#7ea9dc", "#e0b06a", "#b58fe0", "#e28b8b", "#6cc4be", "#79c393"]

# 摘要排版层级：小节标题 > 段落导语 > 要点条目，字号/行距拉开方便扫读。
# 整体基准1.35em（默认字号偏小，翻倍又过大，取中间偏上），内部用em按层级缩放
_SUMMARY_CSS = """
<style>
.sum-rich {{ font-size: 1.35em; }}
.sum-rich h2 {{
  font-size: 1.3em; font-weight: 700; line-height: 1.4;
  margin: 1.15em 0 .5em; padding: 1px 0 1px 10px;
  border-left: 4px solid {accent};
}}
.sum-rich p {{ font-size: 1em; line-height: 1.75; margin: .15em 0 .7em; }}
.sum-rich ul {{ margin: .2em 0 .9em; }}
.sum-rich li {{ font-size: .94em; line-height: 1.8; margin: .3em 0; }}
</style>
"""


def render_summary_markdown(text):
    """把摘要渲染成富文本：「逻辑词」-> 徽章，[[术语]] -> 彩色下划线，
    并按 小节标题/导语/要点 做字号层级。深色模式自动换配色。"""
    import re

    is_dark = (st.context.theme.type or "light") == "dark"
    logic_styles = LOGIC_STYLES_DARK if is_dark else LOGIC_STYLES
    term_colors_palette = TERM_COLORS_DARK if is_dark else TERM_COLORS
    fallback = logic_styles["补充"]

    # 先转义再注入受控的span，摘要文本里若混入<>&不会破坏页面结构
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    term_colors = {}
    def _term_repl(m):
        term = m.group(1)
        if term not in term_colors:
            term_colors[term] = term_colors_palette[len(term_colors) % len(term_colors_palette)]
        c = term_colors[term]
        return (f'<span style="font-weight:600;border-bottom:2.5px solid {c};'
                f'padding-bottom:1px;">{term}</span>')

    def _tag_repl(m):
        label = m.group(2)
        border, bg = logic_styles.get(label, fallback)
        return (f'{m.group(1)}<span style="display:inline-block;font-size:14px;'
                f'color:{border};background:{bg};border:1px solid {border};'
                f'border-radius:4px;padding:0 8px;margin-right:8px;'
                f'white-space:nowrap;transform:translateY(-2px);">{label}</span>')

    esc = re.sub(r"\[\[(.+?)\]\]", _term_repl, esc)
    esc = re.sub(r"^(\s*(?:[-*]\s*)?)「([^」]{1,4})」", _tag_repl, esc, flags=re.M)

    # 用div包一层做排版作用域（div后留空行，后面的内容仍走markdown解析）
    css = _SUMMARY_CSS.format(accent="#7ea9dc" if is_dark else "#5b8fc9")
    st.markdown(css + '<div class="sum-rich">\n\n' + esc + "\n\n</div>",
                unsafe_allow_html=True)


# --- 侧边栏：文献库管理与来源过滤 ---
# 每个会话第一次进入时：先回到上次使用的检索范围（否则新开界面默认"全部文献"，
# 而图谱是按范围保存的，范围对不上就会显示成没生成过），再恢复该范围的摘要/概念图谱
if not st.session_state.get("artifacts_restored", False):
    last = None
    # 优先级1：URL里的范围参数（F5刷新后原样保留，最可靠）
    qp = st.query_params.get_all("scope")
    if qp and all(s in st.session_state.sources for s in qp):
        last = qp
    # 优先级2：服务端记录的上次范围
    if last is None:
        last = load_last_scope()
    valid = last is not None and all(s in st.session_state.sources for s in last)
    if not valid:
        # 没有有效范围记录（首次使用新版/记录失效）时，若恰好只有一个已保存图谱
        # 的范围且其文献都还在库里，就直接回到那个范围——别让用户以为图谱丢了
        saved = [
            s for s in list_artifacts("graph")
            if s and all(x in st.session_state.sources for x in s)
        ]
        if len(saved) == 1:
            last, valid = saved[0], True
    if valid:
        st.session_state.selected_sources = last
    invalidate_cached_results()
    st.session_state.artifacts_restored = True
    # 浏览器刷新后会自动恢复上次的滚动位置，页面直接落到底部，观感像"丢了"。
    # 只在页面刚加载/刷新的几秒内拉回顶部（用父页面performance计时区分真刷新
    # 与会话内rerun——rerun时本段不执行，点按钮不会把页面拽来拽去）
    render_html(
        "<script>(function(){"
        "var go=function(){try{var w=window.parent,p=w.document;"
        "if(!w.performance||w.performance.now()>5000)return;"
        "w.history.scrollRestoration='manual';"
        "if(p.activeElement&&p.activeElement.blur)p.activeElement.blur();"
        "w.scrollTo(0,0);"
        "var m=p.querySelector('section.main,[data-testid=stApp],[data-testid=stMain]');"
        "if(m)m.scrollTop=0;}catch(e){}};"
        "go();setTimeout(go,600);setTimeout(go,1800);})();</script>",
        height=0,
    )

with st.sidebar:
    st.header("📚 文献库")
    st.caption(f"已入库：{len(st.session_state.sources)} 篇")
    for s in st.session_state.sources:
        st.markdown(f"- {s}")

    st.divider()
    st.header("1. 上传文献（可多选）")
    uploaded_files = st.file_uploader("选择PDF文件", type="pdf", accept_multiple_files=True)
    st.caption("模型：GLM / Embedding：智谱 embedding-3")

    if uploaded_files:
        if st.button("📥 处理并入库", use_container_width=True):
            try:
                # 重新上传=替换内容：先把引用这些文献的旧摘要/图谱缓存删掉
                delete_artifacts_for([f.name for f in uploaded_files])
                clear_cached_results()
                bar, callback = make_progress_bar("准备入库...")
                total_chunks = 0
                for i, f in enumerate(uploaded_files):
                    # 多篇文件合成一条进度：第i篇的进度占比 = (i + 篇内进度) / 总篇数
                    def scoped(frac, msg, idx=i, name=f.name):
                        callback((idx + frac) / len(uploaded_files), f"[{name}] {msg}")

                    total_chunks += add_pdf(
                        st.session_state.vectorstore, f, f.name, progress_callback=scoped
                    )
                st.session_state.sources = list_sources(st.session_state.vectorstore)
                _load_source_pages.clear()  # 文献内容变了，阅读器缓存作废
                invalidate_cached_results()  # 尝试恢复与本次范围匹配的剩余缓存
                fade_out_progress(bar, f"✅ 完成！共入库 {total_chunks} 块")
                st.success(f"✅ {len(uploaded_files)} 篇文献处理完成！")
            except Exception as e:
                st.error(f"处理失败: {e}")

    # 管理文献：查看/删除已入库的单篇
    with st.expander("🗑️ 管理文献"):
        if st.session_state.sources:
            victim = st.selectbox("选择要删除的文献:", st.session_state.sources)
            if st.button("删除该文献", use_container_width=True):
                remove_source(st.session_state.vectorstore, victim)
                delete_artifacts_for([victim])  # 引用它的摘要/图谱缓存一并作废
                if victim in st.session_state.selected_sources:
                    st.session_state.selected_sources.remove(victim)
                st.session_state.sources = list_sources(st.session_state.vectorstore)
                _load_source_pages.clear()
                invalidate_cached_results()
                st.rerun()
        else:
            st.caption("文献库为空")

    # 查看文献：从向量库把入库文本按页拼回来，弥补"传完就看不到原文"的问题
    with st.expander("📄 查看文献内容"):
        if st.session_state.sources:
            view_src = st.selectbox(
                "选择文献:", st.session_state.sources, key="view_source"
            )
            pages = _load_source_pages(view_src)
            if not pages:
                st.caption("该文献没有可展示的文本")
            else:
                page_nums = sorted(pages)
                st.caption(f"共 {len(page_nums)} 页（按入库文本块拼回，公式/表格可能有少量失真）")
                page_sel = st.selectbox(
                    "跳到第几页:",
                    [f"第 {p + 1} 页" for p in page_nums],
                    key="view_page",
                )
                cur = page_nums[[f"第 {p + 1} 页" for p in page_nums].index(page_sel)]
                st.text_area(
                    "页面内容（可复制）",
                    "\n\n".join(pages[cur]),
                    height=360,
                    key=f"view_page_{cur}",
                )
        else:
            st.caption("文献库为空")

    st.divider()
    st.header("2. 检索范围")
    selected = st.multiselect(
        "按文献过滤（不选 = 检索全部）:",
        options=st.session_state.sources,
        default=st.session_state.selected_sources,
    )
    if selected != st.session_state.selected_sources:
        st.session_state.selected_sources = selected
        invalidate_cached_results()
        st.rerun()

# 库里一篇文献都没有时，先引导上传
if not st.session_state.sources:
    st.info("👈 请先在左侧上传PDF文献并点击「处理并入库」。知识库持久化保存，重启应用也不用重新上传。")
    st.stop()

# --- 文献导航：选中单篇即可查看/生成它的摘要与概念图谱（未生成则显示生成按钮） ---
sel = st.session_state.selected_sources
if len(sel) == 1 and sel[0] in st.session_state.sources:
    current_label = sel[0]
elif not sel:
    current_label = "（全部文献）"
else:
    current_label = "（多选范围）"

browse_options = ["（全部文献）"] + st.session_state.sources
if current_label == "（多选范围）":
    browse_options.append("（多选范围）")  # 占位，避免侧边栏多选时下拉框无处落脚
choice = st.selectbox(
    "📚 选择文献（摘要 / 概念图谱 / 问答都会针对这个范围）:",
    browse_options,
    index=browse_options.index(current_label),
    key="browse_source",
)
if choice != current_label:
    st.session_state.selected_sources = (
        [] if choice in ("（全部文献）", "（多选范围）") else [choice]
    )
    invalidate_cached_results()  # 自动切换到该范围已保存的摘要/图谱缓存
    st.rerun()

filter_label = "、".join(st.session_state.selected_sources) if st.session_state.selected_sources else "全部文献"
st.caption(f"当前范围：**{filter_label}**（也可在左侧多选多篇检索）")

# 混合检索器惰性构建：不再在每次页面加载时同步重建BM25索引（拉全部文本块+jieba
# 分词要1-2秒，会让切换文献时整页卡住），改成第一次提问时才建，之后按范围缓存
sources_key = tuple(st.session_state.selected_sources)


def ensure_retriever():
    """确保检索器与当前检索范围匹配；需要新建时在提问处显示提示。"""
    if st.session_state.retriever is None or st.session_state.retriever_key != sources_key:
        with st.spinner("正在建立检索索引（首次提问需1-2秒）..."):
            st.session_state.retriever = get_hybrid_retriever(
                st.session_state.vectorstore,
                sources=list(st.session_state.selected_sources),
            )
            st.session_state.retriever_key = sources_key

tab_summary, tab_qa, tab_concept, tab_quiz = st.tabs(
    ["📋 论文速读", "📖 问答", "🧠 概念图谱", "📝 自测出题"]
)

# --- Tab 0: 论文速读（结构化摘要） ---
with tab_summary:
    st.caption("AI 通读全文（开头/中间/结尾均匀采样），生成 研究背景 / 核心创新点 / 方法论 / 实验结论 四段式摘要。生成结果自动保存在本地，重启应用或切换检索范围后再回来无需重新生成。")

    if st.button("📝 生成结构化摘要", key="summary_btn"):
        try:
            # st.session_state只能在主线程访问：进后台线程前先把值取出来
            vs = st.session_state.vectorstore
            srcs = list(st.session_state.selected_sources)
            st.session_state.summary = run_with_progress_bar(
                lambda: generate_paper_summary(vs, sources=srcs),
                "正在精读论文...",
            )
            save_artifact("summary", srcs, st.session_state.summary)  # 落盘，重启后仍可看
        except Exception as e:
            st.error(f"摘要生成失败: {e}")

    if st.session_state.summary:
        render_summary_markdown(st.session_state.summary)
        st.caption("💡 要点前的彩色标签是逻辑关系（问题/因果/转折/并列/对比…）；带色下划线的是关键术语，同色=同一术语，全文只标第一次出现。")
    else:
        st.info("选中上方文献后点击按钮生成（模型会先深度思考，约需 1-2 分钟）。生成结果按文献范围保存，下次选中它直接显示，无需重新生成。")

# --- Tab 1: 问答（多轮对话） ---
with tab_qa:
    st.caption("💬 支持追问：针对文献内容连续对话，模型会结合上下文理解“刚才那个公式”这类追问。")

    # 概念图谱联动：把待提问的概念当作新消息处理
    pending = st.session_state.pending_question
    if pending:
        st.session_state.messages.append({"role": "user", "content": pending})
        st.session_state.pending_question = None

    # 渲染完整对话历史（每次重跑都从session_state重绘，天然就是缓存）
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("docs"):
                with st.expander("📖 引用原文"):
                    for i, doc in enumerate(msg["docs"]):
                        page = doc.metadata.get("page", "?")
                        src = doc.metadata.get("source", "")
                        label = f"第 {page + 1} 页" if isinstance(page, int) else f"第 {page} 页"
                        st.markdown(f"**来源 {i + 1}** · {src} · {label}:")
                        st.code(doc.page_content[:500])

    # chat_input提交后rerun，把值存到session_state带进来；此处回答并落盘历史
    new_query = st.session_state.pop("chat_input_value", None)
    if new_query:
        st.session_state.messages.append({"role": "user", "content": new_query})

    if pending or new_query:
        question = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            try:
                ensure_retriever()  # 首次提问时才建索引，页面加载不必等它
                with st.spinner("正在思考..."):
                    result = answer_question(
                    st.session_state.vectorstore,
                    question,
                    history=st.session_state.messages[:-1],
                    sources=list(st.session_state.selected_sources),
                    retriever=st.session_state.retriever,
                )
                st.markdown(result["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "docs": result["source_documents"],
                })
            except Exception as e:
                st.error(f"问答失败: {e}")

# --- 底部输入框：st.chat_input只能放顶层，不能放进Tab内 ---
if st.session_state.sources:
    user_input = st.chat_input("针对文献内容提问，可连续追问...")
    if user_input:
        st.session_state.chat_input_value = user_input
        st.rerun()  # 重跑后问答Tab处理这条新消息

# --- Tab 2: 概念图谱 ---
with tab_concept:
    if st.button("🔍 提取陌生概念", key="extract_btn"):
        try:
            bar, callback = make_progress_bar("准备提取概念...")
            res = extract_concepts(
                st.session_state.vectorstore,
                sources=list(st.session_state.selected_sources),
                progress_callback=callback,
            )
            fade_out_progress(bar, "✅ 提取完成")
            st.session_state.concepts = res["concepts"]
            st.session_state.relations = res["relations"]
            st.session_state.quiz = None  # 概念变了，旧题目作废
            # 落盘缓存：重启/切换检索范围后仍能直接查看，不必重新提取
            save_artifact(
                "graph",
                list(st.session_state.selected_sources),
                {"concepts": res["concepts"], "relations": res["relations"]},
            )
            if res.get("skipped"):
                st.warning(
                    f"⚠️ 有 {len(res['skipped'])} 个文本块两次解析失败被跳过（详情见终端日志），"
                    "核心概念可能不全。首块内容：\n\n> " + res["skipped"][0]["chunk_snippet"][:150]
                )
        except Exception as e:
            st.error(f"提取失败: {e}")

    if st.session_state.concepts and any(len(c.get("definition", "")) < 80 for c in st.session_state.concepts):
        # 旧版提取的解释只有一句话；不想让用户重新提取整个图谱，提供一键扩写
        if st.button("✍️ 扩写概念解释（更详细+举例，约1-2分钟）", key="enrich_btn"):
            try:
                bar, callback = make_progress_bar("准备扩写概念解释...")
                new_concepts = enrich_concept_definitions(
                    st.session_state.vectorstore,
                    st.session_state.concepts,
                    st.session_state.relations,
                    sources=list(st.session_state.selected_sources),
                    progress_callback=callback,
                )
                fade_out_progress(bar, "✅ 扩写完成")
                st.session_state.concepts = new_concepts
                save_artifact(
                    "graph",
                    list(st.session_state.selected_sources),
                    {"concepts": new_concepts, "relations": st.session_state.relations},
                )
                st.rerun()
            except Exception as e:
                st.error(f"扩写失败: {e}")

    if st.session_state.concepts:
        if st.session_state.relations:
            st.subheader(f"概念关系图（{len(st.session_state.relations)} 条关系）")
            concept_defs = {c["name"]: c["definition"] for c in st.session_state.concepts}
            built = build_mermaid_chart(st.session_state.concepts, st.session_state.relations)
            render_mermaid(built["chart"], concept_defs=concept_defs, edge_defs=built["edge_defs"])

            # 图例：主题分类配色说明
            legend = "&nbsp;&nbsp;".join(
                f'<span style="color:{border};font-size:18px;">●</span>'
                f'<span style="color:#444;"> {cat}</span>'
                for cat, (fill, border) in built["cat_colors"].items()
            )
            st.markdown(legend, unsafe_allow_html=True)
            st.caption("💡 滚轮缩放 · 拖拽平移 · 悬停节点看定义 · 悬停边看完整关系 · 字体越大越核心 · 图谱自动保存，重启后仍可查看")

            # 概念 -> 问答联动：选一个概念直接去对话里深挖
            link_concept = st.selectbox(
                "💬 选一个概念，到「📖 问答」里深入讲解:",
                ["（选择概念）"] + [c["name"] for c in st.session_state.concepts],
                key="link_concept_select",
            )
            if link_concept != "（选择概念）" and st.button("去问答Tab详细讲解", key="link_ask_btn"):
                st.session_state.pending_question = (
                    f"请详细讲解概念「{link_concept}」：它在这篇文献里的含义、作用，"
                    "以及和哪些概念相关，用大白话举例说明。"
                )
                st.rerun()

        st.subheader(f"陌生概念（{len(st.session_state.concepts)} 个）")
        for c in st.session_state.concepts:
            with st.expander(f"**{c['name']}**"):
                st.write(c["definition"])
    else:
        st.info("点击上方按钮，让AI通读文献并提取对初学者陌生的概念（约1-2分钟）。提取结果按文献范围保存，重启应用后选中它仍可直接查看。")
        # 当前范围没有图谱，但别的范围可能已经生成过——告诉用户直接选它，别重复花钱生成
        saved = [s for s in list_artifacts("graph") if s]
        if saved and st.session_state.selected_sources not in saved:
            names = "\n".join(f"- {'、'.join(s)}" for s in saved)
            st.markdown(f"💡 **已保存过图谱的范围**（在上方下拉框选中即可直接查看，无需重新提取）：\n{names}")

# --- Tab 3: 自测出题 ---
# 一次生成一组（默认5道）：一半考所选概念，其余考相邻概念及它们与所选概念的关系。
# 题目按"考试模式"渲染：全部作答后一键交卷，统一判分+逐题解析。

def _clear_quiz_choices():
    """新一组题目作废旧作答记录（radio的key按题号固定，必须清掉否则会显示旧选项）。"""
    for i in range(10):
        st.session_state.pop(f"quiz_choice_{i}", None)


with tab_quiz:
    if not st.session_state.concepts:
        st.info("请先在「🧠 概念图谱」Tab提取概念，然后在这里针对概念自测。")
    else:
        names = [c["name"] for c in st.session_state.concepts]
        concept = st.selectbox("选择要考查的概念（会连带考查图中与它相连的概念和关系）:", names)

        if st.button("🎯 生成一组题目（5道）", key="quiz_btn"):
            try:
                definition = next(
                    (c["definition"] for c in st.session_state.concepts if c["name"] == concept), ""
                )
                # 同上：session_state的值先取到局部变量再进后台线程
                vs = st.session_state.vectorstore
                srcs = list(st.session_state.selected_sources)
                concept_list = st.session_state.concepts
                relation_list = st.session_state.relations
                st.session_state.quiz = run_with_progress_bar(
                    lambda: generate_quiz_batch(
                        vs, concept, definition,
                        concepts=concept_list, relations=relation_list, sources=srcs,
                    ),
                    "正在出一组题目（5道，约1-2分钟）...",
                )
                st.session_state.quiz_answered = False
                _clear_quiz_choices()
            except Exception as e:
                st.error(f"出题失败: {e}")

        quiz_list = st.session_state.quiz
        if quiz_list:
            answered = st.session_state.get("quiz_answered", False)

            if answered:
                correct_count = sum(
                    1 for i, q in enumerate(quiz_list)
                    if st.session_state.get(f"quiz_choice_{i}") == q.options[q.answer_index]
                )
                ratio = correct_count / len(quiz_list)
                st.markdown(
                    f"### 🏅 得分：{correct_count} / {len(quiz_list)}"
                    + ("　太棒了，掌握得很扎实！" if ratio >= 0.8
                       else "　不错，再看看错题解析～" if ratio >= 0.5
                       else "　建议回到概念图谱和问答里再巩固一下")
                )

            for i, q in enumerate(quiz_list):
                st.markdown(f"### 第 {i + 1} 题")
                st.markdown(q.question)
                choice = st.radio(
                    "选择你的答案:",
                    q.options,
                    key=f"quiz_choice_{i}",
                    index=None,
                    disabled=answered,
                    label_visibility="collapsed",
                )
                if answered:
                    chosen = st.session_state.get(f"quiz_choice_{i}")
                    correct = q.options[q.answer_index]
                    if chosen == correct:
                        st.success("✅ 回答正确！")
                    elif chosen is None:
                        st.warning(f"⚠️ 这题没作答。正确答案是：{correct}")
                    else:
                        st.error(f"❌ 回答错误。正确答案是：{correct}")
                    st.info(f"**解析：** {q.explanation}")

            if not answered:
                if st.button("✅ 提交全部答案", key="quiz_submit_btn"):
                    missing = [i + 1 for i in range(len(quiz_list))
                               if st.session_state.get(f"quiz_choice_{i}") is None]
                    if missing:
                        st.warning(f"还有第 {'、'.join(map(str, missing))} 题没作答，请先选完再交卷。")
                    else:
                        st.session_state.quiz_answered = True
                        st.rerun()

            if answered:
                if st.button("🔄 再出一组", key="quiz_again_btn"):
                    st.session_state.quiz = None
                    st.session_state.quiz_answered = False
                    _clear_quiz_choices()
                    st.rerun()
