import streamlit as st
import logging
import sys
import os
from config import COLLECTION_NODES, VLLM_URL, DB_CONFIG, LLM_MODEL
from model_loader import load_embedding_model, load_tokenizer_e5, load_tokenizer_llm, load_llm_client
from qdrant import load_client
from pipeline import run_full_pipeline
from tree_builder import build_tree_from_text, save_tree_to_qdrant
from qdrant_client.http.models import VectorParams, Distance


# ── SafeStdout ────────────────────────────────────────────────────────────────
class SafeStdout:
    def __init__(self, wrapped):
        self._wrapped = wrapped
    def write(self, s):
        try:
            self._wrapped.write(s)
        except BrokenPipeError:
            pass
    def flush(self):
        try:
            self._wrapped.flush()
        except BrokenPipeError:
            pass
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

sys.stdout = SafeStdout(sys.stdout)


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/kaggle/working/app.log"),
        logging.StreamHandler(),
    ],
)


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hierarchical IRCoT RAG",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@300;400;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Sora', sans-serif;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] {
    display: block !important;
    height: 0rem !important;
    background: transparent !important;
}

/* ── Chat bubbles ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.6rem 0;
}
.msg-user .bubble {
    background: #1a472a;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.7rem 1.1rem;
    max-width: 70%;
    font-size: 1.05rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.msg-assistant {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 0.55rem;
    margin: 0.6rem 0;
}
.msg-assistant .avatar {
    width: 28px; height: 28px;
    background: #2d6a4f;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; flex-shrink: 0; margin-top: 2px;
}
.msg-assistant .bubble {
    background: #f5f5f5;
    color: #1a1a1a;
    border-radius: 4px 18px 18px 18px;
    padding: 0.7rem 1.1rem;
    max-width: 70%;
    font-size: 1.05rem;
    line-height: 1.6;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.msg-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #888888;
    margin-top: 0.2rem;
    padding-left: 36px;
}

/* ── Input box ── */
div[data-testid="stTextArea"] textarea {
    font-size: 1.05rem !important;
    padding: 1rem 1.2rem !important;
    resize: none !important;
}
textarea::placeholder {
    color: #999999 !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.empty-state .icon { font-size: 5rem; }
.empty-state .title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    color: #1a6b3c;
    margin-top: 1rem;
    letter-spacing: 2px;
    font-weight: 600;
}
.empty-state .hint {
    font-size: 1rem;
    color: #666666;
    margin-top: 0.6rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f8f8f8 !important;
    border-right: 1px solid #e0e0e0 !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.88rem !important;
    color: #1a6b3c !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.3px;
}
[data-testid="stSidebar"] hr {
    border-color: #c0d8c0 !important;
}

/* ── Input placeholder ── */
input::placeholder {
    color: #999999 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #ffffff; }
::-webkit-scrollbar-thumb { background: #2d6a4f; border-radius: 2px; }

/* ── Send button ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: #238636 !important;
    border: 1px solid #2ea043 !important;
    border-radius: 10px !important;
    font-size: 1.3rem !important;
    padding: 0.6rem 1.4rem !important;
    height: 75px !important;
    width: auto !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #2ea043 !important;
}

/* ── Stats panel ── */
.stat-block {
    background: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
}
.stat-block .s-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stat-block .s-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    color: #1a6b3c;
    font-weight: 500;
    margin-top: 1px;
}

/* Sembunyikan teks keterangan ukuran file pada uploader */
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p {
    display: none !important;
}

/* Bunuh semua teks dalam file uploader */
div[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] {
    display: none;
}


</style>

""", unsafe_allow_html=True)


# ── Dataset type mapping ──────────────────────────────────────────────────────
DATASET_TYPE_MAP = {
    "NarrativeQA":  "openended",
    "Qasper":       "openended",
    "Quality":      "multiple_choice",
    "TyDiQA":       "openended",
    "User Dataset": "openended",
}


# ── Helper: Qdrant client (cached) ───────────────────────────────────────────
@st.cache_resource
def get_qdrant_client(dataset: str):
    return load_client(dataset)

@st.cache_resource
def get_universal_client():
    os.makedirs("/kaggle/working/user_qdrant_db", exist_ok=True)
    return load_client("User")


# ── Helper: get all doc_ids ───────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_all_doc_ids(dataset: str, collection_name: str) -> list:
    client = get_qdrant_client(dataset) if dataset != "User Dataset" else get_universal_client()
    doc_ids = set()
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for rec in records:
            did = rec.payload.get("doc_id")
            if did:
                doc_ids.add(str(did))
        if next_offset is None:
            break
        offset = next_offset
    return sorted(doc_ids)


# ── Helper: ensure user collections ──────────────────────────────────────────
def ensure_user_collections(client):
    try:
        if not client.collection_exists("user_hierarchical_nodes"):
            client.create_collection(
                collection_name="user_hierarchical_nodes",
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            client.create_payload_index("user_hierarchical_nodes", "doc_id", field_type="keyword")
            client.create_payload_index("user_hierarchical_nodes", "layer",  field_type="integer")
        if not client.collection_exists("user_docs_stats"):
            client.create_collection(
                collection_name="user_docs_stats",
                vectors_config=VectorParams(size=1, distance=Distance.COSINE)
            )
    except Exception as e:
        st.warning(f"Error membuat collection user: {e}")


# ── Session State Init ────────────────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "show_panel":   False,
    "last_result":  None,
    "input_key":    0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Load Models ───────────────────────────────────────────────────────────────
embedding_model  = load_embedding_model()
tokenizer_e5     = load_tokenizer_e5()
tokenizer_llm    = load_tokenizer_llm()
llm_client       = load_llm_client()

universal_client = get_universal_client()
ensure_user_collections(universal_client)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌲 Hierarchical IRCoT RAG")
    st.markdown("---")

    st.markdown("**📂 Dataset**")
    dataset_options  = list(COLLECTION_NODES.keys()) + ["User Dataset"]
    
    # Cek apakah ada permintaan force dataset (setelah upload)
    force_dataset = st.session_state.pop("force_dataset", None)
    if force_dataset and force_dataset in dataset_options:
        default_dataset_idx = dataset_options.index(force_dataset)
    else:
        default_dataset_idx = 0
    
    selected_dataset = st.selectbox(
        "Pilih Dataset",
        dataset_options,
        index=default_dataset_idx,
        label_visibility="collapsed",
    )

    if selected_dataset == "User Dataset":
        collection_name = "user_hierarchical_nodes"
        qdrant_client   = universal_client
    else:
        collection_name = COLLECTION_NODES[selected_dataset]
        qdrant_client   = get_qdrant_client(selected_dataset)

    dataset_type = DATASET_TYPE_MAP[selected_dataset]

    st.markdown("---")
    st.markdown("**🔍 Filter Dokumen**")
    try:
        all_doc_ids = get_all_doc_ids(selected_dataset, collection_name)
        if all_doc_ids:
            force_doc = st.session_state.pop("force_doc", None)
            if force_doc and force_doc in all_doc_ids:
                default_doc_idx = all_doc_ids.index(force_doc)
            else:
                default_doc_idx = 0
            
            selected_doc = st.selectbox(
                "Pilih Doc ID",
                all_doc_ids,
                index=default_doc_idx,
                label_visibility="collapsed",
            )
            doc_id_filter = selected_doc
        else:
            st.info("Tidak ada dokumen di collection ini.")
            doc_id_filter = None
    except Exception as e:
        st.warning(f"Gagal load doc list: {e}")
        manual = st.text_input("Doc ID (manual)")
        doc_id_filter = manual.strip() or None

    st.markdown("---")
    st.markdown("**⚙️ Hyperparameters**")
    w_query      = st.slider("w_query (α)",             0.0, 1.0,  0.5,  0.05,
                             help="1.0 = non-reasoning | <1.0 = IRCoT aktif")
    similarity_t = st.slider("Selection Threshold (τ)", 0.0, 1.0,  0.65, 0.05)
    delta_t      = st.slider("Delta Threshold (δ)",    -0.2, 0.2,  0.01, 0.01)
    top_k        = st.slider("Top-K Final Nodes",       1,   100,  10,   1)

    st.markdown("---")
    st.markdown("**📤 Upload Dokumen TXT**")
    uploaded_file = st.file_uploader("Pilih file .txt", type=["txt"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        st.caption(f"📄 {uploaded_file.name}")
        if st.button("🚀 Proses & Simpan", use_container_width=True):
            with st.spinner("Membangun hierarchical tree..."):
                try:
                    raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

                    existing_ids = get_all_doc_ids("User Dataset", "user_hierarchical_nodes")
                    max_num = 0
                    for did in existing_ids:
                        if did.startswith("DOC_"):
                            try:
                                max_num = max(max_num, int(did.split("_")[1]))
                            except:
                                pass
                    doc_id = f"DOC_{max_num + 1:03d}"

                    tree = build_tree_from_text(
                        doc_text=raw_text,
                        doc_id=doc_id,
                        title=uploaded_file.name.replace(".txt", ""),
                        embedding_model=embedding_model,
                        tokenizer=tokenizer_e5,
                        llm_client=llm_client,
                        # llm_model_name=LLM_MODEL,
                        max_chunk_tokens=100,
                        max_layers=5,
                        min_nodes_to_cluster=3,
                        max_clusters_global=50,
                    )

                    save_tree_to_qdrant(
                        qdrant_client=universal_client,
                        tree_dict=tree,
                        collection_nodes="user_hierarchical_nodes",
                        collection_stats="user_docs_stats",
                    )

                    st.success(f"✅ Tersimpan! Doc ID: `{doc_id}`")
                    st.cache_data.clear()
                    # Set force values untuk redirect ke User Dataset dan doc baru
                    st.session_state.force_dataset = "User Dataset"
                    st.session_state.force_doc = doc_id
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Gagal: {e}")
                    logging.exception("Upload error")

    st.markdown("---")
    vllm_ok = True
    if vllm_ok:
        st.success("vLLM ✅ online", icon="🟢")
    else:
        st.error("vLLM ❌ offline", icon="🔴")
    st.markdown(
        f"<small style='color:#888;font-family:monospace'>"
        f"dataset: {selected_dataset}<br>"
        f"type: {dataset_type}<br>"
        f"collection: {collection_name}</small>",
        unsafe_allow_html=True
    )
    st.markdown("")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result  = None
        st.session_state.show_panel   = False
        st.rerun()


# ── Main Layout ───────────────────────────────────────────────────────────────
has_result = st.session_state.last_result is not None
show_panel = st.session_state.show_panel and has_result

if show_panel:
    chat_col, panel_col = st.columns([3, 1], gap="small")
else:
    chat_col  = st.columns([1])[0]
    panel_col = None

with chat_col:
    h1, h2 = st.columns([5, 1])
    with h1:
        st.markdown("## 🌲 Hierarchical IRCoT RAG")
        if doc_id_filter:
            st.caption(f"📄 Aktif: `{doc_id_filter}` · Dataset: **{selected_dataset}**")
        else:
            st.caption(f"Dataset: **{selected_dataset}** · Type: `{dataset_type}`")
    with h2:
        if has_result:
            btn_label = "◀ Hide" if show_panel else "▶ Stats"
            if st.button(btn_label, use_container_width=True):
                st.session_state.show_panel = not st.session_state.show_panel
                st.rerun()

    st.markdown("---")

    # ── Chat History ──────────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center; height:55vh;">
            <div class="empty-state">
                <div class="icon">🌲</div>
                <div class="title">HIERARCHICAL IRCoT RAG</div>
                <div class="hint">Masukkan query untuk memulai penelusuran hierarkis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-user"><div class="bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="msg-assistant">'
                    f'<div class="avatar">🌲</div>'
                    f'<div class="bubble">{msg["content"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                meta = msg.get("meta", {})
                if meta:
                    t = meta.get("times", {})
                    s = meta.get("stats", {})
                    st.markdown(
                        f'<div class="msg-meta">'
                        f'⏱ {t.get("total_pipeline", 0):.1f}s &nbsp;·&nbsp; '
                        f'🌲 {s.get("roots_processed", 0)} roots &nbsp;·&nbsp; '
                        f'📄 {s.get("final_nodes_count", 0)} nodes &nbsp;·&nbsp; '
                        f'✂️ {s.get("total_pruned", 0)} pruned'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # ── Input Area ────────────────────────────────────────────────────────────
    if not vllm_ok:
        st.error("❌ vLLM server offline.")
    else:
        in_col, btn_col = st.columns([6, 1])
        with in_col:
            user_input = st.text_area(
                "query_input",
                placeholder="Tanyakan sesuatu tentang dokumen...",
                height=75,
                label_visibility="collapsed",
                key=f"query_input_box_{st.session_state.input_key}",
            )
        with btn_col:
            send = st.button("➤", type="primary", key="send_btn")

        # ── FIX: cek Enter key juga via on_change ─────────────────────────
        if send and user_input.strip():
            query = user_input.strip()
            st.session_state.chat_history.append({"role": "user", "content": query})

            with st.spinner("🌲 Traversing hierarchical tree..."):
                try:
                    result = run_full_pipeline(
                        query=query,
                        client=qdrant_client,
                        collection=collection_name,
                        embedding_model=embedding_model,
                        tokenizer_e5=tokenizer_e5,
                        llm_client=llm_client,
                        tokenizer_llm=tokenizer_llm,
                        dataset_type=dataset_type,
                        options=None,
                        w_query=float(w_query),
                        similarity_t=float(similarity_t),
                        delta_t=float(delta_t),
                        limit_root=50,
                        top_k=int(top_k),
                        doc_id_filter=doc_id_filter,
                    )

                    answer   = result.get("answer", "(tidak ada jawaban)")
                    metadata = result.get("metadata", {})

                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": answer,
                        "meta":    metadata,
                    })
                    st.session_state.last_result = result
                    st.session_state.show_panel  = True

                except Exception as e:
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": f"❌ Error: {e}",
                        "meta":    {},
                    })
                    logging.exception("Pipeline error")

            st.session_state.input_key += 1
            st.rerun()


# ── Right Panel: Stats ────────────────────────────────────────────────────────
if panel_col:
    with panel_col:
        result = st.session_state.last_result
        if result is None:
            st.info("Belum ada hasil.")
        else:
            times = result.get("metadata", {}).get("times", {})
            stats = result.get("metadata", {}).get("stats", {})

            st.markdown("#### 📊 Stats")

            def stat_block(label, val):
                st.markdown(
                    f'<div class="stat-block">'
                    f'<div class="s-label">{label}</div>'
                    f'<div class="s-val">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            stat_block("Total Time",      f'{times.get("total_pipeline", 0):.2f}s')
            stat_block("Roots Processed", stats.get("roots_processed", 0))
            stat_block("Nodes Visited",   stats.get("nodes_visited", 0))
            stat_block("Nodes Found",     stats.get("total_nodes_found", 0))
            stat_block("Top-K Returned",  stats.get("final_nodes_count", 0))
            stat_block("Pruned",          stats.get("total_pruned", 0))
            stat_block("Backtracked",     stats.get("total_backtracked", 0))

            st.markdown("---")
            st.markdown("**⏱ Timing Breakdown**")
            for label, key in [
                ("Embed",     "stage1_embedding"),
                ("Search",    "stage2_search"),
                ("Fetch",     "stage3_fetch"),
                ("Traversal", "stage4_traversal"),
                ("Generate",  "stage5_generation"),
            ]:
                c1, c2 = st.columns([2, 1])
                c1.markdown(f"<span style='font-size:0.85rem;color:#555'>{label}</span>", unsafe_allow_html=True)
                c2.markdown(f"`{times.get(key, 0):.2f}s`")

            layer_dist = stats.get("layer_distribution", {})
            if layer_dist:
                st.markdown("---")
                st.markdown("**🗂 Layer Distribution**")
                for layer, count in sorted(layer_dist.items()):
                    c1, c2 = st.columns([2, 1])
                    c1.markdown(f"<span style='font-size:0.85rem;color:#555'>Layer {layer}</span>", unsafe_allow_html=True)
                    c2.markdown(f"`{count}`")

            reasoning_steps = result.get("reasoning_steps", [])
            if reasoning_steps:
                st.markdown("---")
                st.markdown(f"**🧠 IRCoT Steps ({len(reasoning_steps)})**")
                for i, step in enumerate(reasoning_steps, 1):
                    with st.expander(f"Step {i}"):
                        st.markdown(f"<span style='font-size:0.85rem;color:#444'>{step}</span>", unsafe_allow_html=True)

            context_nodes = result.get("context_nodes", [])
            if context_nodes:
                st.markdown("---")
                st.markdown(f"**📄 Top Nodes (top {min(5, len(context_nodes))})**")
                for i, node in enumerate(context_nodes[:5], 1):
                    score = node.get("retrieval_score", 0)
                    layer = node.get("layer", "?")
                    text  = node.get("text", "")[:120]
                    with st.expander(f"#{i} · L{layer} · {score:.3f}"):
                        st.markdown(f"<span style='font-size:0.85rem;color:#444'>{text}…</span>", unsafe_allow_html=True)