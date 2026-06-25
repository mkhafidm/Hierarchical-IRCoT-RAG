# ── Paths ────────────────────────────────────────────────
EMBED_MODEL = "/kaggle/input/models/mkhafid99/multilingual-e5-large-instruct/transformers/default/1"
LLM_MODEL   = "/kaggle/input/models/mkhafid99/qwen2-5-7b-instruct-awq/transformers/default/1"
DB_CONFIG = {
    "NarrativeQA": "/kaggle/working/narrativeqa_qdrant_db_new",
    "Qasper": "/kaggle/working/qasper_qdrant_db_new",
    "Quality": "/kaggle/working/quality_qdrant_db_new",
    "TyDiQA": "/kaggle/working/tydiqa_qdrant_db_new",
}

 
# ── Qdrant Collections ───────────────────────────────────
COLLECTION_NODES = {
    "NarrativeQA": "narrativeqa_hierarchical_nodes",
    "Qasper": "qasper_hierarchical_nodes",
    "Quality": "quality_hierarchical_nodes",
    "TyDiQA": "tydiqa_hierarchical_nodes",
}

COLLECTION_STATS = {
    "NarrativeQA": "narrativeqa_docs_stats",
    "Qasper": "qasper_docs_stats",
    "Quality": "quality_docs_stats",
    "TyDiQA": "tydiqa_docs_stats",
}
 
# ── vLLM Server ──────────────────────────────────────────
VLLM_URL = "http://localhost:8000/v1"