import time
import gc
import torch
from retrieval import run_hiro_ircot_pipeline_single
from generation import (
    generate_answer_openended,
    generate_answer_multiple_choice,
    parse_mc_answer,
    format_hiro_context
)


def run_full_pipeline(
    query: str,
    client,
    collection: str,
    embedding_model,
    tokenizer_e5,
    llm_client,
    tokenizer_llm,
    dataset_type: str = "openended",       
    options: str = None,                  
    w_query: float = 0.7,
    similarity_t: float = 0.65,
    delta_t: float = 0.05,
    limit_root: int = 20,
    top_k: int = 10,
    doc_id_filter: str = None
) -> dict:
    """
    Pipeline lengkap: retrieval + generation + pengukuran waktu.
    Mengembalikan dictionary dengan semua informasi untuk transparansi.
    """
    overall_start = time.time()
    result = {}

    # ──────────────── RETRIEVAL ────────────────
    t_ret_start = time.time()
    context_nodes, meta_ret = run_hiro_ircot_pipeline_single(
        client=client,
        query_text=query,
        collection=collection,
        w_query=w_query,
        similarity_t=similarity_t,
        delta_t=delta_t,
        limit_root=limit_root,
        embedding_model=embedding_model,
        tokenizer_e5=tokenizer_e5,
        llm_client=llm_client,
        tokenizer_llm=tokenizer_llm,
        top_k=top_k,
        doc_id_filter=doc_id_filter
    )
    t_ret_end = time.time()
    retrieval_time = t_ret_end - t_ret_start
    print("[PIPE] Before generation, context_nodes:", len(context_nodes), flush=True)

    # ──────────────── GENERATION ────────────────
    t_gen_start = time.time()
    if not context_nodes:
        answer = "Tidak ada konteks yang relevan ditemukan."
        raw_output = answer
        generation_time = 0.0
    else:
        print("[PIPE] First node text:", context_nodes[0].get('text', '')[:100], flush=True)
        if dataset_type == "multiple_choice" and options:
            raw_output = generate_answer_multiple_choice(
                query=query,
                options=options,
                context_nodes=context_nodes,
                llm_client=llm_client,
                tokenizer_llm=tokenizer_llm
            )
            answer = parse_mc_answer(raw_output)
            if answer is None:
                answer = raw_output
        else:
            answer = generate_answer_openended(
                query=query,
                context_nodes=context_nodes,
                llm_client=llm_client,
                tokenizer_llm=tokenizer_llm
            )
            raw_output = answer
    print("[PIPE] Answer received, length:", len(answer), flush=True)
    t_gen_end = time.time()
    generation_time = t_gen_end - t_gen_start

    total_time = time.time() - overall_start

    # ──────────────── Siapkan Data untuk Transparansi ────────────────
    times_ret = meta_ret.get("times", {})
    stats_ret = meta_ret.get("stats", {})
    search_trace = meta_ret.get("search_trace", [])
    
    # Reasoning steps (guidance dari search_trace)
    reasoning_steps = [
        t.get('guidance') for t in search_trace
        if t.get('guidance') is not None
    ]
    
    # Detail node yang diretrieve
    retrieved_texts = [n.get('text', '') for n in context_nodes]
    retrieved_ids = [n.get('d_uuid', '') for n in context_nodes]
    retrieved_scores = [n.get('retrieval_score', 0.0) for n in context_nodes]
    retrieved_layers = [n.get('layer', -1) for n in context_nodes]
    nodes_data = meta_ret.get("nodes_data", []) 

    # ──────────────── Metadata Lengkap ────────────────
    metadata = {
        "times": {
            "stage1_embedding": round(times_ret.get("stage1_embedding", 0), 4),
            "stage2_search": round(times_ret.get("stage2_search", 0), 4),
            "stage3_fetch": round(times_ret.get("stage3_fetch", 0), 4),
            "stage4_traversal": round(times_ret.get("stage4_traversal", 0), 4),
            "stage5_generation": round(generation_time, 4),
            "retrieval_total": round(retrieval_time, 4),
            "generation_total": round(generation_time, 4),
            "total_pipeline": round(total_time, 4),
        },
        "stats": {
            "roots_processed": stats_ret.get("roots_processed", 0),
            "nodes_visited": stats_ret.get("nodes_visited", 0),
            "total_nodes_found": stats_ret.get("total_nodes_found", len(context_nodes)),
            "final_nodes_count": stats_ret.get("final_nodes_count", len(context_nodes)),
            "total_pruned": stats_ret.get("total_pruned", 0),
            "total_backtracked": stats_ret.get("total_backtracked", 0),
            "total_explored": stats_ret.get("total_explored", 0),
            "avg_guidance_len": stats_ret.get("avg_guidance_len", 0),
            "layer_distribution": stats_ret.get("layer_distribution", {}),
        },
        "params": {
            "w_query": w_query,
            "similarity_t": similarity_t,
            "delta_t": delta_t,
            "limit_root": limit_root,
            "top_k": top_k,
            "dataset_type": dataset_type,
        },
        "num_context_nodes": len(context_nodes),
    }

    # ──────────────── OUTPUT FINAL ────────────────
    result = {
        "answer": answer,
        "raw_output": raw_output,
        "context_nodes": context_nodes,
        "formatted_context": format_hiro_context(context_nodes),
        "retrieved_texts": retrieved_texts,
        "retrieved_ids": retrieved_ids,
        "retrieved_scores": retrieved_scores,
        "retrieved_layers": retrieved_layers,
        "reasoning_steps": reasoning_steps,
        "search_trace": search_trace,
        "nodes_data": nodes_data,
        "metadata": metadata,
    }

    # ──────────────── CLEANUP ────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result