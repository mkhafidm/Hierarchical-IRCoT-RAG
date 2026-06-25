import time
import gc
import logging
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from config import LLM_MODEL
import concurrent.futures


# ---------- Helper functions ----------
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_embedding(query: str, embedding_model, tokenizer) -> list:
    """
    Menghasilkan vektor query menggunakan embedding_model dan tokenizer yang diberikan.
    """
    task_description = """
    Retrieve the most relevant semantic summaries and technical passages
    from documents context to accurately answer the user query
    """
    prepared_query = get_detailed_instruct(task_description, query)

    token_count = len(tokenizer.encode(prepared_query))
    if token_count > 500:
        # print(f"Tracing: Query cukup panjang ({token_count} tokens)")
        logging.info(f"[STAGE EMBEDDING]: Query terlalu panjang: ({token_count}) tokens")
    
    embedding = embedding_model.encode(
        prepared_query, 
        convert_to_tensor=False, 
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embedding.tolist()



def filtering_root_nodes_qdrant_single(client, query_vector, similarity_threshold, collection, limit_root=50, doc_id_filter=None):
    """Ambil root node yang memenuhi threshold similarity."""
    must_conditions = [
        FieldCondition(key="stage", match=MatchValue(value="root"))
    ]
    if doc_id_filter is not None:
        must_conditions.append(
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))
        )
    
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=Filter(must=must_conditions),
        limit=limit_root,
        score_threshold=similarity_threshold,
    )
    roots_relevant = results.points
    logging.info(f"Root Filter: {len(roots_relevant)} roots passed threshold {similarity_threshold}")
    
    for p in roots_relevant:
        p.payload['d_uuid'] = str(p.id)
    return roots_relevant


    
def fetch_tree_details(client, roots_relevan, collection):
    """Ambil seluruh node yang terkait dengan doc_id dari roots_relevan."""
    start_time = time.time()
    doc_ids = list(set([res.payload.get('doc_id') for res in roots_relevan]))
    all_nodes = []
    
    offset = None
    while True:
        response = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchAny(any=doc_ids)
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=True,
            offset=offset
        )
        nodes, next_offset = response
        all_nodes.extend(nodes)
        if next_offset is None:
            break
        offset = next_offset

    tree_storage = {}
    for record in all_nodes:
        payload = record.payload
        payload['d_uuid'] = str(record.id)
        pid = payload.get('doc_id')
        # node_id = record.id
        node_id = str(record.id)
        
        if pid not in tree_storage:
            tree_storage[pid] = {}
        node_data = payload.copy()
        node_data['vector'] = record.vector
        tree_storage[pid][node_id] = node_data

    duration = time.time() - start_time
    logging.info(f"Fetch Success: {len(all_nodes)} nodes loaded in {duration:.2f}s")
    return tree_storage



def calculate_similarity(v1, v2):
    """Cosine similarity (sudah normalized, cukup dot product)."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    if v1.shape != (1024,) or v2.shape != (1024,):
        logging.error(f"Mismatch Dimensi: {v1.shape} vs {v2.shape}")
        return 0.0
    similarity = np.dot(v1, v2)
    return float(np.clip(similarity, -1.0, 1.0))



def get_ircot_guidance(query, cot_history, factual_context_list, current_title, llm_client, tokenizer_llm, tokenizer_e5):
    """Hasilkan guidance step menggunakan LLM (IRCoT)."""
    
    # from vllm import SamplingParams  # pastikan vllm terinstal
    
    # sampling_params = SamplingParams(
    #     temperature=0.0, top_p=1.0, max_tokens=128
    # )
    
    factual_context = ""
    for i, txt in enumerate(factual_context_list):
        factual_context += f"Passage {i+1}:\n{txt}\n\n"
    
    reasoning_history = (
        " ".join(cot_history) if cot_history
        else "No previous reasoning yet."
    )
    
    prompt = f"""Document: {current_title}

Question: {query}

Retrieved passages:
{factual_context}

Previous reasoning steps:
{reasoning_history}

Task:
Continue the reasoning chain toward answering the question.
Write the NEXT single factual sentence that:
- Can be derived or inferred from the passages above
- Moves one step closer to answering the question
- Contains specific entities, mechanisms, or relationships from the passages
- Does NOT repeat previous reasoning steps
- Does NOT answer the question yet
- Is a declarative factual statement, NOT an instruction or search query

One sentence only, maximum 30 words.

Next reasoning step:"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a chain-of-thought reasoning assistant for document QA. "
                "Generate the next factual reasoning step derived from retrieved passages. "
                "Your output must be a single declarative sentence containing "
                "specific entities from the evidence. Never write instructions or search queries."
            )
        },
        {"role": "user", "content": prompt},
    ]
    
    # Token check optional
    full_tokens = tokenizer_llm.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if len(full_tokens) > 3000:
        print(f"🚨 [TOKEN ALERT] {len(full_tokens)} tokens")
    
    # outputs = llm_engine.chat(messages, sampling_params, use_tqdm=False)
    # Gunakan OpenAI client
    # response = llm_client.chat.completions.create(
    #     model=LLM_MODEL,
    #     messages=messages,
    #     temperature=0.0,
    #     max_tokens=128,
    #     top_p=1.0
    # )
    
    # ── Panggil LLM dengan timeout 60 detik ─────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            llm_client.chat.completions.create,
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=128,
            top_p=1.0
        )
        try:
            response = future.result(timeout=60)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"vLLM timeout pada node '{current_title}'")
        except Exception as e:
            logging.error(f"LLM error pada node '{current_title}': {e}")
            raise  # re-raise biar ketangkep di atas
    
    
    # reasoning_result = outputs[0].outputs[0].text.strip()
    reasoning_result = response.choices[0].message.content.strip()
    
    # Truncate untuk E5
    e5_tokens = tokenizer_e5.encode(reasoning_result)
    if len(e5_tokens) > 450:
        reasoning_result = tokenizer_e5.decode(
            e5_tokens[:450], skip_special_tokens=True
        )
    return reasoning_result



def recursive_hiro_ircot(node_uuid, current_score, query_vector, query_text,
                         paper_tree, w_query, similarity_t, delta_t, doc_id,
                         llm_client, tokenizer_llm, embedding_model, tokenizer_e5,
                         cot_list=None, path_uuids=None, path_texts=None,
                         global_trace=None, visited=None):
    """
    Traversal rekursif HIRO + IRCoT.
    Sekarang semua model/engine diterima sebagai parameter.
    """

    if visited is None:
        visited = set()
    node_uuid_str = str(node_uuid)
    if node_uuid_str in visited:
        logging.warning(f"Cycle detected: node {node_uuid_str} already visited, skip.")
        return []
    visited.add(node_uuid_str)
    
    node = paper_tree.get(str(node_uuid))
    if node is None:
        return []
    # print(f"[TIMESTAMP] {time.strftime('%H:%M:%S')} - ENTER node {node.get('node_index')} layer {node.get('layer')}")
    logging.info(f"[TIMESTAMP] {time.strftime('%H:%M:%S')} - ENTER node {node.get('node_index')} layer {node.get('layer')}")

    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    node_idx        = node.get('node_index')
    node_layer      = node.get('layer')
    indent          = "   " * (4 - node_layer) if node_layer <= 4 else ""
    children_uuids  = node.get('children_uuids', [])

    if cot_list     is None: cot_list     = []
    if path_uuids   is None: path_uuids   = []
    if path_texts   is None: path_texts   = []
    if global_trace is None: global_trace = []

    current_node_text  = node.get('text', '')
    updated_path_texts = path_texts + [current_node_text]
    updated_path_uuids = path_uuids + [node_uuid]
    current_title      = node.get('title', "Unknown Title")

    # logging.info(f"{indent}>>> Node {node_idx} (Layer {node_layer})")
    logging.info(f"{indent}>>> Node {node_idx} (Layer {node_layer}) | Score={current_score:.4f} | Children={len(children_uuids)}")

    # n_children = len(children_uuids)
    # print(f"\n{indent}{'─'*40}")
    # print(f"{indent}📍 Kunjungi Node {node_idx} "
    #       f"(Layer {node_layer}) | "
    #       f"Score={current_score:.4f} | "
    #       f"Jumlah child: {n_children}")
    # print(f"{indent}   Teks (50 char): \"{current_node_text[:50]}...\"")

    # Leaf node
    if not children_uuids:
        # logging.info(f"{indent}🟢 LEAF reached.")
        # print(f"{indent}🟢 LEAF NODE — tidak ada child.")
        # print(f"{indent}{GREEN}{BOLD}✅ [COLLECT LEAF] "
        #       f"Node {node_idx} masuk context.{RESET}")
        logging.info(f"{indent}🟢 LEAF reached. Node {node_idx} collected.")
        node_copy = {**node, 'retrieval_score': current_score}
        return [node_copy]

    # Reasoning
    use_reasoning = (w_query < 1.0)
    if use_reasoning:
        # print(f"[TIMESTAMP] {time.strftime('%H:%M:%S')} - REASONING START node {node_idx}")
        logging.info(f"{indent}🧠 Reasoning triggered for node {node_idx}")
        current_guidance = get_ircot_guidance(
            query=query_text,
            cot_history=cot_list,
            factual_context_list=updated_path_texts,
            current_title=current_title,
            llm_client=llm_client,
            tokenizer_llm=tokenizer_llm,
            tokenizer_e5=tokenizer_e5
        )
        # guidance_vector  = get_embedding(current_guidance, embedding_model, tokenizer_e5)
        # new_cot_history  = cot_list + [current_guidance]
        logging.info(f"{indent}[DEBUG] LLM done node {node_idx}, now embedding...")  # ← tambah
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_embedding, current_guidance, embedding_model, tokenizer_e5)
            try:
                guidance_vector = future.result(timeout=30)
                logging.info(f"{indent}[DEBUG] Embedding done node {node_idx}")  # ← tambah
            except concurrent.futures.TimeoutError:
                logging.warning(f"{indent}[DEBUG] Embedding TIMEOUT node {node_idx}, fallback ke query_vector")
                guidance_vector = query_vector
        
        new_cot_history = cot_list + [current_guidance]

        global_trace.append({
            "doc_id"     : doc_id,
            "node_index" : node_idx,
            "layer"      : node_layer,
            "step_score" : current_score,
            "use_reasoning"   : True,
            "guidance"   : current_guidance,
            "guidance_len"    : len(current_guidance.split()),
        })
        # print(f"\n{indent}🧠 [REASONING] Layer {node_layer} | Node {node_idx}")
        # print(f"{indent}   Guidance: \"{current_guidance[:100]}...\"")
        # print(f"[TIMESTAMP] {time.strftime('%H:%M:%S')} - REASONING END node {node_idx}")
        logging.info(f"{indent}Reasoning: {current_guidance[:100]}...")
    else:
        current_guidance = ""
        guidance_vector  = query_vector
        new_cot_history  = cot_list
        global_trace.append({
            "doc_id"     : doc_id,
            "node_index" : node_idx,
            "layer"      : node_layer,
            "step_score" : current_score,
            "use_reasoning"   : False,
            "guidance"   : None,
            "guidance_len"    : 0,
        })
        logging.info(f"{indent}No reasoning (w_query=1.0)")

    # Evaluasi child
    collected_context = []
    parent_collected  = False
    current_trace_idx = len(global_trace) - 1

    pruned_nodes     = []
    backtracked_nodes = []
    explored_nodes   = []

    # print(f"\n{indent}🔍 Evaluasi {len(children_uuids)} child node:")
    logging.info(f"{indent}🔍 Evaluating {len(children_uuids)} children...")

    for c_uuid in children_uuids:
        logging.info(f"{indent}  [DEBUG] Processing c_uuid={c_uuid}")
        child_node = paper_tree.get(c_uuid)
        if child_node is None: continue

        child_vector = child_node.get('vector')
        if child_vector is None:
            # logging.warning(f"child {c_uuid} has no vector, skip")
            logging.warning(f"{indent}  Child {c_uuid} has no vector, skip.")
            continue

        logging.info(f"{indent}  [DEBUG] Got vector, computing similarity...")
        c_idx          = child_node.get('node_index')
        c_layer        = child_node.get('layer')
        # c_score        = calculate_similarity(query_vector, child_node['vector'])
        # reason_score   = calculate_similarity(guidance_vector, child_node['vector'])
        
        c_score = calculate_similarity(query_vector, child_vector)
        try:
            reason_score = calculate_similarity(guidance_vector, child_vector)
        except Exception as e:
            logging.error(f"reason_score failed: {e}", exc_info=True)
            raise
        # reason_score = calculate_similarity(guidance_vector, child_vector)
        # logging.info(f"{indent}  [DEBUG] reason_score done: {reason_score:.4f}")
        # combined_score = (w_query * c_score) + ((1 - w_query) * reason_score)
        # c_delta        = combined_score - current_score

        combined_score = (w_query * c_score) + ((1 - w_query) * reason_score)
        c_delta = combined_score - current_score

        logging.info(f"{indent}  Child {c_idx}: score={combined_score:.4f} delta={c_delta:.4f}")

        if combined_score < similarity_t:
            pruned_nodes.append({"node_index": c_idx, "layer": c_layer,
                                 "combined_score": combined_score, "delta": None})
            # print(f"{indent}   🔴 PRUNED  Node {c_idx}: "
            #       f"score={combined_score:.4f} < τ={similarity_t} → dibuang")
            logging.info(f"{indent}   🔴 PRUNED child {c_idx}: score={combined_score:.4f} < τ={similarity_t}")
            continue

        elif c_delta < delta_t:
            backtracked_nodes.append({"node_index": c_idx, "layer": c_layer,
                                      "combined_score": combined_score, "delta": c_delta})
            # print(f"{indent}   🟡 BACKTRACK Node {c_idx}: "
            #       f"Δ={c_delta:.4f} < δ={delta_t} → tidak cukup informatif")
            logging.info(f"{indent}   🟡 BACKTRACK child {c_idx}: Δ={c_delta:.4f} < δ={delta_t}, collect parent {node_idx}")
            if not parent_collected:
                # print(f"{indent}   {GREEN}✅ [COLLECT PARENT] "
                #       f"Node {node_idx} (Layer {node_layer}) dikumpulkan sebagai konteks.{RESET}")
                node_copy = {**node, 'retrieval_score': current_score}
                collected_context.append(node_copy)
                parent_collected = True
            continue

        else:
            explored_nodes.append({"node_index": c_idx, "layer": c_layer,
                                   "combined_score": combined_score, "delta": c_delta})
            # print(f"{indent}   🔵 EXPLORE Node {c_idx} (Layer {c_layer}): "
            #       f"score={combined_score:.4f}, Δ={c_delta:.4f} → masuk lebih dalam")
            logging.info(f"{indent}   🔵 EXPLORE child {c_idx}: score={combined_score:.4f}, Δ={c_delta:.4f}")
            res = recursive_hiro_ircot(
                node_uuid=c_uuid,
                current_score=combined_score,
                query_vector=query_vector,
                query_text=query_text,
                paper_tree=paper_tree,
                w_query=w_query,
                similarity_t=similarity_t,
                delta_t=delta_t,
                doc_id=doc_id,
                llm_client=llm_client,
                tokenizer_llm=tokenizer_llm,
                embedding_model=embedding_model,
                tokenizer_e5=tokenizer_e5,
                cot_list=new_cot_history,
                path_uuids=updated_path_uuids,
                path_texts=updated_path_texts,
                global_trace=global_trace,
                visited=visited
            )
            collected_context.extend(res)
            # print(f"{indent}   ⬆️  [Backtrack] ke parent  {node_idx} (Layer {node_layer})")
            # logging.info(f"{indent}   ⬆️  [Backtrack] ke parent {node_idx}")
            # print(f"[TIMESTAMP] {time.strftime('%H:%M:%S')} - CHILD {c_idx} AFTER recursion")

    # Update trace
    global_trace[current_trace_idx].update({
        "pruned_nodes": pruned_nodes,
        "backtracked_nodes": backtracked_nodes,
        "explored_nodes": explored_nodes,
        "n_pruned": len(pruned_nodes),
        "n_backtracked": len(backtracked_nodes),
        "n_explored": len(explored_nodes),
    })

    unique_results = list({n['d_uuid']: n for n in collected_context}.values())
    # print(f"[TIMESTAMP] {time.strftime('%H:%M:%S')} - EXIT node {node_idx}")
    logging.info(f"{indent}<<< EXIT node {node_idx}, collected {len(unique_results)} nodes")
    return unique_results



# ---------- Main pipeline ----------
def run_hiro_ircot_pipeline_single(
    client: QdrantClient,
    query_text: str,
    collection: str,
    w_query: float,
    similarity_t: float,
    delta_t: float,
    limit_root: int,
    embedding_model,
    tokenizer_e5,              # tokenizer untuk embedding
    llm_client,                # vLLM engine
    tokenizer_llm,             # tokenizer untuk LLM
    top_k: int = 50,
    doc_id_filter=None
):
    """
    Pipeline utama HIRO-IRCoT.
    Semua model diterima sebagai argumen (bukan global).
    Mengembalikan tuple: (final_top_k_nodes, raw_metadata)
    """
    logging.info(f"🚀 [PIPELINE START] Query: '{query_text}'")
    overall_start = time.time()

    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Stage 1: Embedding
    t1_start = time.time()
    query_vector = get_embedding(query_text, embedding_model, tokenizer_e5)
    t1_end = time.time() - t1_start
    logging.info(f"TIMER: Stage 1 (Embedding) took {t1_end:.4f}s")

    # Stage 2: Root filtering
    t2_start = time.time()
    filtered_roots_relevant = filtering_root_nodes_qdrant_single(
        client=client,
        query_vector=query_vector,
        similarity_threshold=similarity_t,
        collection=collection,
        limit_root=limit_root,
        doc_id_filter=doc_id_filter
    )
    t2_end = time.time() - t2_start
    logging.info(f"TIMER: Stage 2 (Root Filter) took {t2_end:.4f}s - Top {len(filtered_roots_relevant)} roots selected")

    if not filtered_roots_relevant:
        logging.warning(f"RESULT: No roots passed threshold {similarity_t}")
        return [], {"times": {"total_overall": time.time() - overall_start}}

    # Print query & root filtering (optional untuk debug)
    # print(f"\n{'='*60}")
    # print(f"📝 QUERY: {query_text}")
    # print(f"{'='*60}")
    # print(f"\n📌 STAGE 2: Root Filtering")
    # print(f"   {len(filtered_roots_relevant)} root node lolos threshold τ={similarity_t}:")
    logging.info(f"📝 QUERY: {query_text}")
    logging.info(f"📌 Root Filtering: {len(filtered_roots_relevant)} roots passed threshold {similarity_t}")
    for i, r in enumerate(filtered_roots_relevant):
        logging.info(f"   [{i+1}] doc_id={r.payload.get('doc_id')} | title={r.payload.get('title','?')[:50]} | score={r.score:.4f}")
    logging.info(f"🌳 Tree traversal started")
        # print(f"   [{i+1}] doc_id={r.payload.get('doc_id')} | "
        #       f"title={r.payload.get('title','?')[:50]} | "
        #       f"score={r.score:.4f}")
    # print(f"\n{'='*60}")
    # print(f"🌳 STAGE 4: Tree Traversal Dimulai")
    # print(f"{'='*60}")

    # Stage 3: Fetch tree
    t3_start = time.time()
    tree_detail_storage = fetch_tree_details(client, filtered_roots_relevant, collection)
    t3_end = time.time() - t3_start
    logging.info(f"TIMER: Stage 3 (Fetch Data) took {t3_end:.4f}s")

    # Stage 4: Traversal
    t4_start = time.time()
    all_final_contexts = []
    global_trace_list = []

    for i_root, root_point in enumerate(filtered_roots_relevant):
        pid = root_point.payload.get('doc_id')
        root_uuid = root_point.id
        root_score = root_point.score
        paper_tree = tree_detail_storage.get(pid)
        if not paper_tree:
            continue

        # print(f"\n{'─'*60}")
        # print(f"🌲 ROOT [{i_root+1}/{len(filtered_roots_relevant)}]: doc_id={pid}")
        # print(f"   Score : {root_point.score:.4f}")
        # print(f"{'─'*60}")

        logging.info(f"🌲 ROOT [{i_root+1}/{len(filtered_roots_relevant)}]: doc_id={pid}, score={root_score:.4f}")

        initial_cot = []
        initial_path_uuids = []

        try:
            paper_context = recursive_hiro_ircot(
                node_uuid=root_uuid,
                current_score=root_score,
                query_vector=query_vector,
                query_text=query_text,
                paper_tree=paper_tree,
                w_query=w_query,
                similarity_t=similarity_t,
                delta_t=delta_t,
                doc_id=pid,
                llm_client=llm_client,
                tokenizer_llm=tokenizer_llm,
                embedding_model=embedding_model,
                tokenizer_e5=tokenizer_e5,
                cot_list=initial_cot,
                path_uuids=initial_path_uuids,
                global_trace=global_trace_list
            )
            all_final_contexts.extend(paper_context)
        except Exception as e:
            logging.error(f"Traversal gagal untuk root {pid}: {e}")
            # print(f"❌ ERROR pada root {pid}: {e}")
            continue  # atau raise jika ingin berhenti total
    
    final_unique = list({n['d_uuid']: n for n in all_final_contexts}.values())
    sorted_contexts = sorted(final_unique, key=lambda x: x.get('retrieval_score', 0), reverse=True)
    final_top_k_nodes = sorted_contexts[:top_k]
    t4_end = time.time() - t4_start
    logging.info(f"TIMER: Stage 4 (Traversal) took {t4_end:.4f}s")

    overall_duration = time.time() - overall_start
    logging.info(f"🏁 TOTAL PIPELINE TIME: {overall_duration:.2f}s")

    
    # Statistics
    total_pruned = sum(t.get('n_pruned', 0) for t in global_trace_list)
    total_backtracked = sum(t.get('n_backtracked', 0) for t in global_trace_list)
    total_explored = sum(t.get('n_explored', 0) for t in global_trace_list)
    avg_guidance_len = (
        sum(t.get('guidance_len', 0) for t in global_trace_list) /
        len(global_trace_list) if global_trace_list else 0
    )
    layer_dist = {}
    for n in final_top_k_nodes:
        l = n.get('layer', 'unknown')
        layer_dist[l] = layer_dist.get(l, 0) + 1

    raw_metadata = {
        "query": query_text,
        "search_trace": global_trace_list,
        "retrieved_texts": [n['text'] for n in final_top_k_nodes],
        "times": {
            "stage1_embedding": t1_end,
            "stage2_search": t2_end,
            "stage3_fetch": t3_end,
            "stage4_traversal": t4_end,
            "total_overall": overall_duration
        },
        "total_nodes": len(final_unique),
        "nodes_data": [
            {
                "d_uuid": n['d_uuid'],
                "layer": n.get('layer'),
                "doc_id": n.get('doc_id'),
                "node_index": n.get('node_index')
            } for n in final_top_k_nodes
        ],
        "stats": {
            "roots_processed": len(filtered_roots_relevant),
            "nodes_visited": len(global_trace_list),
            "total_nodes_found": len(final_unique),
            "final_nodes_count": len(final_top_k_nodes),
            "total_pruned": total_pruned,
            "total_backtracked": total_backtracked,
            "total_explored": total_explored,
            "avg_guidance_len": round(avg_guidance_len, 2),
            "layer_distribution": layer_dist,
        }
    }

    logging.info(f"✅ TRAVERSAL SELESAI")
    logging.info(f"   Total node terkumpul (pre top-k): {len(final_unique)}")
    logging.info(f"   Node dikembalikan (top-k={top_k}): {len(final_top_k_nodes)}")
    logging.info(f"   Distribusi layer: {layer_dist}")

    # print(f"\n{'='*60}")
    # print(f"✅ TRAVERSAL SELESAI")
    # print(f"   Total node terkumpul (pre top-k) : {len(final_unique)}")
    # print(f"   Node dikembalikan (top-k={top_k})  : {len(final_top_k_nodes)}")
    # print(f"   Distribusi layer hasil akhir:")
    # for layer, count in sorted(layer_dist.items()):
    #     print(f"     Layer {layer}: {count} node")
    # print(f"{'='*60}\n")

    # Cleanup (opsional)
    for node in final_top_k_nodes:
        node.pop('vector', None)
            
    del tree_detail_storage, all_final_contexts, final_unique, sorted_contexts
    gc.collect()

    # Jika Anda menggunakan torch dan cuda, bisa diaktifkan di GUI nanti
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    return final_top_k_nodes, raw_metadata