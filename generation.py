from openai import OpenAI
from config import LLM_MODEL

def format_hiro_context(nodes):
    """
    Mengubah list node hasil HIRO retrieval menjadi string terstruktur
    untuk prompt LLM.
    """
    if not nodes:
        return "No relevant context found."
    
    all_layers = [n.get('layer', 0) for n in nodes]
    max_l = max(all_layers) if all_layers else 0
    min_l = min(all_layers) if all_layers else 0
    
    # Urutkan dari layer terbesar ke terkecil (summary → detail)
    sorted_nodes = sorted(
        nodes,
        key=lambda x: (-x.get('layer', 0), x.get('node_index', 0))
    )
    
    context_str = ""
    current_layer = -1
    
    for i, node in enumerate(sorted_nodes, 1):
        layer    = node.get('layer', 0)
        doc_id   = node.get('doc_id', 'Unknown')
        title    = node.get('title', 'Unknown')
        text     = node.get('text', '').strip()
        
        # Layer header – hanya tampil saat ganti layer
        if layer != current_layer:
            if layer == min_l and layer == max_l:
                layer_label = f"CONTEXT (LAYER {layer})"
            elif layer == min_l:
                layer_label = f"DETAILED PASSAGES (LAYER {layer})"
            elif layer == max_l:
                layer_label = f"HIGH-LEVEL SUMMARY (LAYER {layer})"
            else:
                layer_label = f"INTERMEDIATE SUMMARY (LAYER {layer})"
            
            separator = "\n" if current_layer != -1 else ""
            context_str += f"{separator}{'='*50}\n"
            context_str += f"[{layer_label}]\n"
            context_str += f"{'='*50}\n"
            current_layer = layer
        
        # Node content
        context_str += f"[Passage {i} | Paper: {doc_id} | {title}]\n"
        context_str += f"{text}\n"
        context_str += f"{'-'*30}\n"
    
    return context_str.strip()


def generate_answer_multiple_choice(
    query: str,
    options: str,          
    context_nodes: list,
    llm_client,
    tokenizer_llm,
    max_prompt_limit: int = 5500
) -> str:
    
    context_combined = format_hiro_context(context_nodes)
    
    
    TEMPLATE = (
        "Context:\n"
        "[CONTEXT_AREA]\n\n"
        "Question: {query}\n\n"
        "Options:\n"
        "{options}\n\n"
        "Instructions:\n"
        "1. Use ONLY the provided context to select the correct option.\n"
        "2. Eliminate options contradicted by the context.\n"
        "3. Prefer specific, detailed evidence over general summaries.\n"
        "4. Do NOT use outside knowledge.\n\n"
        "Respond in this exact format:\n"
        "Evidence: [most relevant sentence from context]\n"
        "Reasoning: [why this option is correct]\n"
        "Answer: [A, B, C, or D]\n\n"
        "Answer:"
    )
    
    static_prompt = TEMPLATE.replace("[CONTEXT_AREA]", "").format(
        query=query, options=options
    )
    static_tokens = len(tokenizer_llm.encode(static_prompt))
    allowed_context_tokens = max_prompt_limit - static_tokens
    
    context_tokens = tokenizer_llm.encode(context_combined)
    if len(context_tokens) > allowed_context_tokens:
        truncated = context_tokens[:allowed_context_tokens - 20]
        context_combined = (
            tokenizer_llm.decode(truncated, skip_special_tokens=True)
            + "\n[CONTEXT TRUNCATED]"
        )
    
    final_prompt = TEMPLATE.replace("[CONTEXT_AREA]", context_combined)
    final_prompt = final_prompt.format(query=query, options=options)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise multiple-choice question answering assistant. "
                "Select the best answer based strictly on the provided context. "
                "Always respond with Evidence, Reasoning, and Answer fields."
            )
        },
        {"role": "user", "content": final_prompt},
    ]

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        top_p=1.0
    )
    return response.choices[0].message.content.strip()

def generate_answer_openended(query, context_nodes, llm_client, tokenizer_llm, max_prompt_limit=5500):
    print("[GEN] Starting...")
    if not context_nodes:
        print("[GEN] No context nodes")
        return "No context nodes."
    if tokenizer_llm is None:
        print("[GEN] ERROR: tokenizer_llm is None")
        return "Tokenizer missing."
    try:
        context_combined = format_hiro_context(context_nodes)
        print(f"[GEN] Context length: {len(context_combined)} chars")
        
        TEMPLATE = (
            "Context:\n"
            "[CONTEXT_AREA]\n\n"
            "Question: [QUERY]\n\n"
            "Instructions:\n"
            "1. Answer using ONLY the provided context.\n"
            "2. Be concise and direct — 1-2 sentences maximum.\n"
            "3. If the answer is a specific name, number, or term, state it explicitly.\n"
            "4. If the context is insufficient, briefly state what the context does discuss.\n"
            "5. Do NOT use phrases like 'Based on the context' or 'According to'.\n\n"
            "Answer:"
        )
        
        static_prompt = TEMPLATE.replace("[CONTEXT_AREA]", "").replace("[QUERY]", query)
        static_tokens = len(tokenizer_llm.encode(static_prompt))
        print(f"[GEN] Static tokens: {static_tokens}")
        
        allowed_context_tokens = max_prompt_limit - static_tokens
        if allowed_context_tokens <= 0:
            print(f"[GEN] WARNING: allowed_context_tokens = {allowed_context_tokens}, force to 500")
            allowed_context_tokens = 500
        
        context_tokens = tokenizer_llm.encode(context_combined)
        print(f"[GEN] Context tokens: {len(context_tokens)}")
        if len(context_tokens) > allowed_context_tokens:
            truncated = context_tokens[:allowed_context_tokens - 20]
            context_combined = tokenizer_llm.decode(truncated, skip_special_tokens=True) + "\n[CONTEXT TRUNCATED]"
            print(f"[GEN] Truncated to {len(truncated)} tokens")
        
        final_prompt = TEMPLATE.replace("[CONTEXT_AREA]", context_combined).replace("[QUERY]", query)
        # messages = [...]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise question answering assistant. "
                    "Answer questions based strictly on the provided context. "
                    "Never hallucinate or use outside knowledge."
                )
            },
            {"role": "user", "content": final_prompt},
        ]
        
        print("[GEN] Sending request to LLM...")
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            top_p=1.0,
            timeout=60
        )
        print("[GEN] Response received")
        answer = response.choices[0].message.content.strip()
        print(f"[GEN] Answer: '{answer[:100]}'")
        return answer if answer else "(empty response)"
    except Exception as e:
        print(f"[GEN] EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        return f"Generation error: {e}"


def parse_mc_answer(raw_output):
    import re
    match = re.search(r'Answer:\s*([A-D])', raw_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.findall(r'\b([A-D])\b', raw_output)
    if match:
        return match[-1].upper()
    return None