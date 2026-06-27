import re
import uuid
import logging
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any, Set
import numpy as np
from sklearn.mixture import GaussianMixture
import umap
from qdrant_client.http.models import PointStruct
from config import LLM_MODEL

logger = logging.getLogger(__name__)

class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(
        self,
        index: int,
        title: str,
        text: str,                          
        embeddings: Optional[np.ndarray],    
        children: Optional[Set[int]] = None,
        parents: Optional[Set[int]] = None,
        layer: int = 0,
        cluster_id: Optional[str] = None,  
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.title = title
        self.embeddings = embeddings
        self.children = children if children is not None else set()
        self.parents = parents if parents is not None else set()
        self.layer = layer
        self.cluster_id = cluster_id
        self.metadata = metadata


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes, metadata, processing_time: float = 0.0
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes
        self.metadata = metadata
        self.processing_time = processing_time


# -------------------------------------------------------------------
# 1. PREPROCESSING & CHUNKING
# -------------------------------------------------------------------
def preprocess_general_txt(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()

def chunking(text: str, tokenizer, max_tokens: int = 100) -> List[str]:
    """Sentence‑preserving chunking with max_tokens limit."""
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_ids = tokenizer.encode(sent, add_special_tokens=False, truncation=False)
        sent_len = len(sent_ids)
        if sent_len > max_tokens:
            # split by punctuation inside
            subs = re.split(r"[,:;]", sent)
            for sub in subs:
                sub = sub.strip()
                if not sub:
                    continue
                sub_ids = tokenizer.encode(sub, add_special_tokens=False)
                if len(sub_ids) > max_tokens:
                    # forced token split
                    for i in range(0, len(sub_ids), max_tokens):
                        piece = tokenizer.decode(sub_ids[i:i+max_tokens])
                        chunks.append(piece)
                    current_chunk = []
                    current_len = 0
                else:
                    if current_len + len(sub_ids) > max_tokens:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    current_chunk.append(sub)
                    current_len += len(sub_ids)
            continue
        if current_len + sent_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent)
        current_len += sent_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# -------------------------------------------------------------------
# 2. EMBEDDING
# -------------------------------------------------------------------
def embedding(texts: List[str], model, batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """Return embeddings as float32 numpy array."""
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False
    ).astype(np.float32)

# -------------------------------------------------------------------
# 3. DIMENSIONALITY REDUCTION (UMAP)
# -------------------------------------------------------------------
def reduce_dimension_umap(embeddings: np.ndarray, n_neighbors: int, n_components: int = 10,
                          metric: str = "cosine", min_dist: float = 0.0, random_state: int = 42) -> np.ndarray:
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, metric=metric,
                        min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(embeddings).astype(np.float32)

# -------------------------------------------------------------------
# 4. BIC OPTIMAL CLUSTERS
# -------------------------------------------------------------------
def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50,
                         min_clusters: int = 2, random_state: int = 42) -> Tuple[int, Dict[int, float]]:
    max_clusters = min(max_clusters, len(embeddings))
    min_clusters = min(min_clusters, max_clusters)
    best_k = min_clusters
    best_bic = float("inf")
    bic_scores = {}
    for k in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=random_state, reg_covar=1e-6)
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        bic_scores[k] = bic
        if bic < best_bic:
            best_bic = bic
            best_k = k
    return best_k, bic_scores

# -------------------------------------------------------------------
# 5. GMM SOFT CLUSTERING
# -------------------------------------------------------------------
def GMM_cluster(embeddings: np.ndarray, n_clusters: int, threshold: float = 0.1,
                random_state: int = 42) -> Tuple[List[np.ndarray], int]:
    n_clusters = min(n_clusters, len(embeddings))
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state, reg_covar=1e-6)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    hard_labels = probs.argmax(axis=1)
    labels = []
    for i, prob in enumerate(probs):
        hits = np.where(prob > threshold)[0]
        if hits.size == 0:
            hits = np.array([hard_labels[i]], dtype=int)
        labels.append(hits)
    return labels, n_clusters

# -------------------------------------------------------------------
# 6. MAIN CLUSTERING PIPELINE
# -------------------------------------------------------------------
def compute_global_dimension(dim: int, N: int) -> int:
    return max(2, min(dim, N // 5, N - 2))

def main_clustering(embeddings: np.ndarray, dim: int = 20, threshold: float = 0.1,
                    max_clusters: int = 50, random_state: int = 42,
                    global_n_neighbors: Optional[int] = None) -> Tuple[List[List[int]], List[np.ndarray], List[List[int]]]:
    """
    Returns:
        all_assignments: list of list of int (global cluster IDs for each point)
        global_assignments: list of arrays (soft global)
        local_assignments: list of list (final local cluster IDs)
    """
    N = len(embeddings)
    if N < 10:
        logger.info("Data terlalu kecil, assign single cluster.")
        all_assign = [[0] for _ in range(N)]
        global_assign = [np.array([0]) for _ in range(N)]
        local_assign = [[0] for _ in range(N)]
        return all_assign, global_assign, local_assign

    all_assignments = [[] for _ in range(N)]
    total_clusters = 0
    all_local_assignments = [[] for _ in range(N)]

    # GLOBAL
    if global_n_neighbors is None:
        global_n_neighbors = max(5, int(N ** 0.5))
    global_n_neighbors = min(global_n_neighbors, N-1)
    global_dim = compute_global_dimension(dim, N)
    reduced_global = reduce_dimension_umap(embeddings, n_components=global_dim,
                                           n_neighbors=global_n_neighbors, random_state=random_state)
    safe_max_k = min(max_clusters, max(2, int(N ** 0.5)))
    best_k_global, _ = get_optimal_clusters(reduced_global, max_clusters=safe_max_k, min_clusters=2, random_state=random_state)
    logger.info(f"Global optimal clusters: {best_k_global}")
    global_assignments, n_global_clusters = GMM_cluster(reduced_global, best_k_global, threshold, random_state)

    # LOCAL per global cluster
    for g_id in range(n_global_clusters):
        global_indices = np.array([g_id in clusters for clusters in global_assignments])
        if not global_indices.any():
            continue
        local_embeddings = embeddings[global_indices]
        local_indices = np.where(global_indices)[0]
        N_local = len(local_embeddings)

        if N_local < 10:
            local_assign = [np.array([0])] * N_local
            n_local = 1
        else:
            n_neighbors = max(5, int(N_local ** 0.5))
            n_neighbors = min(n_neighbors, N_local-1)
            local_dim = max(2, min(global_dim, N_local // 5, N_local-2))
            reduced_local = reduce_dimension_umap(local_embeddings, n_components=local_dim,
                                                  n_neighbors=n_neighbors, random_state=random_state)
            safe_max_local = min(max_clusters, max(2, int(N_local ** 0.5)))
            best_k_local, _ = get_optimal_clusters(reduced_local, max_clusters=safe_max_local, min_clusters=2)
            local_assign, n_local = GMM_cluster(reduced_local, best_k_local, threshold, random_state)

        # Map to global IDs
        for local_cid in range(n_local):
            mask = [local_cid in cl for cl in local_assign]
            for idx in local_indices[np.array(mask)]:
                all_assignments[idx].append(local_cid + total_clusters)
                all_local_assignments[idx].append(local_cid + total_clusters)
        total_clusters += n_local

    logger.info(f"Total unique clusters: {total_clusters}")
    return all_assignments, global_assignments, all_local_assignments


# -------------------------------------------------------------------
# 7. SUMMARIZATION BATCH (vLLM)
# -------------------------------------------------------------------

def summarize_cluster(all_cluster_texts, llm_client, max_tokens=512):
    """Ringkas setiap cluster menggunakan OpenAI client."""
    if not all_cluster_texts:
        return []
    all_summaries = []
    for texts in all_cluster_texts:
        joined = "\n".join(f"- {t.strip()}" for t in texts if t.strip())
        user_prompt = (
            "Summarize the text below factually and concisely.\n\n"
            "RULES:\n"
            "1. Keep entity roles consistent.\n"
            "2. Preserve all names, types, and relationships.\n"
            "3. Do NOT add events or outcomes not in source.\n"
            "4. Output ONLY the summary, no meta-commentary.\n"
            "5. Max 150 words.\n\n"
            f"TEXT:\n{joined}\n\nSUMMARY:"
        )
        messages = [{"role": "user", "content": user_prompt}]
        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                top_p=1.0,
                timeout=60
            )
            summary = response.choices[0].message.content.strip()
            all_summaries.append(summary)
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            all_summaries.append("") 
    return all_summaries



# -------------------------------------------------------------------
# 8. MAIN TREE BUILDER
# -------------------------------------------------------------------
def build_tree_from_text(
    doc_text: str,
    doc_id: str,
    title: str,
    embedding_model,
    tokenizer,            
    llm_client,         
    max_chunk_tokens: int = 100,
    max_layers: int = 5,
    min_nodes_to_cluster: int = 3,
    max_clusters_global: int = 50,
) -> Tree:
    """
    Build hierarchical tree from a document text.
    """
    logger.info(f"Building tree for doc_id={doc_id}, title={title}")
    start_time = time.time()

    # Preprocess & chunk
    cleaned = preprocess_general_txt(doc_text)
    chunks = chunking(cleaned, tokenizer, max_tokens=max_chunk_tokens)
    if not chunks:
        raise ValueError("No chunks generated from document")
    logger.info(f"Generated {len(chunks)} chunks")

    # Embed leaf nodes
    leaf_embeddings = embedding(chunks, embedding_model, batch_size=32)

    # Build leaf nodes (layer 0)
    all_nodes = {}
    leaf_nodes = {}
    layer_to_nodes = {0: []}
    next_idx = 0
    for text, emb in zip(chunks, leaf_embeddings):
        node = Node(index=next_idx, title=title, text=text, embeddings=emb, layer=0,
                    metadata={"stage": "leaf", "n_children": 0})
        all_nodes[next_idx] = node
        leaf_nodes[next_idx] = node
        layer_to_nodes[0].append(next_idx)
        next_idx += 1

    current_nodes = dict(leaf_nodes)  

    # Build higher layers
    for layer in range(1, max_layers+1):
        n_current = len(current_nodes)
        if n_current <= min_nodes_to_cluster:
            logger.info(f"Stopping at layer {layer-1} because nodes <= {min_nodes_to_cluster}")
            break

        # Prepare embeddings for current layer
        node_ids = list(current_nodes.keys())
        node_list = [current_nodes[i] for i in node_ids]
        if not node_list:  
            break
        layer_embs = np.vstack([n.embeddings for n in node_list])

        # Clustering
        memberships, _, _ = main_clustering(layer_embs, max_clusters=max_clusters_global)
        cluster_to_positions = defaultdict(list)
        for pos, cluster_ids in enumerate(memberships):
            for cid in cluster_ids:
                cluster_to_positions[cid].append(pos)

        # Prepare cluster texts & children
        all_cluster_texts = []
        all_cluster_children = []
        cluster_ids_ordered = []

        for cid, positions in cluster_to_positions.items():
            children = [node_list[p] for p in positions]
            child_texts = [c.text for c in children]
            # Token limit check (optional split)
            total_tokens = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in child_texts)
            if total_tokens > max_chunk_tokens * 5:   
                logger.info(f"Cluster {cid} too large ({total_tokens} tokens), splitting further")
                sub_embs = np.vstack([c.embeddings for c in children])
                sub_memberships, _, _ = main_clustering(sub_embs, max_clusters=10)
                sub_cluster_map = defaultdict(list)
                for sub_pos, sub_cids in enumerate(sub_memberships):
                    for scid in sub_cids:
                        sub_cluster_map[scid].append(sub_pos)
                for scid, sub_positions in sub_cluster_map.items():
                    sub_children = [children[p] for p in sub_positions]
                    sub_texts = [c.text for c in sub_children]
                    all_cluster_texts.append(sub_texts)
                    all_cluster_children.append(sub_children)
                    cluster_ids_ordered.append(f"{cid}_sub{scid}")
            else:
                all_cluster_texts.append(child_texts)
                all_cluster_children.append(children)
                cluster_ids_ordered.append(str(cid))

        summaries = summarize_cluster(all_cluster_texts, llm_client, max_tokens=512)

        # Embed summaries
        if not summaries:
            break
        summary_embs = embedding(summaries, embedding_model, batch_size=32)

        # Create parent nodes
        new_nodes = {}
        for i, (summary_text, summary_emb, children_list) in enumerate(zip(summaries, summary_embs, all_cluster_children)):
            parent = Node(
                index=next_idx,
                title=title,
                text=summary_text,
                embeddings=summary_emb,
                children={c.index for c in children_list},
                layer=layer,
                cluster_id=cluster_ids_ordered[i],
                metadata={"stage": "branch", "n_children": len(children_list)}
            )
            # Add parent to children's parents set
            for child in children_list:
                child.parents.add(parent.index)
            all_nodes[next_idx] = parent
            new_nodes[next_idx] = parent
            next_idx += 1

        if not new_nodes:
            break
        layer_to_nodes[layer] = list(new_nodes.keys())
        current_nodes = new_nodes

    # Determine root nodes (highest layer)
    max_layer_built = max(layer_to_nodes.keys())
    root_indices = layer_to_nodes[max_layer_built]
    root_nodes = {idx: all_nodes[idx] for idx in root_indices}
    for rid in root_indices:
        all_nodes[rid].metadata["stage"] = "root"

    total_time = time.time() - start_time
    logger.info(f"Tree built: {len(all_nodes)} nodes, {len(root_nodes)} roots, {max_layer_built+1} layers")
    return Tree(
        all_nodes=all_nodes,
        root_nodes=root_nodes,
        leaf_nodes=leaf_nodes,
        num_layers=max_layer_built+1,
        layer_to_nodes=layer_to_nodes,
        processing_time=total_time,
        metadata={"title": title, "doc_id": doc_id}
    )

# -------------------------------------------------------------------
# 9. SAVE TO QDRANT
# -------------------------------------------------------------------
def save_tree_to_qdrant(
    qdrant_client,
    tree_dict: Tree,                  
    collection_nodes: str,
    collection_stats: str,
    tokenizer_for_token_count=None,  
) -> None:
    """
    Save tree nodes and statistics into Qdrant collections.
    """
    doc_id = tree_dict.metadata.get("doc_id", "unknown")

    # Create UUID mapping from node index
    node_id_map = {}
    for idx in tree_dict.all_nodes.keys():
        node_id_map[idx] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{idx}"))

    # Root UUIDs
    max_layer = max(tree_dict.layer_to_nodes.keys()) if tree_dict.layer_to_nodes else 0
    root_indices = tree_dict.layer_to_nodes.get(max_layer, [])
    root_uuids = [node_id_map[idx] for idx in root_indices]

    # Layer distribution
    layer_distribution = {str(layer): len(nodes) for layer, nodes in tree_dict.layer_to_nodes.items()}

    # Optional: total tokens
    total_tokens = 0
    if tokenizer_for_token_count:
        total_tokens = sum(len(tokenizer_for_token_count.encode(node.text, add_special_tokens=False))
                           for node in tree_dict.all_nodes.values())

    # Save stats
    stats_point = PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"stats_{doc_id}")),
        vector=[0.0],  
        payload={
            "doc_id": doc_id,
            "title": tree_dict.metadata.get("title", "Unknown"),
            "processing_time": tree_dict.processing_time,
            "num_layers": tree_dict.num_layers,
            "root_uuids": root_uuids,
            "n_root_nodes": len(root_uuids),
            "total_nodes": len(tree_dict.all_nodes),
            "total_tokens": total_tokens,
            "layer_distribution": layer_distribution,
        }
    )
    qdrant_client.upsert(collection_name=collection_stats, points=[stats_point])

    # Save nodes in batches
    node_points = []
    for idx, node in tree_dict.all_nodes.items():
        current_uuid = node_id_map[idx]
        child_uuids = [node_id_map[c_idx] for c_idx in node.children if c_idx in node_id_map]
        parent_indices = list(node.parents) if hasattr(node, 'parents') and node.parents else []
        parent_uuids = [node_id_map[p_idx] for p_idx in parent_indices if p_idx in node_id_map]
        stage = node.metadata.get("stage", "branch")

        node_points.append(PointStruct(
            id=current_uuid,
            vector=node.embeddings.flatten().tolist(),
            payload={
                "doc_id": doc_id,
                "title": node.title,
                "node_index": node.index,
                "text": node.text,
                "layer": node.layer,
                "stage": stage,
                "is_root": stage == "root",
                "cluster_id": getattr(node, 'cluster_id', None),
                "parent_uuids": parent_uuids,
                "children_uuids": child_uuids,
                "n_children": node.metadata.get("n_children", 0)
            }
        ))

    for i in range(0, len(node_points), 100):
        qdrant_client.upsert(collection_name=collection_nodes, points=node_points[i:i+100])

    logger.info(f"Saved tree for doc_id={doc_id}: {len(node_points)} nodes, {len(root_uuids)} roots")