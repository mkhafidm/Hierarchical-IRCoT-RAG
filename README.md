# Hierarchical RAG with Interleaved Chain-of-Thought Tree Traversal

A hierarchical Retrieval-Augmented Generation (RAG) system that integrates 
Interleaved Chain-of-Thought (IRCoT) reasoning into DFS-based tree traversal 
for multi-hop question answering.

## Overview

This system combines RAPTOR-style hierarchical tree construction with IRCoT-guided 
recursive DFS traversal. Documents are encoded into a multi-layer tree structure 
using GMM clustering, then retrieved via a combined scoring function:

**S(q, r, n) = α · cos(q, n) + (1 − α) · cos(r, n)**

where `α` controls the balance between query similarity and reasoning-chain similarity. Higher `α` means the system relies more on direct query matching, while lower `α` activates IRCoT reasoning influence. Reasoning chains are generated only when α < 1.0 and at non-leaf nodes.

During traversal, a node is selected only if its similarity score exceeds `τ` (selection threshold); nodes below this are pruned immediately. If the score improvement between a parent and child node falls below `δ` (delta threshold), the traversal backtracks and collects the parent node as context instead of going deeper.
