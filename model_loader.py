import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from openai import OpenAI
from config import EMBED_MODEL, LLM_MODEL, VLLM_URL

# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer(EMBED_MODEL, device="cpu")
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL, device="cuda:1")

@st.cache_resource
def load_tokenizer_e5():
    return AutoTokenizer.from_pretrained(EMBED_MODEL)

@st.cache_resource
def load_tokenizer_llm():
    return AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)

@st.cache_resource
def load_llm_client():
    # OpenAI client yang terhubung ke vLLM server
    return OpenAI(base_url=VLLM_URL, api_key="dummy")