import streamlit as st
from qdrant_client import QdrantClient
from config import DB_CONFIG

@st.cache_resource
def load_client(dataset_name):
    return QdrantClient(path=DB_CONFIG[dataset_name])

