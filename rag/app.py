import re
import requests
import tiktoken

import pandas as pd
import streamlit as st

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.llms import HuggingFaceTextGenInference
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain.chains import HypotheticalDocumentEmbedder, LLMChain

@st.cache_resource
def get_embedder():
    return SentenceTransformerEmbeddings(model_name="bert-base-nli-stsb-mean-tokens")

def get_hyde_embedder(model="LLaMA 2 (13B)"):
    # Base embeddings to use
    base_embeddings = SentenceTransformerEmbeddings(model_name="bert-base-nli-stsb-mean-tokens")

    # Define model URL based on selection
    if model == "LLaMA 2 (13B)":
        model_url = "http://llm-llama2:80/"
    elif model == "LLaMA 2 Chat (13B)":
        model_url = "http://llm-llama2-chat:80/"
    
    # Instantiate the LLM
    llm = HuggingFaceTextGenInference(
        inference_server_url="http://llm-llama2:80/",
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=rep_pen,
    )

    # Build HyDE embedder
    embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search")
    
    return embeddings

@st.cache_resource
def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
    )

@st.cache_data
def tiktoken_len(text: str) -> int:
    """Returns the number of tokens in a string of text"""
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

@st.cache_data
def add_metadata(page_text, page_num, filename) -> list[Document]:
    """Takes a list of strings and returns a list of Documents
    with page number metadata"""

    doc_chunks = []

    # Split page text into chunks
    text_splitter = get_splitter()
    chunks = text_splitter.split_text(page_text)
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk, metadata={"page": page_num, "chunk": i, "name": filename}
        )
        # Add page number to metadata
        doc.metadata["source"] = f"page {doc.metadata['page']}"
        doc_chunks.append(doc)

    return doc_chunks

@st.cache_data
def parse_pdf(file) -> list[str]:
    """Returns a list of pages from the supplied PDF file. Each
    list item is the extracted text of one page.
    """
    pdf = PdfReader(file)
    document = []
    for page in pdf.pages:
        text = page.extract_text()
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        document.append(text)
    return document

@st.cache_resource
def get_db():
    # Create a vectorstore
    qdrant_url = "http://qdrant:6333"
    client = QdrantClient(url=qdrant_url)
    qdrant = Qdrant(
        client=client,
        collection_name="st_rag",
        embeddings=embedder,
        # prefer_grpc=True,
    )
    return qdrant

def reset_qdrant():
    qdrant_url = "http://qdrant:6333"
    client = QdrantClient(url=qdrant_url)
    client.recreate_collection(
        collection_name="st_rag",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    if "qdrant_docs" in st.session_state:
        del st.session_state["qdrant_docs"]
        st.session_state.qdrant_docs = list_qdrant_documents()

def delete_qdrant_doc(doc_name):

    qdrant_url = "http://qdrant:6333"
    client = QdrantClient(url=qdrant_url)

    # Get all points for this document
    results, _ = client.scroll(
        collection_name="st_rag",
        scroll_filter=models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.name",
                    match=models.MatchValue(value=doc_name),
                )
            ],
        ),
    )

    delete_ids = [record.id for record in results]

    client.delete(
        collection_name="st_rag",
        points_selector=models.PointIdsList(
            points=delete_ids,
        ),
    )

    if "qdrant_docs" in st.session_state:
        del st.session_state["qdrant_docs"]
        st.session_state.qdrant_docs = list_qdrant_documents()


def list_qdrant_documents():
    qdrant_url = "http://qdrant:6333"
    client = QdrantClient(url=qdrant_url)

    try:
        results, _ = client.scroll(
            collection_name="st_rag",
            scroll_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.name",
                        match=models.MatchExcept(**{"except": ["return_all.pdf"]}),
                    )
                ],
            ),
        )

        file_names = []
        for record in results:
            if record.payload["metadata"]["name"] not in file_names:
                file_names.append(record.payload["metadata"]["name"])
    except:
        reset_qdrant()

    return file_names

@st.cache_resource
def add_to_db(uploaded_file):
    page_progress = st.progress(0, "Extracting text from PDF")
    page_texts = parse_pdf(uploaded_file)
    documents = []
    for idx, page_text in enumerate(page_texts):
        page_progress.progress(idx/len(page_texts), "Extracting text from PDF")
        documents += add_metadata(page_text, idx+1, uploaded_file.name)

    page_progress.progress(0.98, "Embedding...")

    qdrant_url = "http://qdrant:6333"
    qdrant = Qdrant.from_documents(
        documents,
        embedder,
        url=qdrant_url,
        prefer_grpc=True,
        collection_name="st_docqa1",
    )

    page_progress.empty()

    return qdrant

def generate_response(query_text, model, db):
    # Load document if file is uploaded
    if db is not None:        

        # Define model URL based on selection
        if model == "LLaMA 2 (13B)":
            model_url = "http://llm-llama2:80/"
        elif model == "LLaMA 2 Chat (13B)":
            model_url = "http://llm-llama2-chat:80/"
        
        # Create retriever interface
        retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 5})
        
        # Instantiate the LLM
        llm = HuggingFaceTextGenInference(
            inference_server_url=model_url,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=rep_pen,
        )
        
        # Create QA chain
        qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, chain_type="map_reduce",
                                                         return_source_documents=True)
        res = qa({"question": query_text})#, return_only_outputs=True)
        return res["answer"], res["source_documents"]

# Page title & setup
st.set_page_config(page_title='Retrieval Augmented Generation (RAG)')
st.title('Retrieval Augmented Generation (RAG)')

# Sidebar setup for model parameters
st.sidebar.title('LLM Parameters')
model = st.sidebar.selectbox("Choose Model", ["LLaMA 2 (13B)", "LLaMA 2 Chat (13B)"])
max_new_tokens = st.sidebar.slider('max_new_tokens', 0, 2048, 128)
hyde_emb = st.sidebar.checkbox('HyDE Embedding Process', value=False)
top_p = st.sidebar.slider('top_p', 0.0, 2.0, 0.95)
top_k = st.sidebar.slider('top_k', 0, 50, 10)
temperature = st.sidebar.slider('temperature', 0.0, 1.0, 0.005, 0.001)
rep_pen = st.sidebar.slider('repetition_penalty', 1.0, 2.0, 1.03, 0.01)

# Configure embedding model
with st.spinner('Starting up Embedding model...'):
    if hyde_emb:
        embedder = get_hyde_embedder(model=model)
    else:
        embedder = get_embedder()

    tokenizer = tiktoken.get_encoding("p50k_base")
    db = get_db()

# Vector'd documents
if 'qdrant_docs' not in st.session_state:
    st.session_state.qdrant_docs = list_qdrant_documents()

# File Management Tabs
doc_tab, upload_tab = st.tabs(["Existing Documents", "Upload New Document"])

with doc_tab:
    st.caption("Press 'R' to refresh document list. Click the checkbox next to the document's name to remove it, or the button below to remove everything.")
    for doc in st.session_state.qdrant_docs:
        st.checkbox(doc, value=True, on_change=delete_qdrant_doc, args=(doc,))
    if len(st.session_state.qdrant_docs) == 0:
        st.caption("No documents uploaded yet.")
    else:
        st.button("Reset Vector Store", on_click=reset_qdrant)

with upload_tab:
    # File upload
    uploaded_file = st.file_uploader('Upload a PDF', type='pdf')
    if uploaded_file:
        db = add_to_db(uploaded_file)
        del st.session_state["qdrant_docs"]
        st.session_state.qdrant_docs = list_qdrant_documents()

st.divider()

# Form input and query
answer = ""
sources = []
with st.form('rag_form', clear_on_submit=False):

    query_text = st.text_input('Question:', placeholder = 'Ask questions about your uploaded document(s)!')
    submitted = st.form_submit_button('Submit')
    
    if submitted and db:
        with st.spinner('Calculating...'):
            answer, sources = generate_response(query_text, model, db)

if len(sources) and answer:
    st.info(answer)
    for source_doc in sources:
        source_body = "**Source: " + source_doc.metadata["name"] + " (" + source_doc.metadata["source"] + ")"
        source_body += "**  \n" + source_doc.page_content
        st.success(source_body, icon="🗃️")