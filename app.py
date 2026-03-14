import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

VECTOR_DB_DIR = "genrocket_db"

st.title("GenRocket AI Assistant")

@st.cache_resource
def load_qa():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k":4})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    template = """
Answer ONLY from the context.

Context:
{context}

Question:
{question}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context","question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt":prompt}
    )

    return qa


qa = load_qa()

question = st.text_input("Ask a question about GenRocket")

if question:

    with st.spinner("Searching..."):
        answer = qa.run(question)

    st.subheader("Answer")
    st.write(answer)

