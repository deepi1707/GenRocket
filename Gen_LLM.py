import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
BASE_URL = "https://www.genrocket.com"
START_URL = "https://www.genrocket.com/download-literature/"

CACHE_FILE = "genrocket_pages.json"
VECTOR_DB_DIR = "genrocket_db"

MAX_PAGES = 50
MAX_WORKERS = 10

def is_internal(link):
    return urlparse(link).netloc.endswith("genrocket.com")

def clean_text(text):
    return " ".join(text.split())
    def fetch_page(url):
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")

            text = clean_text(soup.get_text())

            links = []
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if is_internal(link):
                    links.append(link)

            return {"url": url, "text": text, "links": links}

        except Exception as e:
            print("Error:", e)
            return None

        def crawl_site():

            visited = set()
            to_visit = [START_URL]
            pages = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        while to_visit and len(visited) < MAX_PAGES:

            futures = []

            while to_visit and len(futures) < MAX_WORKERS:
                url = to_visit.pop(0)

                if url not in visited:
                    visited.add(url)
                    futures.append(executor.submit(fetch_page, url))

            for future in futures:
                result = future.result()

                if result:
                    pages.append({
                        "url": result["url"],
                        "text": result["text"]
                    })

                    for link in result["links"]:
                        if link not in visited:
                            to_visit.append(link)

    return pages
if os.path.exists(CACHE_FILE):

    print("Loading cached pages...")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

else:

    print("Crawling website...")
    pages = crawl_site()

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f)

print("Total pages:", len(pages))
texts = [p["text"] for p in pages]
metadatas = [{"source": p["url"]} for p in pages]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

documents = splitter.create_documents(texts, metadatas=metadatas)
#from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
if os.path.exists(VECTOR_DB_DIR):

    print("Loading existing vector DB...")
    vectordb = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

else:

    print("Creating vector DB...")
    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vectordb.persist()
    os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
template = """
You are a QA assistant.

Answer ONLY from the provided context.

If the answer is not in the context say:
"I could not find the answer in the provided website data."

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)
while True:
    q = input("\nAsk a question (or type exit): ")

    if q.lower() == "exit":
        break

    result = qa.run(q)
    #response = chain.invoke( {"input_documents":docs, "question": user_question} )
    print("\nAnswer:", result)
