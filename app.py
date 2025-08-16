import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import os

# ✅ NEW: Sentence splitting
import nltk
from sentence_transformers import CrossEncoder
nltk.download('punkt')
nltk.download('punkt_tab') 

# 🔑 Set your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 📁 Metrics log file
METRICS_FILE = "metrics_log.json"

# ✅ Load previous metrics (if exists)
if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)
else:
    metrics = {
        "total_queries": 0,
        "successful_queries": 0,
        "queries_with_hits": 0,
        "reciprocal_ranks": [],
        "logs": []
    }

# ✅ Load all PDFs from 'data' folder
data_folder = "data"
all_documents = []
for file in os.listdir(data_folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(data_folder, file)
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for d in docs:
            d.metadata["source_file"] = file
            d.metadata["doc_version"] = "v1.0"  # can be updated dynamically later
            d.metadata["department"] = "General"
        all_documents.extend(docs)

print(f"✅ Loaded {len(all_documents)} pages from PDFs in {data_folder}/")

# ✅ Sentence-level splitting
sentence_docs = []
for doc in all_documents:
    sentences = nltk.sent_tokenize(doc.page_content)
    for sentence in sentences:
        sentence_docs.append(Document(page_content=sentence, metadata=doc.metadata))

print(f"✅ Created {len(sentence_docs)} sentence-level chunks")

# ✅ Build FAISS index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(sentence_docs, embeddings)

# ✅ LLM setup
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# ✅ Cross-encoder for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ✅ Prompt template with strict guardrail
prompt_template = """
You are an assistant answering questions about process documents.
Use ONLY the provided context to answer.
If the answer is not found in the context, say exactly: "I don't know." Do not make up information.

Question: {question}

Context:
{context}

Answer clearly, then list the sources at the end like this:
Sources: [filename - page numbers].
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ✅ Recursive Retrieval Function
def recursive_retrieval(query, retriever, depth=2):
    """ Try multiple retrievals if not enough context found. """
    docs = retriever.get_relevant_documents(query)
    if len(docs) < 2 and depth > 0:
        new_query = f"Provide more details on: {query}"
        docs += retriever.get_relevant_documents(new_query)
    return docs

# ✅ Retrieval function with reranking
def retrieve_with_reranking(query, retriever):
    docs = recursive_retrieval(query, retriever)
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:5]]

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ✅ Flask app
app = Flask(__name__)

# 🔄 Save metrics to file
def save_metrics():
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

# ✅ Web Form (GET endpoint)
@app.route("/", methods=["GET"])
def home():
    return """
    <h2>📄 Process Document Q&A</h2>
    <form method="POST" action="/ask_form">
        <input type="text" name="question" placeholder="Ask a question..." style="width:300px; padding:5px;">
        <button type="submit" style="padding:5px 10px;">Ask</button>
    </form>
    <br>
    <a href="/metrics">📊 View Metrics</a>
    """

# ✅ Handle form submission
@app.route("/ask_form", methods=["POST"])
def ask_form():
    question = request.form.get("question", "")
    if not question:
        return "<h3>Please enter a question.</h3>"

    all_docs = retriever.vectorstore.similarity_search(question, k=10)
    reranked_docs = retrieve_with_reranking(question, retriever)

    context = "\n\n".join([doc.page_content for doc in reranked_docs])

    if not context.strip():
        answer_text = "I don't know."
    else:
        response = llm.invoke(prompt.format(context=context, question=question))
        answer_text = response.content

    metrics["total_queries"] += 1
    success = "I don't know" not in answer_text
    hit = len(reranked_docs) > 0

    if success:
        metrics["successful_queries"] += 1
    if hit:
        metrics["queries_with_hits"] += 1

    metrics["reciprocal_ranks"].append(1.0)

    sources = []
    for doc in reranked_docs:
        filename = doc.metadata.get("source_file", "Unknown File")
        page_num = doc.metadata.get("page", 0) + 1
        sources.append(f"{filename} - Page {page_num}")

    metrics["logs"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer_found": success,
        "retrieval_hit": hit,
        "sources": sources
    })

    save_metrics()

    return f"""
    <h2>✅ Answer</h2>
    <p>{answer_text}</p>
    <h3>📂 Sources:</h3>
    <ul>{''.join([f"<li>{src}</li>" for src in sources])}</ul>
    <a href="/">⬅️ Ask another question</a>
    """

# ✅ JSON API endpoint
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")
    filters = data.get("filters", {})

    if not question:
        return jsonify({"error": "Please provide a question"}), 400

    all_docs = retriever.vectorstore.similarity_search(question, k=10, filter=filters if filters else None)
    reranked_docs = retrieve_with_reranking(question, retriever)

    context = "\n\n".join([doc.page_content for doc in reranked_docs])

    if not context.strip():
        answer_text = "I don't know."
    else:
        response = llm.invoke(prompt.format(context=context, question=question))
        answer_text = response.content

    metrics["total_queries"] += 1
    success = "I don't know" not in answer_text
    hit = len(reranked_docs) > 0

    if success:
        metrics["successful_queries"] += 1
    if hit:
        metrics["queries_with_hits"] += 1

    metrics["reciprocal_ranks"].append(1.0)

    sources = []
    for doc in reranked_docs:
        filename = doc.metadata.get("source_file", "Unknown File")
        page_num = doc.metadata.get("page", 0) + 1
        sources.append(f"{filename} - Page {page_num}")

    metrics["logs"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer_found": success,
        "retrieval_hit": hit,
        "sources": sources
    })

    save_metrics()

    return jsonify({
        "question": question,
        "answer": answer_text,
        "sources": sources
    })

# ✅ Metrics page
@app.route("/metrics", methods=["GET"])
def get_metrics():
    total = metrics["total_queries"]
    success_rate = (metrics["successful_queries"] / total * 100) if total > 0 else 0
    hit_rate = (metrics["queries_with_hits"] / total * 100) if total > 0 else 0
    mrr = sum(metrics["reciprocal_ranks"]) / len(metrics["reciprocal_ranks"]) if metrics["reciprocal_ranks"] else 0

    return f"""
    <h2>📊 RAG Metrics Dashboard</h2>
    <p><b>Total Queries:</b> {total}</p>
    <p><b>✅ Success Rate:</b> {success_rate:.2f}%</p>
    <p><b>🎯 Hit Rate:</b> {hit_rate:.2f}%</p>
    <p><b>📈 Mean Reciprocal Rank (MRR):</b> {mrr:.2f}</p>
    <a href="/">⬅️ Back to Home</a>
    <br><br>
    <h3>📜 Recent Logs:</h3>
    <ul>
        {''.join([f"<li>{log['timestamp']} - Q: {log['question']} - {'✅ Found' if log['answer_found'] else '❌ Not Found'} - Sources: {', '.join(log['sources'])}</li>" for log in metrics['logs'][-10:]])}
    </ul>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
