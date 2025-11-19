# main.py — Gemini RAG app (All features except voice)
import os
import io
import json
import tempfile
from typing import List, Dict, Optional, Tuple

import streamlit as st

# ---------- GOOGLE GEMINI SETUP ----------
from google.generativeai.client import configure
import google.generativeai as genai

# LangChain Google wrappers
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

# LangChain utilities
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional extras (pdf preview, PDF report)
try:
    from pdf2image import convert_from_bytes
    HAS_PDF2IMAGE = True
except Exception:
    HAS_PDF2IMAGE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False


# ---------------- CONFIG ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("⚠ GOOGLE_API_KEY not set. Use PowerShell:\n$env:GOOGLE_API_KEY=\"your_key\"")
# configure global SDK (used implicitly by wrappers)
configure(api_key=GOOGLE_API_KEY)

# Model choices
LLM_MODEL = "models/gemini-2.5-flash"
EMBED_MODEL = "models/text-embedding-004"


# ---------------- SESSION STATE INIT (Pylance-friendly) ----------------
if "vs_by_file" not in st.session_state:
    st.session_state.vs_by_file = {}  # type: Dict[str, FAISS]
if "chunks_by_file" not in st.session_state:
    st.session_state.chunks_by_file = {}  # type: Dict[str, List[dict]]
if "docs_by_file" not in st.session_state:
    st.session_state.docs_by_file = {}  # type: Dict[str, List]
if "combined_vs" not in st.session_state:
    st.session_state.combined_vs = None
if "page_summaries" not in st.session_state:
    st.session_state.page_summaries = {}  # type: Dict[str, List[dict]]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # type: List[Dict]
if "persona" not in st.session_state:
    st.session_state.persona = "Strict RAG"
if "theme" not in st.session_state:
    st.session_state.theme = "dark"


# ---------------- HELPERS ----------------
def load_pdf_pages(uploaded_file) -> List:
    """Load PDF pages using PyPDFLoader — returns list of Documents with page_content & metadata."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    loader = PyPDFLoader(tmp.name)
    docs = loader.load()
    for i, d in enumerate(docs):
        d.metadata["source_file"] = uploaded_file.name
        d.metadata["page_index"] = i + 1
    try:
        os.remove(tmp.name)
    except Exception:
        pass
    return docs


def chunk_documents(docs, chunk_size: int = 900, chunk_overlap: int = 200) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[dict] = []
    for d in docs:
        meta = dict(d.metadata) if hasattr(d, "metadata") else {}
        page = meta.get("page_index") or meta.get("page")
        meta["page"] = page
        parts = splitter.split_text(d.page_content)
        for i, p in enumerate(parts):
            chunks.append({"text": p, "metadata": {**meta, "chunk_id": i}})
    return chunks


def build_faiss_from_chunks(chunks: List[dict]) -> FAISS:
    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_texts(texts, embeddings, metadatas=metas)
    return vs


def save_chunks_json(name: str, chunks: List[dict]) -> None:
    os.makedirs("indexes", exist_ok=True)
    with open(os.path.join("indexes", f"{name}_chunks.json"), "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)


def load_chunks_json(name: str) -> Optional[List[dict]]:
    p = os.path.join("indexes", f"{name}_chunks.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def save_faiss_local(vs: FAISS, name: str) -> bool:
    try:
        os.makedirs("indexes", exist_ok=True)
        vs.save_local(os.path.join("indexes", name))
        return True
    except Exception:
        return False


def load_faiss_local(name: str) -> Optional[FAISS]:
    p = os.path.join("indexes", name)
    if os.path.exists(p):
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        try:
            return FAISS.load_local(p, embeddings)
        except Exception:
            return None
    return None


def highlight_snippet(snippet: str, query: str) -> str:
    import re
    q_terms = [t for t in re.split(r"\W+", query) if len(t) > 2]
    out = snippet
    for t in sorted(set(q_terms), key=len, reverse=True)[:8]:
        out = re.sub(f"(?i)({re.escape(t)})", r"<mark>\1</mark>", out)
    return out


# ---------------- PROMPTS / PERSONAS ----------------
PERSONAS = {
    "Strict RAG": "Answer using only the provided document context. If not present, say exactly: 'The document does not contain that information.'",
    "Teacher": "Answer as a friendly teacher. Use short examples. Base all facts on the document or say you can't find them.",
    "Expert": "Answer precisely and concisely like an expert. Base on the document only.",
    "Beginner-friendly": "Explain simply for a beginner. Use document content only."
}


def run_rag_answer(context: str, question: str, persona: str) -> str:
    llm = GoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    persona_text = PERSONAS.get(persona, PERSONAS["Strict RAG"])
    prompt = ChatPromptTemplate.from_template("""
{persona_text}

Context:
{context}

Question:
{question}
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"persona_text": persona_text, "context": context, "question": question})


def run_llm_simple(prompt_text: str) -> str:
    llm = GoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    return llm.invoke(prompt_text)


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Gemini RAG ", layout="wide")
st.title("📚 Google Gemini RAG — Sahil Bhayre")

left, right = st.columns([1, 2])

with left:
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=900, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=200, step=50)
    top_k = st.slider("Retriever top-k", 1, 12, 4)
    process_btn = st.button("Process & Index PDFs")

    st.markdown("---")
    if st.button("Save indexes to disk"):
        saved_any = False
        for fname, vs in st.session_state.vs_by_file.items():
            ok = save_faiss_local(vs, fname)
            ch = st.session_state.chunks_by_file.get(fname)
            if ch:
                save_chunks_json(fname, ch)
                ok = ok or True
            saved_any = saved_any or ok
        if st.session_state.combined_vs:
            save_faiss_local(st.session_state.combined_vs, "combined_index")
            save_chunks_json("combined_index", sum(list(st.session_state.chunks_by_file.values()), []))
            saved_any = True
        if saved_any:
            st.success("Saved indexes & chunks to indexes/")
        else:
            st.info("Nothing to save or save failed.")

    if st.button("Load indexes from disk"):
        loaded_any = False
        if os.path.exists("indexes"):
            for name in os.listdir("indexes"):
                path = os.path.join("indexes", name)
                if os.path.isdir(path):
                    emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
                    try:
                        vs = FAISS.load_local(path, emb)
                        if name == "combined_index":
                            st.session_state.combined_vs = vs
                        else:
                            st.session_state.vs_by_file[name] = vs
                        loaded_any = True
                    except Exception:
                        pass
            # load chunk JSONs
            for f in os.listdir("indexes"):
                if f.endswith("_chunks.json"):
                    key = f.replace("_chunks.json", "")
                    try:
                        with open(os.path.join("indexes", f), "r", encoding="utf-8") as fh:
                            st.session_state.chunks_by_file[key] = json.load(fh)
                            loaded_any = True
                    except Exception:
                        pass
        if loaded_any:
            st.success("Loaded saved indexes/chunks (where possible).")
        else:
            st.info("No saved indexes/chunks found.")

    st.markdown("---")
    st.markdown("Optional features (preview/export)")
    st.write("pdf2image (preview):", "available" if HAS_PDF2IMAGE else "missing")
    st.write("reportlab (PDF export):", "available" if HAS_REPORTLAB else "missing")

with right:
    st.markdown("### Ask (RAG)")
    scope_options = ["All documents"] + list(st.session_state.docs_by_file.keys())
    selected_scope = st.selectbox("Query scope", scope_options, index=0)
    st.session_state.selected_doc = selected_scope
    persona = st.selectbox("Persona", list(PERSONAS.keys()), index=list(PERSONAS.keys()).index(st.session_state.persona) if st.session_state.persona in PERSONAS else 0)
    st.session_state.persona = persona
    query = st.text_input("Ask a question about the uploaded PDF(s)")
    ask_btn = st.button("Ask")
    summarize_btn = st.button("Generate per-page summaries (all)")
    full_summary_btn = st.button("Generate Full Document Summary")
    clear_btn = st.button("Clear chat & session")

    if clear_btn:
        st.session_state.chat_history = []
        st.session_state.vs_by_file = {}
        st.session_state.chunks_by_file = {}
        st.session_state.docs_by_file = {}
        st.session_state.combined_vs = None
        st.session_state.page_summaries = {}
        st.success("Cleared session data.")


# ---------------- PROCESS UPLOADS ----------------
if process_btn:
    if not uploaded:
        st.error("Please upload one or more PDFs.")
    else:
        combined_chunks = []
        for f in uploaded:
            docs = load_pdf_pages(f)
            st.session_state.docs_by_file[f.name] = docs
            chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.session_state.chunks_by_file[f.name] = chunks
            with st.spinner(f"Indexing {f.name} ..."):
                try:
                    vs = build_faiss_from_chunks(chunks)
                    st.session_state.vs_by_file[f.name] = vs
                except Exception as e:
                    st.error(f"Index creation failed for {f.name}: {e}")
            combined_chunks.extend(chunks)
        # build combined index
        try:
            st.session_state.combined_vs = build_faiss_from_chunks(combined_chunks)
            st.success("Indexed uploaded PDFs.")
        except Exception as e:
            st.error(f"Combined index build failed: {e}")


# ---------------- PER-PAGE SUMMARIES ----------------
if summarize_btn:
    if not st.session_state.docs_by_file:
        st.error("No uploaded docs to summarize.")
    else:
        llm = GoogleGenerativeAI(model=LLM_MODEL, temperature=0)
        for fname, docs in st.session_state.docs_by_file.items():
            sums = []
            for d in docs:
                prompt = f"Summarize this page in 2-4 concise sentences:\n\n{d.page_content}"
                try:
                    s = llm.invoke(prompt)
                except Exception as e:
                    s = f"Error generating summary: {e}"
                sums.append({"page": d.metadata.get("page_index"), "summary": s})
            st.session_state.page_summaries[fname] = sums
        st.success("Per-page summaries generated.")


# ---------------- FULL DOCUMENT SUMMARY ----------------
if full_summary_btn:
    if not st.session_state.docs_by_file:
        st.error("No uploaded docs.")
    else:
        all_text = "\n\n".join([d.page_content for docs in st.session_state.docs_by_file.values() for d in docs])
        prompt = f"Write a concise 1-page summary of the following document:\n\n{all_text}"
        try:
            full = run_llm_simple(prompt)
        except Exception as e:
            full = f"Error generating summary: {e}"
        st.session_state.page_summaries["__full_summary__"] = full
        st.success("Full document summary generated.")


# ---------------- RETRIEVAL / RAG ----------------
def retrieve_docs(scope: str, q: str, k: int):
    vs = st.session_state.combined_vs if scope == "All documents" else st.session_state.vs_by_file.get(scope)
    if not vs:
        return []
    try:
        res = vs.similarity_search_with_score(q, k=k)
        return res
    except Exception:
        try:
            retriever = vs.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(q)
            return [(d, None) for d in docs]
        except Exception:
            return []


if ask_btn:
    if not query or not query.strip():
        st.error("Please type a question.")
    else:
        results = retrieve_docs(st.session_state.selected_doc, query, top_k)
        if not results:
            st.error("No results — ensure you processed PDFs and selected correct scope.")
        else:
            docs = [r[0] for r in results]
            context = "\n\n".join([d.page_content for d in docs])
            with st.spinner("Generating answer..."):
                try:
                    answer = run_rag_answer(context, query, st.session_state.persona)
                except Exception as e:
                    st.error(f"LLM failed: {e}")
                    answer = "Error generating answer."
            # save to chat
            st.session_state.chat_history.append({"q": query, "a": answer, "sources": [{"file": d.metadata.get("source_file"), "page": d.metadata.get("page"), "snippet": d.page_content[:300]} for d in docs]})
            # display
            st.markdown("## 🔎 Answer")
            st.write(answer)

            # follow-ups
            follow_prompt = f"Based on the question and answer, suggest 3 concise follow-up questions:\nQ: {query}\nA: {answer}\nFollow-ups:"
            try:
                follow_text = run_llm_simple(follow_prompt)
            except Exception:
                follow_text = ""
            fu_lines = [ln.strip(" -•\t\n") for ln in follow_text.splitlines() if ln.strip()]
            st.markdown("### 💡 Follow-up suggestions")
            for i, fu in enumerate(fu_lines[:6]):
                if st.button(fu, key=f"fu_{i}"):
                    st.session_state.query = fu
                    st.info(f"Follow-up placed into input: {fu}")

            # show retrieved sources
            st.markdown("---")
            st.markdown("### 📚 Retrieved sources")
            for idx, (d, score) in enumerate(results):
                meta = d.metadata
                src = meta.get("source_file", "unknown")
                page = meta.get("page", meta.get("page_index", "N/A"))
                st.markdown(f"**{idx+1}. {src} — page {page} — score: {score}**")
                snippet = d.page_content[:1500]
                st.markdown(highlight_snippet(snippet, query), unsafe_allow_html=True)
                # page summary if available
                sums = st.session_state.page_summaries.get(src)
                if sums:
                    for s in sums:
                        if s["page"] == page:
                            st.markdown(f"*Page summary:* {s['summary']}")
                            break


# ---------------- CHAT HISTORY & EXPORTS ----------------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 💬 Chat history")
    for t in st.session_state.chat_history[-50:]:
        st.markdown(f"**Q:** {t['q']}")
        st.markdown(f"**A:** {t['a']}")
    hist_text = "\n\n".join([f"Q: {t['q']}\nA: {t['a']}" for t in st.session_state.chat_history])
    st.download_button("Download chat (.txt)", data=hist_text, file_name="chat_history.txt", mime="text/plain")

    if st.button("Generate report (TXT & PDF)"):
        report_lines = ["Gemini RAG Report", "="*40, ""]
        for t in st.session_state.chat_history:
            report_lines.append(f"Q: {t['q']}\nA: {t['a']}\n")
        report_txt = "\n".join(report_lines)
        st.download_button("Download report (.txt)", data=report_txt, file_name="report.txt", mime="text/plain")
        if HAS_REPORTLAB:
            try:
                buf = io.BytesIO()
                c = canvas.Canvas(buf, pagesize=letter)
                w, h = letter
                y = h - 72
                c.setFont("Helvetica-Bold", 14)
                c.drawString(72, y, "Gemini RAG Report")
                y -= 28
                c.setFont("Helvetica", 10)
                for t in st.session_state.chat_history:
                    lines = (f"Q: {t['q']}\nA: {t['a']}").splitlines()
                    for ln in lines:
                        if y < 72:
                            c.showPage(); y = h - 72
                        c.drawString(72, y, ln[:120])
                        y -= 12
                c.save()
                buf.seek(0)
                st.download_button("Download report (.pdf)", data=buf, file_name="report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
        else:
            st.info("Install reportlab to enable PDF export: pip install reportlab")


# ---------------- PDF page preview (optional) ----------------
if HAS_PDF2IMAGE and st.session_state.docs_by_file:
    st.markdown("---")
    st.markdown("## 📄 PDF Preview")
    for fname, docs in st.session_state.docs_by_file.items():
        st.markdown(f"### {fname}")
        # To preview, ask user to re-upload the file via uploader and convert_from_bytes(uploaded.getbuffer())
        st.info("For a visual preview, re-upload the specific file in the uploader and view the preview area.")


st.caption("Features included: multi-PDF, per-page + full summaries, RAG retrieval, highlights, follow-ups, chat history, save/load index, export. No voice features included.")

