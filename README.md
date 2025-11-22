# 📄 AI-Powered PDF Assistant  
Built by **Sahil Bhayre**

An intelligent PDF-question-answering system powered by Google's Gemini model and LangChain.  
Upload PDFs, ask questions, generate summaries, extract information, and much more.

---

## 🚀 Features
- Upload multiple PDFs (up to 200MB each)
- Chunking & indexing with adjustable chunk size/overlap
- RAG-based question answering
- Persona-based responses (Strict RAG / Hybrid AI)
- Summary generation per page & full document
- Save & load vector indexes
- Audio input (speech-to-text)
- Chat history export (PDF)
- Mobile-friendly UI
- Easy to use—no setup required

---

## 🧠 Tech Stack
- **Google Gemini 1.5 Flash** for LLM  
- **LangChain** for RAG pipeline  
- **FAISS** for vector store  
- **Streamlit** UI  
- **SpeechRecognition + gTTS** (optional)  
- **Python**

---

## 📁 Project Structure
```
📦 project/
 ┣ 📄 main.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
 ┗ 📁 .streamlit/
```

---

## ▶️ How to Use
1. Upload PDF(s)  
2. Click **Process & Index PDFs**  
3. Choose:  
   - Chunk size  
   - Overlap  
   - Persona  
   - Retriever Top-K  
4. Type your question  
5. Get instant AI answers  
6. Download summaries or vector index  

---

## 🌐 Live Demo
🔗 **https://sahil-bhayre-rag-app-kqp3itcweumphvq6fo9mwn.streamlit.app/**

---

## 📦 Local Installation
```
git clone https://github.com/youruser/gemini-rag-app/
cd gemini-rag-app
pip install -r requirements.txt
streamlit run main.py
```

---

## 🧾 LICENSE
Open-source — feel free to modify & improve.

---

## 👨‍💻 Author
**Sahil Bhayre**  
AI & Full‑Stack Developer  



