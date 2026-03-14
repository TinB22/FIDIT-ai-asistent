# FIDIT AI Assistant

AI assistant for learning support based on course materials.  
The system uses a Retrieval-Augmented Generation (RAG) approach with a vector database to answer student questions using official course documents.

The application is built with:

- **Streamlit** (user interface)
- **ChromaDB** (vector database)
- **Sentence Transformers** (embeddings)
- **Ollama** (local LLM inference)
- **PyPDF** (document parsing)

---

# Project Structure

```text
FIDIT-ai-asistent-main
│
├── app.py              # Streamlit chatbot application
├── ingest.py           # Script for building the vector database
├── requirements.txt    # Project dependencies
│
├── chroma_db/          # Vector database (created after ingest)
│
├── data/
│   ├── materials/      # Course PDF materials
│   ├── mappings/       # Routing and keyword mappings
│   └── questions/      # Test questions database
│
└── src/
    ├── rag.py          # Core RAG logic and prompt engineering
```

# Installation and Setup

Running the project requires **Python**, **Ollama**, and the required Python libraries.

## 1. Open the project in a terminal
In VS Code, go to `Terminal` → `New Terminal`.  
Make sure you are in the project directory:
```bash
cd FIDIT-ai-asistent-main
```

## 2. Create a Virtual Environment (Recommended)
Create a Python virtual environment to keep dependencies isolated:

```bash
python -m venv venv
```
Activate it (windows):

```bash
venv\Scripts\activate
```


## 3. Install Python Dependencies
Install all required packages:

```bash
pip install -r requirements.txt
```


## 4. Install and Configure Ollama
1. Download and install Ollama from [ollama.com](https://ollama.com).
2. After installation, download the language model:

```bash
ollama pull mistral
```

## 5. Build the Vector Database
The system requires embeddings of the course materials before running the chatbot.  
Run the ingestion script:

```bash
python ingest.py
```

The script will:
- Read all PDF files from data/materials.
- Split documents into smaller chunks.
- Generate embeddings using Sentence Transformers.
- Store vectors in ChromaDB.


## 6. Start the Chatbot
Run the Streamlit application:

```bash
streamlit run app.py
```


# How the System Works
The assistant uses a **Retrieval-Augmented Generation (RAG)** pipeline:
1. **User submits a question.**
2. **Search:** The system searches the vector database for relevant document chunks.
3. **Context:** Retrieved context is passed to a local LLM via Ollama.
4. **Generation:** The model generates an answer grounded in the course materials using a professional academic tone.
5. **Sources:** Specific sources and page numbers are displayed to the user for verification.

---

# Notes
- **Database Rebuild:** Before running the chatbot for the first time, you **must** run `python ingest.py`.
- **Troubleshooting:** If errors related to ChromaDB appear, delete the `chroma_db/` folder and run `ingest.py` again.
- **Privacy:** The system includes a privacy layer to redact personal data before processing.

---

# License
This project is intended for educational purposes as part of the TBI course project at FIDIT.
