# 📄 PaperCraft AI — Autonomous Research & Document Creation Agent

**PaperCraft AI** is an advanced multi-agent document generation system powered by **LangGraph**, **FastAPI**, **ChatGroq**, and **Tavily**. It autonomously analyzes a topic, conducts targeted web research, formulates a section-by-section document outline, spawns parallel worker agents to draft each section, and compiles the result into a styled PDF research report.

---

## ✨ Features

- 🧭 **Intelligent Router Node**: Analyzes topic complexity to automatically determine the research mode (`closed_book`, `hybrid`, `open_book`) and generates targeted web search queries.
- 🔍 **Web Research Engine**: Leverages Tavily API to fetch real-time, verified sources and evidence snippets.
- 📐 **Structured Document Planner**: Builds 4–6 non-overlapping, logically ordered sections complete with word targets, specific bullet points, audience context, and tone settings.
- ✍️ **Parallel Section Drafting**: Uses LangGraph's dynamic `Send` API to execute parallel LLM worker agents, writing distinct document sections simultaneously.
- 📑 **Sequential Merging & Layout Synthesis**: Automatically orders, merges, and cleans section markdown.
- 🖨️ **Automated PDF Compilation**: Formats markdown into a publication-ready PDF report (with cover page, running header/footer, custom typography, table styles, and code blocks) using `xhtml2pdf`.
- ⚡ **Async Background Job Processing**: Supports asynchronous task polling (`/api/generate` & `/api/job/{job_id}`) as well as synchronous execution.
- 🖥️ **Integrated Web UI & Browser Launcher**: Automatically launches the local web dashboard upon starting up `uvicorn`.

---

## 🏗️ Multi-Agent Architecture Workflow

```
                   ┌─────────────────┐
                   │   Start Input   │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Router Node   │ (Determines mode & search queries)
                   └────────┬────────┘
                            │
               ┌────────────┴────────────┐
               │                         │
      (Needs Research)           (Closed Book)
               │                         │
               ▼                         │
    ┌────────────────────┐               │
    │   Research Node    │               │
    │  (Tavily API Call) │               │
    └──────────┬─────────┘               │
               │                         │
               └────────────┬────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Planner Node   │ (Generates document plan & sections)
                   └────────┬────────┘
                            │
                [ LangGraph Send API ]
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │ Worker 1    │    │ Worker 2    │    │ Worker N    │ (Parallel Section Drafts)
  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Merge Node    │ (Combines sections in order)
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │    PDF Node     │ (Generates styled A4 PDF document)
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │    END State    │
                   └─────────────────┘
```

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Agent Framework**: [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph dynamic routing)
- **LLM Integration**: [LangChain Groq](https://github.com/langchain-ai/langchain) (`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`)
- **Web Search**: [Tavily API](https://tavily.com/)
- **Document Rendering**: `xhtml2pdf`, `markdown`, `pisa`
- **Data Validation**: Pydantic v2

---

## 📂 Project Structure

```text
.
├── main.py                # Core FastAPI app, LangGraph engine, and PDF compiler
├── frontend.html          # Web UI interface
├── reports/               # Output directory for generated PDF documents
├── .env                   # API keys and environment variables
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Environment Setup & Configuration

Create a `.env` file in the root directory:

```env
# Groq API Keys
GROQ_API_KEY=your_groq_api_key_here
# Optional secondary key for key rotation/routing fallback
groq1=your_groq_api_key_here
groq2=your_groq_api_key_here

# Tavily Web Search API Key
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## 📦 Installation & Running

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/papercraft-ai.git
   cd papercraft-ai
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install fastapi uvicorn langchain-groq langgraph tavily-python pydantic markdown xhtml2pdf python-dotenv
   ```

4. **Launch the Server**
   ```bash
   python main.py
   ```
   *The server will start at `http://127.0.0.1:8000` and automatically launch your default web browser.*

---

## 📡 API Reference

### 1. Check System Health
- **URL**: `GET /health`
- **Response**:
  ```json
  {
    "status": "ok",
    "tavily_enabled": true,
    "models": {
      "worker": "openai/gpt-oss-20b",
      "planner": "openai/gpt-oss-20b",
      "router": "qwen/qwen3.8-27b"
    },
    "reports_count": 5
  }
  ```

### 2. Submit Generation Job (Asynchronous)
- **URL**: `POST /api/generate`
- **Body**:
  ```json
  {
    "topic": "Quantum Computing Applications in Cryptography",
    "audience": "Graduate_student",
    "mode": "auto",
    "tone": "Academic"
  }
  ```
- **Response**:
  ```json
  {
    "job_id": "a1b2c3d4",
    "status": "queued"
  }
  ```

### 3. Check Job Status
- **URL**: `GET /api/job/{job_id}`
- **Response**: Returns job logs, current execution step, status (`running`, `completed`, `failed`), and output PDF URL when finished.

### 4. Direct Synchronous Generation
- **URL**: `POST /generate`
- **Body**: Same schema as `/api/generate`. Blocks execution until full document and PDF generation complete.

### 5. List Reports
- **URL**: `GET /reports`
- **Response**: Lists all generated PDF reports stored in the `reports/` folder.

### 6. View / Download Report PDF
- **URL**: `GET /reports/{filename}?download=false`

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
