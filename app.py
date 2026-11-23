import os
import re
import sys
import uuid
import html
import time
import asyncio
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, Optional, Literal, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator
import markdown
from xhtml2pdf import pisa

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from tavily import TavilyClient
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=BASE_DIR / ".env")

app = FastAPI(title="PaperCraft AI - Autonomous Research & Document Creation Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_key1 = os.getenv("groq1") or os.getenv("GROQ_API_KEY")
groq_key2 = os.getenv("groq2") or os.getenv("groq1") or os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not groq_key1:
    print("[WARNING] No Groq API key found in .env or environment!")

planner_llm = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_key1, max_tokens=1500)
router_llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=groq_key2, max_tokens=800)
writer_llm = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_key1, max_tokens=1500)

tavily_client = None
if tavily_key:
    try:
        tavily_client = TavilyClient(api_key=tavily_key)
    except Exception as e:
        print(f"[Warning] Failed to initialize TavilyClient: {e}")

class Task(BaseModel):
    id: int = Field(description="Unique sequential integer starting from 1")
    title: str = Field(description="Specific section title")
    goal: str = Field(description="Clear section objective")
    bullets: list[str] = Field(description="3 to 6 key coverage points", min_length=3, max_length=6)
    target_words: int = Field(default=250, ge=100, le=600)

class Plan(BaseModel):
    title: str = Field(description="Meaningful, professional document title")
    audience: str = Field(default="College_student", description="Target audience")
    tone: str = Field(default="Academic", description="Document tone")
    constraints: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(description="4 to 6 logically ordered, non-overlapping section tasks", min_length=4, max_length=6)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class EvidenceItem(BaseModel):
    title: str = ""
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: list[str] = Field(default_factory=list)

class MainState(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: list[str]
    evidence: list[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[list[tuple[int, str]], operator.add]
    merge_text: str
    final: str
    pdf_path: str

class GenerateRequest(BaseModel):
    topic: str
    audience: Optional[str] = "College_student"
    mode: Optional[str] = "auto"
    tone: Optional[str] = "Academic"

ROUTER_PROMPT = """
You are an expert research router.

Analyze the topic and select the appropriate research mode.

Modes:
- closed_book: Stable knowledge that needs no external research.
- hybrid: General knowledge supported by external verification.
- open_book: Specialized, complex, or current research.

Rules:
- Return needs_research as a boolean.
- Return mode using the permitted values.
- Generate 3 focused search queries when research is required.
- Return an empty queries list when research is unnecessary.
- Never rename schema fields.
- Do not return explanations or Markdown.
"""

PLANNER_PROMPT = """
You are an expert academic research planner.

Create a comprehensive research document plan.

Requirements:
- Create a professional, compelling document title.
- Select an appropriate audience.
- Choose a suitable tone.
- Create 4 to 6 logically ordered, non-overlapping tasks.
- Every task must have 3 to 6 detailed coverage bullets.
- Every task must have a target word count between 150 and 400.
- Task IDs must be sequential, starting at 1.
- Include relevant constraints.
- Follow the Plan schema strictly. Return only structured output.
"""

WORKER_SYSTEM = """
You are an expert academic researcher and professional writer.

Your task is to write ONE complete section of a larger research
document based strictly on the assigned task.

INPUT:
- Original topic: {topic}
- Document plan: {plan}
- Assigned task: {task}
- Available evidence: {evidence}

INSTRUCTIONS:
1. Follow the assigned task's goal, title, bullets, and word target.
2. Cover every required bullet with meaningful explanations.
3. Respect the document's audience, tone, and constraints.
4. Explain important concepts, terminology, and mechanisms.
5. Include relevant examples, applications, and analysis.
6. Use logical paragraphs and meaningful subsections.
7. Maintain consistent terminology and academic quality.
8. Avoid filler, repetition, vague statements, and unrelated content.
9. Write only your assigned section.
10. Target the assigned word count.

OUTPUT:
Return only clean Markdown.
- Use # for the main section heading.
- Use ## and ### for subsections.
- Use lists, tables, and code blocks where helpful.
"""

def generate_pdf_report(
    content: str,
    title: str = "Research Report",
    subtitle: str = "Comprehensive Research Document",
    audience: str = "General",
    tone: str = "Academic",
    mode: str = "hybrid",
    author: str = "AI Research System",
    output: Optional[str] = None,
) -> str:
    if not content or not content.strip():
        raise ValueError("Document content cannot be empty.")
    normalized_content = (
        content.replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "--")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
        .replace("\u2265", ">=")
        .replace("\u2264", "<=")
        .replace("\u00b1", "+/-")
    )

    md = markdown.Markdown(extensions=["extra", "toc", "sane_lists"])
    body_html = md.convert(normalized_content)

    escaped_title = html.escape(title)
    escaped_subtitle = html.escape(subtitle)
    escaped_author = html.escape(author)
    generated_date = datetime.now().strftime("%d %B %Y")

    css = """
    @page {
        size: A4;
        margin: 20mm;
        @bottom-left {
            content: "PaperCraft AI System";
            font-size: 8pt;
            color: #64748b;
        }
        @bottom-right {
            content: "Page " counter(page);
            font-size: 8pt;
            color: #64748b;
        }
    }

    body {
        font-family: Helvetica, Arial, sans-serif;
        color: #1e293b;
        font-size: 10pt;
        line-height: 1.6;
    }

    .cover {
        page-break-after: always;
        text-align: center;
        padding-top: 55mm;
    }

    .cover h1 {
        font-size: 26pt;
        color: #0f172a;
        margin-bottom: 12px;
        line-height: 1.25;
    }

    .subtitle {
        font-size: 13pt;
        color: #64748b;
        margin-bottom: 35mm;
    }

    .cover-meta {
        font-size: 9pt;
        color: #475569;
        border-top: 1px solid #cbd5e1;
        padding-top: 15px;
        display: inline-block;
        width: 80%;
    }

    h1 {
        font-size: 18pt;
        color: #0f172a;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 5px;
        margin-top: 22pt;
    }

    h2 {
        font-size: 13pt;
        color: #0284c7;
        margin-top: 16pt;
    }

    h3 {
        font-size: 11pt;
        color: #334155;
        margin-top: 12pt;
    }

    p {
        margin-bottom: 10pt;
        text-align: justify;
    }

    ul, ol {
        margin-bottom: 10pt;
        padding-left: 20px;
    }

    li {
        margin-bottom: 4pt;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15pt 0;
        font-size: 9pt;
    }

    th {
        background: #0f172a;
        color: white;
        padding: 8px;
        text-align: left;
    }

    td {
        border: 1px solid #cbd5e1;
        padding: 6px;
    }

    code {
        font-family: Courier, monospace;
        background: #f1f5f9;
        color: #be123c;
        font-size: 8.5pt;
        padding: 2px 4px;
    }

    pre {
        background: #0f172a;
        color: #f8fafc;
        padding: 10px;
        font-family: Courier, monospace;
        font-size: 8pt;
        line-height: 1.4;
        border-radius: 4px;
    }
    """

    full_html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{escaped_title}</title>
        <style>{css}</style>
    </head>
    <body>
        <div class="cover">
            <h1>{escaped_title}</h1>
            <div class="subtitle">{escaped_subtitle}</div>
            <div class="cover-meta">
                <strong>{escaped_author}</strong> &bull; Generated: {generated_date}<br>
                Audience: {html.escape(audience)} | Tone: {html.escape(tone)} | Mode: {html.escape(mode)}
            </div>
        </div>

        <main class="document">
            {body_html}
        </main>
    </body>
    </html>
    """

    if not output:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", title.lower())[:40].strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(REPORTS_DIR / f"{safe_name}_{timestamp}.pdf")

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        raise RuntimeError(f"PDF generation failed with error code {pisa_status.err}")
    try:
        latest_path = REPORTS_DIR / "research_report.pdf"
        with open(latest_path, "wb") as f_latest:
            with open(path, "rb") as f_curr:
                f_latest.write(f_curr.read())
    except Exception:
        pass

    return str(path.resolve())
def router_node(state: MainState):
    manual_mode = state.get("mode")
    if manual_mode and manual_mode in ["closed_book", "hybrid", "open_book"]:
        needs_research = manual_mode != "closed_book"
        queries = [f"{state['topic']} overview", f"{state['topic']} key concepts", f"{state['topic']} recent developments"] if needs_research else []
        return {
            "needs_research": needs_research,
            "mode": manual_mode,
            "queries": queries
        }

    try:
        decision = router_llm.with_structured_output(RouterDecision, method="json_schema").invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content="Topic: " + state["topic"])
        ])
        return {
            "needs_research": decision.needs_research,
            "mode": decision.mode,
            "queries": decision.queries
        }
    except Exception as e:
        print(f"[Router Warning] Fallback to hybrid mode: {e}")
        return {
            "needs_research": True,
            "mode": "hybrid",
            "queries": [f"{state['topic']} core concepts", f"{state['topic']} applications"]
        }

def route_next(state: MainState):
    return "research" if state.get("needs_research", False) else "planning"

def research_node(state: MainState):
    if tavily_client is None:
        return {"evidence": []}

    results_by_url = {}
    queries = state.get("queries", [])[:3]

    for query in queries:
        if not query or not query.strip():
            continue
        try:
            response = tavily_client.search(
                query=query,
                max_results=3,
                search_depth="basic"
            )
            for item in response.get("results", []):
                url = item.get("url")
                if not url or url in results_by_url:
                    continue
                results_by_url[url] = EvidenceItem(
                    title=item.get("title") or "",
                    url=url,
                    published_at=item.get("published_date") or item.get("published_at"),
                    snippet=(item.get("content") or "")[:700],
                    source=item.get("source")
                )
        except Exception as error:
            print(f"[Research Warning] Search for '{query}' failed: {error}")

    return {"evidence": list(results_by_url.values())[:8]}

def planning_node(state: MainState):
    evidence = state.get("evidence") or []
    evidence_data = [
        item.model_dump() if isinstance(item, EvidenceItem) else item
        for item in evidence
    ]

    try:
        planner = planner_llm.with_structured_output(Plan, method="json_schema")
        plan = planner.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=(
                f"Topic: {state['topic']}\n"
                f"Research mode: {state.get('mode', 'closed_book')}\n"
                f"Research decision: {state.get('needs_research', False)}\n"
                f"Available evidence:\n{str(evidence_data[:8])}"
            ))
        ])
    except Exception as e:
        print(f"[Planner Warning] JSON Schema error, falling back to default plan structure: {e}")
        plan = Plan(
            title=f"Research Analysis: {state['topic']}",
            audience="College_student",
            tone="Academic",
            tasks=[
                Task(id=1, title="Introduction and Foundational Concepts", goal="Introduce core definitions, historical context, and motivation.", bullets=["Background and definitions", "Key terminology", "Significance in domain"], target_words=250),
                Task(id=2, title="Core Architecture and Mechanisms", goal="Detail technical principles, working methodologies, and structural frameworks.", bullets=["System architecture", "Underlying algorithms/processes", "Technical mechanics"], target_words=280),
                Task(id=3, title="Applications, Case Studies, and Impact", goal="Explore practical use cases, industrial implementations, and empirical outcomes.", bullets=["Real-world applications", "Case study review", "Comparative performance"], target_words=260),
                Task(id=4, title="Challenges, Limitations, and Future Directions", goal="Analyze current bottlenecks, open research challenges, and future trends.", bullets=["Current technical limitations", "Open research questions", "Emerging frontiers"], target_words=250),
            ]
        )
    if not plan.tasks:
        plan.tasks = [
            Task(id=1, title="Introduction and Foundational Concepts", goal="Introduce core definitions, historical context, and motivation.", bullets=["Background and definitions", "Key terminology", "Significance in domain"], target_words=250),
            Task(id=2, title="Core Architecture and Mechanisms", goal="Detail technical principles, working methodologies, and structural frameworks.", bullets=["System architecture", "Underlying algorithms/processes", "Technical mechanics"], target_words=280),
            Task(id=3, title="Applications, Case Studies, and Impact", goal="Explore practical use cases, industrial implementations, and empirical outcomes.", bullets=["Real-world applications", "Case study review", "Comparative performance"], target_words=260),
            Task(id=4, title="Challenges, Limitations, and Future Directions", goal="Analyze current bottlenecks, open research challenges, and future trends.", bullets=["Current technical limitations", "Open research questions", "Emerging frontiers"], target_words=250),
        ]

    return {"plan": plan}

def agent_request_sender(state: MainState):
    plan = state["plan"]
    evidence_dump = [
        item.model_dump() if isinstance(item, EvidenceItem) else item
        for item in state.get("evidence", [])
    ]
    return [
        Send(
            "Agents",
            {
                "topic": state["topic"],
                "mode": state.get("mode", "closed_book"),
                "plan": plan.model_dump(),
                "task": task.model_dump(),
                "evidence": evidence_dump
            }
        )
        for task in plan.tasks
    ]

def Agents_node(agent_call: dict) -> dict:
    task = Task(**agent_call["task"])
    plan = Plan(**agent_call["plan"])
    evidence = [EvidenceItem(**e) for e in agent_call["evidence"]]
    topic = agent_call["topic"]
    mode = agent_call.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets) if task.bullets else "\n- Detailed coverage of topic"
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.snippet or ''}"
            for e in evidence[:6]
        )

    prompt_content = (
        f"Document title: {plan.title}\n"
        f"Audience: {plan.audience}\n"
        f"Tone: {plan.tone}\n"
        f"Topic: {topic}\n"
        f"Mode: {mode}\n\n"
        f"Section title: {task.title}\n"
        f"Goal: {task.goal}\n"
        f"Target words: {task.target_words}\n"
        f"Bullets:{bullets_text}\n\n"
        f"Evidence (Use these sources when relevant):\n{evidence_text}\n"
    )

    section_md = writer_llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=prompt_content),
    ]).content.strip()

    return {"sections": [(task.id, section_md)]}

def merge_content_node(state: MainState):
    ordered = sorted(state["sections"], key=lambda x: x[0])
    merged = "\n\n".join(text for _, text in ordered)
    return {"merge_text": merged, "final": merged}

def pdf_node_func(state: MainState):
    plan = state["plan"]
    path = generate_pdf_report(
        content=state["final"],
        title=plan.title,
        subtitle=f"Comprehensive Academic Report on {state['topic']}",
        audience=plan.audience,
        tone=plan.tone,
        mode=state.get("mode") or "hybrid",
    )
    return {"pdf_path": path}
workflow = StateGraph(MainState)
workflow.add_node("router", router_node)
workflow.add_node("research", research_node)
workflow.add_node("planning", planning_node)
workflow.add_node("Agents", Agents_node)
workflow.add_node("merge", merge_content_node)
workflow.add_node("pdf", pdf_node_func)

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", route_next, {"research": "research", "planning": "planning"})
workflow.add_edge("research", "planning")
workflow.add_conditional_edges("planning", agent_request_sender, ["Agents"])
workflow.add_edge("Agents", "merge")
workflow.add_edge("merge", "pdf")
workflow.add_edge("pdf", END)

doc_agent_graph = workflow.compile()

jobs: Dict[str, Dict[str, Any]] = {}

def execute_agent_job(job_id: str, topic: str, audience: str, mode: str, tone: str):
    job = jobs[job_id]
    job["status"] = "running"
    job["current_step"] = "router"
    job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Started autonomous research for topic: '{topic}'")

    try:
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] 🧭 Routing Agent: Analyzing topic complexity...")
        init_state = {
            "topic": topic.strip(),
            "mode": mode if mode != "auto" else "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "merge_text": "",
            "final": "",
            "pdf_path": "",
        }

        r_out = router_node(init_state)
        init_state.update(r_out)
        job["mode"] = init_state["mode"]
        job["queries"] = init_state["queries"]
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Mode selected: '{init_state['mode']}' (Needs Research: {init_state['needs_research']})")

        if init_state["needs_research"]:
            job["current_step"] = "research"
            job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Tavily Agent: Executing web search queries...")
            res_out = research_node(init_state)
            init_state.update(res_out)
            job["evidence"] = [e.model_dump() if hasattr(e, "model_dump") else e for e in init_state["evidence"]]
            job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Retrieved {len(init_state['evidence'])} verified web sources.")
        else:
            job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Skipping web search (Closed Book mode).")

        job["current_step"] = "planning"
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] 📐 Planner Agent: Structuring academic outline...")
        p_out = planning_node(init_state)
        init_state.update(p_out)
        plan: Plan = init_state["plan"]
        job["plan"] = plan.model_dump()
        job["title"] = plan.title
        job["tasks_count"] = len(plan.tasks)
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Plan created: '{plan.title}' with {len(plan.tasks)} structured sections.")

        job["current_step"] = "writing"
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ✍️ Writer Agents: Spawning {len(plan.tasks)} parallel section agents...")
        agent_calls = agent_request_sender(init_state)
        
        sections = []
        for call in agent_calls:
            call_arg = call.arg if hasattr(call, "arg") else getattr(call, "args", {})
            sec_dict = Agents_node(call_arg)
            sections.extend(sec_dict["sections"])
            task_title = call_arg.get("task", {}).get("title", f"Section {len(sections)}")
            job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] Section completed: '{task_title}'")
        
        init_state["sections"] = sections
        job["current_step"] = "merging"
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] 📑 Assembly Node: Synthesizing sections...")
        m_out = merge_content_node(init_state)
        init_state.update(m_out)
        job["current_step"] = "pdf"
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] 🖨️ PDF Engine: Formatting layout & compiling PDF report...")
        pdf_out = pdf_node_func(init_state)
        init_state.update(pdf_out)

        pdf_path = Path(init_state["pdf_path"])
        job["status"] = "completed"
        job["current_step"] = "completed"
        job["pdf_path"] = str(pdf_path)
        job["pdf_filename"] = pdf_path.name
        job["pdf_url"] = f"/reports/{pdf_path.name}"
        job["final_markdown"] = init_state["final"]
        job["word_count"] = len(init_state["final"].split())
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 Document generation complete! PDF saved as '{pdf_path.name}'.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        job["status"] = "failed"
        job["error"] = str(e)
        job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}")
@app.get("/")
def serve_frontend():
    frontend_path = BASE_DIR / "frontend.html"
    if not frontend_path.exists():
        return HTMLResponse("<h2>frontend.html not found in workspace.</h2>", status_code=404)
    return FileResponse(str(frontend_path), media_type="text/html")

@app.get("/frontend.html")
def serve_frontend_alias():
    return serve_frontend()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "tavily_enabled": tavily_client is not None,
        "models": {
            "worker": "openai/gpt-oss-20b",
            "planner": "openai/gpt-oss-20b",
            "router": "qwen/qwen3.8-27b"
        },
        "reports_count": len(list(REPORTS_DIR.glob("*.pdf")))
    }

@app.get("/reports")
def get_reports_list():
    reports = []
    for f in sorted(REPORTS_DIR.glob("*.pdf"), key=os.path.getmtime, reverse=True):
        size_kb = round(f.stat().st_size / 1024, 1)
        reports.append({
            "filename": f.name,
            "size_kb": size_kb,
            "created_at": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %b %Y, %H:%M"),
            "url": f"/reports/{f.name}"
        })
    return {"reports": reports}

@app.get("/reports/{filename}")
def download_or_view_pdf(filename: str, download: bool = False):
    file_path = REPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Requested PDF report not found.")
    
    disposition = "attachment" if download else "inline"
    return FileResponse(
        str(file_path),
        media_type="application/pdf",
        content_disposition_type=disposition,
        filename=filename
    )

@app.post("/api/generate")
def start_generation_job(req: GenerateRequest, background_tasks: BackgroundTasks):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "topic": topic,
        "status": "queued",
        "current_step": "init",
        "logs": [],
        "created_at": datetime.now().isoformat()
    }

    background_tasks.add_task(
        execute_agent_job,
        job_id=job_id,
        topic=topic,
        audience=req.audience or "College_student",
        mode=req.mode or "auto",
        tone=req.tone or "Academic"
    )

    return {"job_id": job_id, "status": "queued"}

@app.get("/api/job/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs[job_id]

@app.post("/generate")
def generate_document_sync(req: GenerateRequest):
    """Direct synchronous endpoint for generating a document."""
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "topic": topic,
        "status": "queued",
        "current_step": "init",
        "logs": [],
        "created_at": datetime.now().isoformat()
    }
    execute_agent_job(job_id, topic, req.audience or "College_student", req.mode or "auto", req.tone or "Academic")
    return jobs[job_id]

def open_browser():
    url = "http://127.0.0.1:8000"
    print(f"\n[PaperCraft AI] Automatically launching browser at {url} ...\n")
    webbrowser.open(url)

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    print("=" * 64)
    print(" PaperCraft AI - Autonomous Research & Document Creation Agent")
    print(f" * Backend API:   http://{host}:{port}")
    print(f" * Frontend UI:   http://{host}:{port}/")
    print("=" * 64)

    threading.Timer(1.2, open_browser).start()
    uvicorn.run(app, host=host, port=port)
