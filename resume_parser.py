# resume_parser.py
import json, os, re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader
from docx import Document

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from resume_manager import read_file_to_text


DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:0.5b")


def _normalize_resume_json(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out = dict(data)

    # ensure links sub-dict exists
    links = out.get("links") or {}
    if not isinstance(links, dict):
        links = {}
    # merge common website/portfolio synonyms found in the raw text if present later
    out["links"] = {
        "linkedin": links.get("linkedin", "") or "",
        "github": links.get("github", "") or "",
        "portfolio": links.get("portfolio", "") or "",
        "other": links.get("other", []) or []
    }

    # lower, de-dup skills
    skills = out.get("skills") or []
    if isinstance(skills, list):
        seen = set()
        norm = []
        for s in skills:
            t = (s or "").strip().lower()
            if t and t not in seen:
                seen.add(t)
                norm.append(t)
        out["skills"] = norm

    # trim long strings
    for k, v in list(out.items()):
        if isinstance(v, str) and len(v) > 1000:
            out[k] = v[:1000]

    return out


# ---------- File -> text ----------
def _read_pdf(b: bytes) -> str:
    out = []
    reader = PdfReader(BytesIO(b))
    try:
        if reader.is_encrypted:
            reader.decrypt("")  # try empty password for campus PDFs
    except Exception:
        pass
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(out)


def _read_docx(b: bytes) -> str:
    buf = BytesIO(b)
    doc = Document(buf)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_resume_text(uploaded_file) -> str:
    return read_file_to_text(uploaded_file)

    
    

# ---------- LLM extraction ----------
def llm_resume_extract(resume_text: str) -> Dict[str, Any]:
    """
    Uses a single {resume_text} variable and escapes all braces in the prompt
    so ChatPromptTemplate never sees stray placeholders.
    """
    if not (resume_text or "").strip():
        return {}

    system = (
        "You extract structured résumé data and output compact JSON only. "
        "Follow this strict schema (omit null/empty fields). "
        "{{{{"
        '  "name": "string",'
        '  "email": "string",'
        '  "phone": "string",'
        '  "links": {{"linkedin": "url", "github": "url", "portfolio": "url", "other": ["url", ...] }},'
        '  "summary": "1-2 sentences",'
        '  "skills": ["token", ...],'
        '  "education": [{{"school":"", "degree":"", "field":"", "start":"", "end":"", "gpa":""}}],'
        '  "experience": [{{"company":"", "title":"","start":"","end":"","location":"","bullets":["..."]}}],'
        '  "projects": [{{"name":"", "tech":["..."], "summary":""}}],'
        '  "certifications": ["..."]'
        "}}}}"
        "Return strictly minified JSON. Do not include any commentary."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "RESUME TEXT:\n{resume_text}\n\nReturn JSON now.")
    ])

    llm = ChatOllama(
        base_url=OLLAMA_HOST,
        model=MODEL_NAME,
        temperature=0.1,
        streaming=False,
        model_kwargs={"num_ctx": 4096, "num_predict": 350}
    )

    # Truncate to keep it snappy
    text = (resume_text or "").strip()
    if len(text) > 12000:
        text = text[:12000]

    try:
        response = (prompt | llm).invoke({"resume_text": text})
        out = response.content or "{}"
    except Exception:
        out = "{}"

    # Extract the first JSON object from the reply
    m = re.search(r"\{[\s\S]*\}", out)
    json_str = m.group(0) if m else out

    data: Dict[str, Any] = {}
    try:
        data = json.loads(json_str)
    except Exception:
        data = {}

    # ---- SAFE defaults (no unbound 'links') ----
    if not isinstance(data, dict):
        data = {}
    links: Dict[str, Any] = data.get("links") if isinstance(data.get("links"), dict) else {}
    links = {
        "linkedin": links.get("linkedin", "") or "",
        "github": links.get("github", "") or "",
        "portfolio": links.get("portfolio", "") or "",
        "other": links.get("other", []) or [],
    }

    # Regex fallbacks
    email_rx = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    phone_rx = re.search(r"(\+?\d{1,2}\s*)?(\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4})", text or "")
    linkedin_rx = re.search(r"(https?://)?(www\.)?linkedin\.com/[A-Za-z0-9_/\-]+", text or "", re.I)
    github_rx = re.search(r"(https?://)?(www\.)?github\.com/[A-Za-z0-9_\-]+", text or "", re.I)

    # apply fallbacks if missing
    data.setdefault("email", email_rx.group(0) if email_rx else "")
    data.setdefault("phone", phone_rx.group(0) if phone_rx else "")
    if linkedin_rx and not links.get("linkedin"):
        links["linkedin"] = linkedin_rx.group(0)
    if github_rx and not links.get("github"):
        links["github"] = github_rx.group(0)

    # try a generic website for portfolio if still empty
    if not links.get("portfolio"):
        generic_site = re.search(r"(https?://[^\s]+)", text or "")
        if generic_site:
            url = generic_site.group(0)
            if "linkedin.com" not in url and "github.com" not in url:
                links["portfolio"] = url

    data["links"] = links

    # crude name heuristic
    if not data.get("name"):
        for line in (text.splitlines()[:8]):
            l = (line or "").strip()
            if l and "@" not in l and len(l) <= 60 and not re.search(r"(objective|summary|resume|curriculum vitae)", l, re.I):
                data["name"] = l
                break

    # normalize skills to lower + unique
    skills = data.get("skills") or []
    if isinstance(skills, list):
        seen = set()
        norm = []
        for s in skills:
            t = (s or "").strip().lower()
            if t and t not in seen:
                seen.add(t)
                norm.append(t)
        data["skills"] = norm

    return data



def save_resume(data: Dict[str, Any], resume_text: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "resume.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (DATA_DIR / "resume.txt").write_text(resume_text, encoding="utf-8")

# ---------- Answering from stored résumé ----------
def answer_from_resume(question: str, data: Dict[str, Any]) -> str:
    q = (question or "").lower()

    def bullets(items: List[str]) -> str:
        return "\n".join([f"- {i}" for i in items])

    if "name" in q:
        v = data.get("name") or "Not found"
        return f"**Name:** {v}"
    if "email" in q:
        v = data.get("email") or "Not found"
        return f"**Email:** {v}"
    if "phone" in q or "mobile" in q:
        v = data.get("phone") or "Not found"
        return f"**Phone:** {v}"
    if "linkedin" in q:
        v = (data.get("links") or {}).get("linkedin") or "Not found"
        return f"**LinkedIn:** {v}"
    if "github" in q:
        v = (data.get("links") or {}).get("github") or "Not found"
        return f"**GitHub:** {v}"
    if "portfolio" in q or "website" in q:
        v = (data.get("links") or {}).get("portfolio") or "Not found"
        return f"**Portfolio:** {v}"
    if "skill" in q:
        skills = data.get("skills") or []
        return "**Skills**\n\n" + (bullets(skills) if skills else "_None captured_")
    if "education" in q or "school" in q or "degree" in q:
        edu = data.get("education") or []
        if not edu: return "_No education entries captured_"
        lines = []
        for e in edu:
            parts = [e.get("degree"), e.get("field"), e.get("school")]
            when = " - ".join([e.get("start") or "", e.get("end") or ""]).strip(" -")
            if when: parts.append(f"({when})")
            if e.get("gpa"): parts.append(f"GPA {e['gpa']}")
            lines.append(" • ".join([p for p in parts if p]))
        return "**Education**\n\n" + bullets(lines)
    if "project" in q:
        projs = data.get("projects") or []
        if not projs: return "_No projects captured_"
        lines = []
        for p in projs[:5]:
            tech = ", ".join(p.get("tech") or [])
            s = f"{p.get('name','Project')} — {p.get('summary','')}"
            if tech: s += f"  \n   _Tech_: {tech}"
            lines.append(s)
        return "**Projects**\n\n" + "\n\n".join([f"- {x}" for x in lines])
    if "experience" in q or "work" in q or "employment" in q:
        ex = data.get("experience") or []
        if not ex: return "_No experience captured_"
        lines = []
        for e in ex[:5]:
            when = " - ".join([e.get("start") or "", e.get("end") or ""]).strip(" -")
            head = " • ".join([v for v in [e.get("title"), e.get("company"), when, e.get("location")] if v])
            bullets_ = "\n".join([f"   - {b}" for b in (e.get("bullets") or [])[:4]])
            lines.append(head + ("\n" + bullets_ if bullets_ else ""))
        return "**Experience**\n\n" + "\n\n".join(lines)

    # default summary
    name = data.get("name", "Candidate")
    skills = ", ".join(data.get("skills")[:10] or [])
    summary = data.get("summary") or ""
    return f"**Résumé on file for {name}.**\n\n{summary}\n\n**Key skills:** {skills if skills else '_n/a_'}"
