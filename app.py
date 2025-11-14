# app.py — Fast résumé parsing + LLM-only cover letter (Ollama) + CSUSB listings
# Requirements you should have installed somewhere in your setup:
#   pip install streamlit requests bs4 lxml pandas pymupdf python-docx
# And make sure an Ollama server is running with a model pulled, e.g.:
#   ollama serve  (separate)  and  ollama pull llama3.2:3b

import os, re, io, hashlib, json
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# --------------------------- Config ---------------------------
CSUSB_URL = "https://www.csusb.edu/cse/internships-careers"
UA        = "Mozilla/5.0 (CSUSB Internship Assistant)"
MODEL     = os.getenv("MODEL_NAME", "llama3.2:3b")   # try qwen2:0.5b or llama3.2:3b
OLLAMA    = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
NUM_CTX   = int(os.getenv("NUM_CTX", "1024"))        # small ctx for speed
NUM_PRED  = int(os.getenv("NUM_PREDICT", "220"))
TEMP      = float(os.getenv("TEMPERATURE", "0.2"))

# --------------------------- Lexicons (for quick skill/role scan) ---------------------------
SKILL_LEXICON = {
    "python","java","c++","c#","javascript","typescript","go","rust","kotlin","swift","r","sql",
    "react","angular","vue","node","express","django","flask","fastapi","spring","spring boot",".net",
    "pandas","numpy","pytorch","tensorflow","sklearn","spark","hadoop","tableau","power bi",
    "selenium","cypress","playwright","pytest","junit","postman",
    "aws","azure","gcp","docker","kubernetes","terraform","linux","bash","git",
    "mysql","postgresql","mongodb","redis"
}
ROLE_LEXICON = {"software","data","ml","ai","security","cloud","devops","qa","sre","web","backend","frontend","mobile"}

# --------------------------- Fast résumé extract & parse (cached) ---------------------------
# Uses PyMuPDF (pymupdf) for fast PDF text; for DOCX uses python-docx.
import fitz  # PyMuPDF

@st.cache_data(show_spinner=False)
def extract_resume_fast(file_bytes: bytes, filename: str) -> dict:
    """Parse résumé once (cached by SHA256): name/email/phone/skills/roles + text snippet."""
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    name = (filename or "").lower()
    text = ""

    try:
        if name.endswith(".pdf"):
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                pages = min(2, doc.page_count)  # first 1–2 pages is usually enough (speed)
                text = "\n".join(doc[i].get_text("text") or "" for i in range(pages))
        elif name.endswith(".docx"):
            from docx import Document
            bio = io.BytesIO(file_bytes)
            doc = Document(bio)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            text = file_bytes.decode("utf-8", "ignore")
    except Exception:
        text = file_bytes.decode("utf-8", "ignore")

    # Contacts
    email = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text or "")
    phone = re.search(r'(\+?\d[\d\-\s\(\)]{7,}\d)', text or "")
    email = email.group(0) if email else ""
    phone = phone.group(0) if phone else ""

    # First non-empty non-header line as name guess
    guessed_name = ""
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s: continue
        if email and email in s: continue
        if phone and phone in s: continue
        if re.search(r"resume|curriculum vitae|contact", s, re.I): continue
        guessed_name = s[:80]
        break

    toks = re.findall(r"[A-Za-z][A-Za-z0-9\+\.\-#]{1,}", (text or "").lower())
    skills = sorted({t for t in toks if t in SKILL_LEXICON})[:10]   # cap to keep prompts small
    roles  = sorted({t for t in toks if t in ROLE_LEXICON})[:4]

    return {
        "hash": file_hash,
        "text": text[:6000],   # truncate for speed
        "name": guessed_name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "roles": roles,
    }

# --------------------------- Direct Ollama call (fast) ---------------------------
def ollama_generate(prompt: str, system: str = "", model: str = None,
                    num_ctx: int = NUM_CTX, num_predict: int = NUM_PRED, temperature: float = TEMP, timeout=45) -> str:
    """Direct /api/generate call to Ollama (no LangChain)."""
    base = OLLAMA
    model = model or MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system or "",
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature}
    }
    r = requests.post(f"{base}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

# --------------------------- CSUSB internship scraper ---------------------------
BAD_LAST = {"careers","career","jobs","job","students","graduates","early-careers"}
JUNK_KEYWORDS = {
    "proposal form","evaluation form","student evaluation","supervisor evaluation",
    "report form","handbook","resume","cv","scholarship","scholarships","grant program",
    "career center","advising","policy","forms","pdf"
}

def _clean(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "")).strip()

def _path_is_specific(path: str) -> bool:
    p = (path or "/").lower()
    if "intern" in p or "co-op" in p: return True
    seg = [s for s in p.split("/") if s]
    if any(re.search(r"\d{5,}", s) for s in seg): return True
    if seg and seg[-1] in BAD_LAST: return False
    return len(seg) >= 3

def _is_intern_link(text, url) -> bool:
    low = f"{text} {url}".lower()
    if any(k in low for k in JUNK_KEYWORDS): return False
    if not ("intern" in low or "co-op" in low): return False
    try:
        return _path_is_specific(urlparse(url).path)
    except Exception:
        return False

@st.cache_data(show_spinner=False, ttl=3600)
def scrape_csusb() -> pd.DataFrame:
    try:
        r = requests.get(CSUSB_URL, headers={"User-Agent": UA}, timeout=20)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()
    soup = BeautifulSoup(r.text, "lxml")
    main = soup.find("main") or soup
    rows, seen = [], set()
    for a in main.find_all("a", href=True):
        t = _clean(a.get_text(" ", strip=True))
        if not t: continue
        absu = urljoin(CSUSB_URL, a["href"])
        k = (t.lower(), absu)
        if k in seen: continue
        if not _is_intern_link(t, absu): continue
        host = urlparse(absu).netloc.lower()
        comp = host.split(".")[-2].capitalize() if host else ""
        rows.append({"title": t, "company": comp, "link": absu})
        seen.add(k)
    return pd.DataFrame(rows)

# --------------------------- Job page helpers (cached) ---------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_job_text(url: str, limit: int = 5000) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code != 200: return ""
        s = BeautifulSoup(r.text, "lxml")
        main = s.find("main") or s.find("article") or s.find("section") or s
        return (main.get_text(" ", strip=True) or "")[:limit]
    except Exception:
        return ""

def infer_company_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc
        parts = [p for p in host.split(".") if p not in {"www","jobs","careers","boards","gh","lever","greenhouse"}]
        if not parts: return ""
        return parts[-2].capitalize() if len(parts) >= 2 else parts[-1].capitalize()
    except Exception:
        return ""

def infer_role_from_text(txt: str) -> str:
    m = re.search(r"(?i)\b(software|data|machine learning|ml|ai|security|cloud|devops).{0,40}intern", txt or "")
    if m:
        start = max(0, m.start()-30); end = min(len(txt), m.end()+30)
        seg = re.sub(r"\s+", " ", (txt or "")[start:end])
        mm = re.search(r"([A-Z][A-Za-z0-9\-\s]{2,60}Intern)", seg)
        if mm: return mm.group(1).strip()
    m2 = re.search(r"(?i)(?:title|position)\s*[:\-]\s*([^\n\r]+)", txt or "")
    return (m2.group(1).strip() if m2 else "")

# --------------------------- LLM-only cover letter (fast) ---------------------------
def draft_cover_letter(company: str, role: str, job_url: str, job_text: str, who: dict, profile: dict) -> str:
    """
    LLM-only cover letter (no hardcoded fallback).
    - Uses ONLY facts from résumé parsing + job snippet.
    - 140–180 words, 3 short paragraphs.
    - No placeholders, no invented facts.
    """
    nm = (who.get("name") or "").strip()
    # Title-case to avoid ALL CAPS names while preserving Mixed Case tokens
    name = " ".join([w.capitalize() if w.isupper() or w.islower() else w for w in nm.split()]) or "Candidate"
    company = (company or "").strip() or "your team"
    email = (who.get("email") or "").strip()
    phone = (who.get("phone") or "").strip()
    linkedin = (who.get("linkedin") or "").strip()

    roles  = ", ".join((profile.get("roles") or [])[:4])
    skills = ", ".join((profile.get("skills") or [])[:10])

    snippet = re.sub(r"\s+", " ", (job_text or ""))[:800]

    prompt = f"""
You are an expert career writer. Produce a SHORT, specific, truthful cover letter grounded ONLY in the facts below.
If a detail is missing, omit it—never invent it.

FACTS:
- Name: {name}
- Email: {email or '—'}
- Phone: {phone or '—'}
- LinkedIn: {linkedin or '—'}
- Roles: {roles or '—'}
- Skills: {skills or '—'}
- Company: {company}
- Role: {role or 'Intern'}
- Job URL: {job_url or '—'}
- Job snippet: {snippet or '—'}

STRICT RULES:
- 140–180 words, 3 short paragraphs.
- First sentence must name {company} and the {role or 'intern'} role.
- Tie 3–4 skills to the snippet, ONLY from the facts above.
- Do NOT mention years of experience or past employers unless present in facts.
- No placeholders like [Company] or [Your Name].
- Close with availability and contact (email/LinkedIn if present).
- Start with "Dear {company}" or "Dear Hiring Team at {company}".
- End with a signature block including the real candidate name: {name}.

Return ONLY the letter text.
"""
    system = "Follow the rules exactly. Be concise, warm, and honest. Do not invent facts."
    letter = ollama_generate(prompt, system, MODEL, NUM_CTX, NUM_PRED, TEMP)
    # light cleanup
    return re.sub(r"\[[^\]]*\]", "", letter).strip() if letter else ""

# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title="CSUSB Internships + Cover Letter (fast)", page_icon="💼", layout="wide")
st.title("💼 CSUSB Internships + ✉️ Cover Letter (fast, LLM-only)")

# Session state
for k, v in {
    "mode": "greet",
    "wizard_step": 0,
    "resume_parsed": {},
    "job_url": "",
    "company": "",
    "role": "",
    "name": "",
    "email": "",
    "phone": "",
    "linkedin": "",
    "role_hint": "",
    "skills_hint": "",
}.items():
    st.session_state.setdefault(k, v)

# -------- Simple Q&A (greet) --------
if st.session_state["mode"] == "greet":
    st.subheader("Ask me something")
    user = st.text_input("Try: 'hi', 'what is this', 'show internships', or 'cover letter' 👇")

    if user:
        t = user.strip().lower()
        if re.search(r'^(hi|hello|hey)\b', t):
            st.markdown("**Assistant:** Hi! I can list internships and write a cover letter from a job link using your résumé.")
            if st.button("Start Cover Letter Wizard ✍️"):
                st.session_state["mode"] = "cover_wizard"
                st.session_state["wizard_step"] = 0
                st.rerun()
        elif "what is this" in t or "about" in t:
            st.markdown("**Assistant:** I fetch internship links from the CSUSB CSE page and create tailored cover letters using your résumé.")
        elif re.search(r'\bcover\s*letter\b', t):
            st.session_state["mode"] = "cover_wizard"; st.session_state["wizard_step"] = 0; st.rerun()
        elif re.search(r'\b(show\s+internships?|list\s+internships?|internships?)\b', t):
            st.session_state["mode"] = "list"; st.rerun()
        else:
            # quick single-turn LLM answer (optional)
            try:
                ans = ollama_generate(user, "Answer concisely in 2–5 sentences.")
            except Exception:
                ans = "I'm here! Try 'show internships' or 'cover letter'."
            st.markdown("**Assistant:** " + (ans or "Try 'show internships' or 'cover letter'."))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Internships 🔎"):
            st.session_state["mode"] = "list"; st.rerun()
    with col2:
        if st.button("Cover Letter Wizard ✍️"):
            st.session_state["mode"] = "cover_wizard"; st.session_state["wizard_step"] = 0; st.rerun()

# -------- Show internships --------
if st.session_state["mode"] == "list":
    st.subheader("🔎 Internships from CSUSB – Internships & Careers")
    with st.spinner("Fetching internships…"):
        df = scrape_csusb()
    if df.empty:
        st.warning("No internship postings found on the CSUSB page right now.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
    if st.button("Back ↩️"):
        st.session_state["mode"] = "greet"; st.rerun()
    st.markdown("---")
    st.caption("Want a cover letter for a specific job? Go back and pick **Cover Letter Wizard**.")

# -------- Cover letter wizard (resume → autofill → letter) --------
if st.session_state["mode"] == "cover_wizard":
    st.subheader("✉️ Cover Letter Wizard (uses your résumé)")
    step = st.session_state["wizard_step"]

    def next_step():
        st.session_state["wizard_step"] += 1
        st.rerun()

    def prev_step():
        st.session_state["wizard_step"] = max(0, st.session_state["wizard_step"] - 1)
        st.rerun()

    # Step 0: Upload résumé (cached parse)
    if step == 0:
        st.write("**Step 1/4 — Upload your résumé** (PDF/DOCX/TXT). I’ll extract your name, email, phone, and skills.")
        new_file = st.file_uploader("Upload résumé", type=["pdf","docx","txt"], key="resume_upload")
        if new_file is not None:
            data = new_file.getvalue()
            parsed = extract_resume_fast(data, new_file.name)
            st.session_state["resume_parsed"] = parsed
            # prefill UI fields
            st.session_state["name"]  = parsed.get("name","")
            st.session_state["email"] = parsed.get("email","")
            st.session_state["phone"] = parsed.get("phone","")
            st.session_state["role_hint"]  = ", ".join(parsed.get("roles", []))
            st.session_state["skills_hint"]= ", ".join(parsed.get("skills", []))
            st.success("Résumé parsed (cached).")
        cols = st.columns(2)
        if cols[0].button("Next ➡️", disabled=not st.session_state["resume_parsed"]):
            next_step()
        if cols[1].button("Back ↩️"):
            st.session_state["mode"] = "greet"; st.rerun()

    # Step 1: Job link
    elif step == 1:
        st.write("**Step 2/4 — Paste the job posting link** (Lever/Greenhouse/company careers).")
        st.session_state["job_url"] = st.text_input("Job URL (required)", value=st.session_state["job_url"])
        cols = st.columns(3)
        if cols[0].button("⬅️ Previous"):
            prev_step()
        if cols[2].button("Next ➡️", disabled=not bool(st.session_state["job_url"].strip())):
            next_step()

    # Step 2: Review auto-filled info (you can edit)
    elif step == 2:
        st.write("**Step 3/4 — Review your info** (auto-filled; you may edit).")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["name"]  = st.text_input("Your name", value=st.session_state["name"])
            st.session_state["email"] = st.text_input("Email", value=st.session_state["email"])
            st.session_state["phone"] = st.text_input("Phone", value=st.session_state["phone"])
        with c2:
            st.session_state["linkedin"]   = st.text_input("LinkedIn URL", value=st.session_state["linkedin"])
            st.session_state["role_hint"]  = st.text_input("Desired role(s)", value=st.session_state["role_hint"])
            st.session_state["skills_hint"]= st.text_input("Top skills (comma-separated)", value=st.session_state["skills_hint"])
        cols = st.columns(3)
        if cols[0].button("⬅️ Previous"):
            prev_step()
        if cols[2].button("Next ➡️"):
            next_step()

    # Step 3: Generate & show letter
    elif step == 3:
        st.write("**Step 4/4 — Generate Cover Letter**")

        # Fetch job text and infer company/role
        job_url = st.session_state["job_url"]
        job_text = fetch_job_text(job_url)
        company  = infer_company_from_url(job_url)
        role     = infer_role_from_text(job_text)

        # Build profile from parsed résumé + user edits
        parsed = st.session_state.get("resume_parsed", {})
        roles  = parsed.get("roles", [])[:4]
        skills = parsed.get("skills", [])[:10]
        if st.session_state["role_hint"]:
            for t in re.split(r"[,/; ]+", st.session_state["role_hint"].lower()):
                t=t.strip()
                if t and t not in roles: roles.append(t)
        if st.session_state["skills_hint"]:
            for t in re.split(r"[,/; ]+", st.session_state["skills_hint"].lower()):
                t=t.strip()
                if t and t not in skills: skills.append(t)
        profile = {"roles": roles, "skills": skills}

        who = {
            "name": st.session_state["name"],
            "email": st.session_state["email"],
            "phone": st.session_state["phone"],
            "linkedin": st.session_state["linkedin"]
        }

        # Generate (LLM-only)
        with st.form("gen_cover"):
            st.write("Ready to generate?")
            submitted = st.form_submit_button("Generate Cover Letter ✍️")
            if submitted:
                try:
                    letter = draft_cover_letter(company, role, job_url, job_text, who, profile)
                except Exception as e:
                    letter = ""
                    st.error(f"LLM call failed: {e}")

                if not letter:
                    st.error("The LLM returned an empty response. Ensure Ollama is running and the model is available (e.g., `ollama pull llama3.2:3b`).")
                else:
                    st.markdown("### Your Cover Letter")
                    st.text_area("Preview", value=letter, height=280)
                    fname = f"cover_letter_{re.sub(r'[^a-z0-9]+','_', (company or 'company').lower())}_" \
                            f"{re.sub(r'[^a-z0-9]+','_', (role or 'intern').lower())}.txt"
                    st.download_button("⬇️ Download Cover Letter", data=letter.encode("utf-8"),
                                       file_name=fname, mime="text/plain")

        cols = st.columns(3)
        if cols[0].button("⬅️ Start Over"):
            st.session_state["wizard_step"] = 0; st.rerun()
        if cols[2].button("Back to Home"):
            st.session_state["mode"] = "greet"; st.rerun()
