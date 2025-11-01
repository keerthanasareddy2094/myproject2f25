import os, re, json, time
from pathlib import Path
from typing import Dict, List

import streamlit as st
import pandas as pd

# === Config ===
APP_TITLE = "LLM Internship Assistant — Interview + Deep Search"
DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME  = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
MAX_QUESTIONS = 10

st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
if Path("styles.css").exists():
    st.markdown(Path("styles.css").read_text(encoding="utf-8"), unsafe_allow_html=True)

st.title(APP_TITLE)
st.caption("Phase 1: LLM asks up to 10 questions. Phase 2: deep-scrape CSUSB + company career pages for matches.")

# ==== Import your deep scraper (required) ====
try:
    from scraper import (
        scrape_csusb_listings,         # must support deep=True
        quick_company_links_playwright, # must support deep=True
        CSUSB_CSE_URL
    )
except Exception as e:
    scrape_csusb_listings = None
    quick_company_links_playwright = None
    CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"
    st.error("Deep scraper not found. Make sure your deep scraper.py is in the same folder.")
    st.stop()

# ==== Ollama plumbing ====
@st.cache_resource(show_spinner=False)
def _ollama_ok() -> bool:
    import urllib.request, json as _j
    try:
        with urllib.request.urlopen(OLLAMA_HOST.rstrip("/") + "/api/tags", timeout=2) as r:
            if r.status != 200: return False
            data = _j.loads(r.read().decode() or "{}")
            models = [m.get("name","") for m in data.get("models",[])]
            return any(str(n).startswith(MODEL_NAME) for n in models)
    except Exception:
        return False

def _ollama_chat(msgs: List[Dict], num_predict=200, temp=0.2, timeout_s: float = 15.0) -> str:
    """Small wrapper around /api/chat with hard timeout. Returns '' on errors."""
    try:
        import httpx
        payload = {
            "model": MODEL_NAME,
            "messages": msgs,
            "stream": False,
            "options": {"num_ctx": 4096, "num_predict": num_predict, "temperature": temp},
        }
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(OLLAMA_HOST.rstrip("/") + "/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return ((data.get("message") or {}).get("content") or "").strip()
    except Exception:
        return ""

HAVE_LLM = _ollama_ok()

# ==== Resume helpers (LLM preferred, has fallback) ====
from io import BytesIO
from pypdf import PdfReader
from docx import Document

def _read_pdf(b: bytes, max_pages: int = 12) -> str:
    out=[]
    try:
        pdf=PdfReader(BytesIO(b))
        for p in list(pdf.pages)[:max_pages]:
            try: out.append(p.extract_text() or "")
            except Exception: pass
    except Exception:
        pass
    return "\n".join(out)

def _read_docx(b: bytes) -> str:
    try:
        doc = Document(BytesIO(b))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def _resume_text(upload) -> str:
    name = (upload.name or "").lower()
    b = upload.getvalue()
    if name.endswith(".pdf"): return _read_pdf(b)
    if name.endswith(".docx"): return _read_docx(b)
    try: return b.decode("utf-8","ignore")
    except Exception: return b.decode("latin-1","ignore")

def _fallback_resume_json(text: str) -> Dict:
    import re
    d: Dict = {}
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    phone = re.search(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})", text or "")
    linkedin = re.search(r"(https?://)?(www\.)?linkedin\.com/[A-Za-z0-9_/\-]+", text or "", re.I)
    github = re.search(r"(https?://)?(www\.)?github\.com/[A-Za-z0-9_\-]+", text or "", re.I)
    name = ""
    for ln in (text.splitlines()[:8]):
        s = ln.strip()
        if s and "@" not in s and len(s) < 60 and not re.search(r"(objective|summary|resume|curriculum vitae)", s, re.I):
            name = s; break
    d.update({
        "name": name,
        "email": email.group(0) if email else "",
        "phone": phone.group(0) if phone else "",
        "links": {
            "linkedin": linkedin.group(0) if linkedin else "",
            "github": github.group(0) if github else "",
            "portfolio": "",
            "other": []
        },
        "skills": [],
        "education": [],
        "experience": []
    })
    return d

def _llm_resume_json(text: str) -> Dict:
    if not text.strip(): return {}
    if not HAVE_LLM: return _fallback_resume_json(text)
    sys = (
        "You are a resume parser. Return ONLY compact JSON (no prose): "
        '{"name":"","email":"","phone":"","links":{"linkedin":"","github":"","portfolio":"","other":[]},'
        '"skills":[],"education":[],"experience":[]}'
    )
    out = _ollama_chat(
        [{"role":"system","content":sys},{"role":"user","content":text[:16000]}],
        num_predict=360, temp=0.1, timeout_s=15.0
    )
    if out:
        m = re.search(r"\{[\s\S]*\}", out)
        try:
            parsed = json.loads(m.group(0) if m else out)
            if isinstance(parsed, dict): return parsed
        except Exception:
            pass
    return _fallback_resume_json(text)

# ==== Phase 1: LLM-only interview ====
def next_question(hist: List[Dict], remaining: int) -> str:
    """Return exactly ONE short question from the LLM (no manual list)."""
    if not HAVE_LLM:  # absolute minimal fallback (won't be used if HAVE_LLM)
        return "Tell me your preferred internship role."
    sys = (
        "You are interviewing a student to recommend internships. "
        "Ask SHORT, specific questions (one at a time), up to 10 total. "
        "Cover: roles, companies, location, skills, work mode (remote/hybrid/onsite), "
        "timeline, work authorization/visa, industries, project/team preferences, final notes. "
        "Return ONE question only. No prefaces."
    )
    convo = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in hist])
    resp = _ollama_chat(
        [{"role":"system","content":sys},
         {"role":"user", "content": f"Questions left: {remaining}\n{convo}\nNext question:"}],
        num_predict=80, temp=0.2, timeout_s=10.0
    )
    # take the first non-empty line
    for ln in (resp or "").splitlines():
        t = ln.strip(" -•:").strip()
        if t:
            return t[:220]
    return "What internship role are you most interested in?"

def summarize_profile(p: Dict) -> str:
    def bullets(xs): return "".join([f"\n- {x}" for x in xs]) if xs else " —"
    return "\n\n".join([
        f"**Name:** {p.get('name') or '—'}",
        f"**Roles:**{bullets(p.get('roles'))}",
        f"**Companies:**{bullets(p.get('companies'))}",
        f"**Locations:**{bullets(p.get('locations'))}",
        f"**Skills:**{bullets(p.get('skills'))}",
        f"**Work Mode:** {p.get('work_mode') or '—'}",
        f"**Timeline:** {p.get('timeline') or '—'}",
        f"**Authorization:** {p.get('auth') or '—'}",
        f"**Industries:**{bullets(p.get('industries'))}",
        f"**Notes:** {p.get('notes') or '—'}"
    ])

def normalize_answer(q: str, a: str, prof: Dict):
    s=a.strip()
    def split(x): return [i.strip() for i in re.split(r"[,;/\n]+", x) if i.strip()]
    ql=q.lower()
    if "role" in ql or "position" in ql:
        prof["roles"] = list(dict.fromkeys(prof.get("roles",[]) + split(s)))
    elif "comp" in ql or "company" in ql:
        prof["companies"] = list(dict.fromkeys(prof.get("companies",[]) + split(s)))
    elif "where" in ql or "location" in ql:
        prof["locations"] = list(dict.fromkeys(prof.get("locations",[]) + split(s)))
    elif "skill" in ql:
        prof["skills"] = list(dict.fromkeys(prof.get("skills",[]) + [x.lower() for x in split(s)]))
    elif "remote" in ql or "onsite" in ql or "hybrid" in ql or "work mode" in ql:
        prof["work_mode"] = s
    elif "when" in ql or "timeline" in ql:
        prof["timeline"] = s
    elif "author" in ql or "visa" in ql:
        prof["auth"] = s
    elif "industry" in ql:
        prof["industries"] = list(dict.fromkeys(prof.get("industries",[]) + split(s)))
    else:
        prof["notes"] = (prof.get("notes","") + " " + s).strip()

# ==== State ====
st.session_state.setdefault("hist", [])  # list of {q,a}
st.session_state.setdefault("q_i", 0)
st.session_state.setdefault("done", False)
st.session_state.setdefault("profile", {
    "name":"", "roles":[], "companies":[], "locations":[], "skills":[],
    "work_mode":"", "timeline":"", "auth":"", "industries":[], "notes":"", "resume":{}
})

# ==== Sidebar: résumé upload (LLM parsing) ====
with st.sidebar:
    st.subheader("Progress")
    st.progress(min(st.session_state.q_i, MAX_QUESTIONS)/MAX_QUESTIONS)
    st.caption(f"{st.session_state.q_i}/{MAX_QUESTIONS} questions")
    st.markdown("---")
    st.subheader("Upload Résumé")
    up = st.file_uploader("PDF/DOCX/TXT", type=["pdf","docx","txt"], label_visibility="collapsed")
    if up is not None:
        with st.spinner("Parsing résumé with LLM…"):
            text = _resume_text(up)
            js = _llm_resume_json(text)
            st.session_state.profile["resume"] = js
            # fill name/skills if helpful
            if not st.session_state.profile.get("name") and js.get("name"):
                st.session_state.profile["name"] = js["name"]
            if not st.session_state.profile.get("skills") and js.get("skills"):
                st.session_state.profile["skills"] = [str(s).lower() for s in js["skills"][:20] if str(s).strip()]
        (DATA_DIR/"resume.txt").write_text(text, encoding="utf-8")
        (DATA_DIR/"resume.json").write_text(json.dumps(js, indent=2), encoding="utf-8")
        st.success("Résumé uploaded and parsed ✅")
        st.rerun()

# ==== Phase switch ====
if st.session_state["done"]:
    st.success("✅ Phase 1 complete — here’s your profile")
    st.info(summarize_profile(st.session_state.profile))

    st.markdown("---")
    st.subheader("🔎 Phase 2 — Deep Internship Search")
    st.caption(f"Sources: CSUSB CSE page + company career sites (deep scrape)")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        deep_csusb = st.toggle("Deep scrape CSUSB links", value=True, help="Visit each career link from CSUSB page to find internship postings.")
    with colB:
        deep_companies = st.toggle("Deep scrape companies from answers", value=True, help="Visit company career sites from your answers.")
    with colC:
        max_pages = st.slider("Max pages/company", 10, 100, 40, 10)

    run = st.button("Run Deep Search")
    if run:
        prof = st.session_state.profile
        out_rows: List[Dict] = []

        with st.spinner("Scraping CSUSB page…"):
            try:
                df_csusb = scrape_csusb_listings(deep=deep_csusb, max_pages=max_pages)
            except TypeError:
                # older signature
                df_csusb = scrape_csusb_listings()
            except Exception as e:
                df_csusb = pd.DataFrame()
                st.error(f"CSUSB scrape error: {e}")

        # Normalize columns
        cols = ["title","company","location","posted_date","salary","education","remote","host","link","source","details"]
        if not df_csusb.empty:
            for c in cols:
                if c not in df_csusb.columns:
                    df_csusb[c] = None
            df_csusb["blob"] = (
                df_csusb["title"].fillna("").str.lower() + " " +
                df_csusb["company"].fillna("").str.lower() + " " +
                df_csusb["details"].fillna("").str.lower() + " " +
                df_csusb["link"].fillna("").str.lower()
            )
        else:
            df_csusb = pd.DataFrame(columns=cols + ["blob"])

        # Filter CSUSB by profile
        blob = df_csusb["blob"].fillna("")
        def contains_any(series, toks):
            if not toks: return pd.Series([True]*len(series))
            m = pd.Series([False]*len(series))
            for t in toks:
                m |= series.str.contains(re.escape(t), na=False)
            return m

        roles = [r.lower() for r in prof.get("roles", []) if r]
        skills= [s.lower() for s in prof.get("skills", []) if s]
        comps = [c.lower() for c in prof.get("companies", []) if c]
        locs  = [l.lower() for l in prof.get("locations", []) if l]

        csusb_mask = pd.Series([True]*len(df_csusb))
        if comps:  csusb_mask &= contains_any(blob, comps)
        if roles:  csusb_mask &= contains_any(blob, roles + ["intern","internship"])
        if skills: csusb_mask &= contains_any(blob, skills)
        if locs:   csusb_mask &= contains_any(blob, locs + ["remote","hybrid","onsite","on-site"])

        df_a = df_csusb[csusb_mask].copy()

        # Deep scrape company career sites from answers
        df_b_list = []
        if deep_companies and prof.get("companies"):
            with st.spinner("Scraping company career sites from your answers…"):
                for comp in prof["companies"]:
                    try:
                        df_comp = quick_company_links_playwright(comp, deep=True)
                        if df_comp is not None and not df_comp.empty:
                            df_b_list.append(df_comp)
                    except Exception as e:
                        st.warning(f"{comp}: {e}")
        df_b = pd.concat(df_b_list, ignore_index=True) if df_b_list else pd.DataFrame(columns=cols)

        # Combine + dedupe
        combined = pd.concat([df_a[cols], df_b[cols]], ignore_index=True)
        if not combined.empty:
            combined = combined.drop_duplicates(subset=["link"]).reset_index(drop=True)

        st.markdown(f"**Results:** {len(combined)} posting(s)")
        if combined.empty:
            st.info("No postings matched your preferences. Try different roles/companies or increase max pages.")
        else:
            # Links block
            def links_md(df: pd.DataFrame) -> str:
                out=[]
                for i, r in enumerate(df.itertuples(index=False), start=1):
                    t = (getattr(r,"title","") or "Internship").strip()
                    c = (getattr(r,"company","") or "").strip()
                    link = (getattr(r,"link","") or "").strip()
                    label = f"**{t}**" + (f" — {c}" if c else "")
                    out.append(f"{i}. {label}  \n   🔗 [Open Posting]({link})")
                return "\n\n".join(out)

            st.markdown(links_md(combined))
            st.dataframe(
                combined,
                use_container_width=True,
                hide_index=True,
                column_config={"link": st.column_config.LinkColumn("Open")},
            )
            st.download_button(
                "📥 Download CSV",
                combined.to_csv(index=False).encode(),
                file_name="deep_internships.csv",
                mime="text/csv"
            )

    st.markdown("---")
    if st.button("🔁 Restart Interview"):
        st.session_state["hist"] = []
        st.session_state["q_i"] = 0
        st.session_state["done"] = False
        st.session_state["profile"] = {k: [] if isinstance(v, list) else "" for k, v in st.session_state["profile"].items()}
        st.session_state["profile"].update({"resume":{}})
        st.rerun()

else:
    # ===== Phase 1: LLM-only interview loop =====
    remain = MAX_QUESTIONS - st.session_state["q_i"]
    if remain <= 0:
        st.session_state["done"] = True
        st.experimental_rerun()

    q = next_question(st.session_state["hist"], remain)
    st.subheader(f"Question {st.session_state['q_i']+1} of {MAX_QUESTIONS}")
    st.write(q)

    ans = st.text_input("Your answer", key=f"ans_{st.session_state['q_i']}")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Skip"):
            st.session_state["hist"].append({"q": q, "a": "(skipped)"})
            st.session_state["q_i"] += 1
            st.rerun()
    with c2:
        if st.button("Next"):
            if not ans.strip():
                st.warning("Please type an answer, or click Skip.")
            else:
                st.session_state["hist"].append({"q": q, "a": ans})
                normalize_answer(q, ans, st.session_state["profile"])
                # populate name from resume if helpful
                if (not st.session_state["profile"].get("name")) and st.session_state["profile"].get("resume",{}).get("name"):
                    st.session_state["profile"]["name"] = st.session_state["profile"]["resume"]["name"]
                st.session_state["q_i"] += 1
                st.rerun()

    st.markdown("---")
    st.caption("Live summary (so far):")
    st.info(summarize_profile(st.session_state["profile"]))
