import os, json, re, io
from pathlib import Path
from typing import Dict, List

import streamlit as st
from pypdf import PdfReader
from docx import Document

# ===== CONFIG =====
APP_TITLE = "Internship Onboarding – Phase 1"
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True, parents=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME  = os.getenv("MODEL_NAME", "qwen2:0.5b")
MAX_QUESTIONS = 10

st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
if Path("styles.css").exists():
    st.markdown(Path("styles.css").read_text(), unsafe_allow_html=True)
st.title(APP_TITLE)
st.caption("We’ll ask up to 10 short questions to understand your goals and background.")

# ===== LLM HELPERS =====
@st.cache_resource(show_spinner=False)
def _ollama_ok() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(OLLAMA_HOST.rstrip("/") + "/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

def _ollama_chat(msgs: List[Dict], num_predict=160, temp=0.2) -> str:
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST)
        out = client.chat(
            model=MODEL_NAME,
            messages=msgs,
            stream=False,
            options={"num_ctx":2048,"num_predict":num_predict,"temperature":temp}
        )
        return (out.get("message",{}) or {}).get("content","")
    except Exception:
        return ""

# ===== RESUME UTILITIES =====
def _read_pdf(b: bytes) -> str:
    text=[]
    try:
        pdf=PdfReader(io.BytesIO(b))
        for p in pdf.pages: text.append(p.extract_text() or "")
    except: pass
    return "\n".join(text)

def _read_docx(b: bytes) -> str:
    try:
        doc=Document(io.BytesIO(b))
        return "\n".join(p.text for p in doc.paragraphs)
    except: return ""

def _resume_text(file) -> str:
    name=file.name.lower(); b=file.getvalue()
    if name.endswith(".pdf"): return _read_pdf(b)
    if name.endswith(".docx"): return _read_docx(b)
    try: return b.decode("utf-8","ignore")
    except: return b.decode("latin-1","ignore")

def _llm_resume_json(text:str)->Dict:
    if not text.strip(): return {}
    sys=("You are a résumé parser. Return ONLY compact JSON with keys:"
         '{"name":"","email":"","phone":"","skills":[],"education":[],"experience":[]}')
    out=_ollama_chat([{"role":"system","content":sys},
                      {"role":"user","content":text[:8000]}],num_predict=300,temp=0.1)
    m=re.search(r"\{[\s\S]*\}",out)
    try: return json.loads(m.group(0) if m else out)
    except: return {}

# ===== QUESTION LOGIC =====
BASE_Q=[
 "What internship roles interest you?",
 "Which companies do you admire or want to work for?",
 "Where would you like to work (city, state, or remote)?",
 "What are your top technical skills?",
 "Do you prefer remote, hybrid, or onsite work?",
 "When are you looking for internships (e.g., Summer 2026)?",
 "What is your work authorization or visa status?",
 "Which industries excite you the most?",
 "Any preferred project or team types?",
 "Anything else we should know about your goals?"
]

def _llm_next_q(hist:List[Dict],remain:int)->str:
    sys=("You ask at most 10 short questions to understand a student’s internship goals. "
         "Cover roles, companies, location, skills, work mode, timeline, visa, industries, and final comments. "
         "Return one concise question only.")
    convo="\n".join([f"Q:{h['q']}\nA:{h['a']}" for h in hist])
    out=_ollama_chat([
        {"role":"system","content":sys},
        {"role":"user","content":f"Questions left:{remain}\n{convo}\nNext question:"}
    ],num_predict=80,temp=0.2)
    for l in out.splitlines():
        l=l.strip(" -•")
        if l: return l[:220]
    return ""

def _summarize(p:Dict)->str:
    def bl(l): return "".join([f"\n- {i}" for i in l]) if l else " _n/a_"
    txt=[
      f"**Name:** {p.get('name') or '—'}",
      f"**Roles:**{bl(p.get('roles'))}",
      f"**Companies:**{bl(p.get('companies'))}",
      f"**Locations:**{bl(p.get('locations'))}",
      f"**Skills:**{bl(p.get('skills'))}",
      f"**Timeline:** {p.get('timeline') or '—'}",
      f"**Work Mode:** {p.get('work_mode') or '—'}",
      f"**Authorization:** {p.get('auth') or '—'}",
      f"**Industries:**{bl(p.get('industries'))}",
      f"**Notes:** {p.get('notes') or '—'}"
    ]
    return "\n\n".join(txt)

def _normalize(q,a,p):
    s=a.strip(); ql=q.lower()
    def split(x): return [i.strip() for i in re.split(r"[,;/\n]+",x) if i.strip()]
    if "role" in ql: p["roles"]=split(s)
    elif "comp" in ql: p["companies"]=split(s)
    elif "where" in ql or "location" in ql: p["locations"]=split(s)
    elif "skill" in ql: p["skills"]=[x.lower() for x in split(s)]
    elif "timeline" in ql or "when" in ql: p["timeline"]=s
    elif "remote" in ql or "onsite" in ql or "hybrid" in ql: p["work_mode"]=s
    elif "author" in ql or "visa" in ql: p["auth"]=s
    elif "industry" in ql: p["industries"]=split(s)
    else: p["notes"]=p.get("notes","")+" "+s

# ===== STATE =====
st.session_state.setdefault("have_llm",_ollama_ok())
st.session_state.setdefault("q_i",0)
st.session_state.setdefault("hist",[])
st.session_state.setdefault("profile",{
 "name":"","roles":[],"companies":[],"locations":[],
 "skills":[],"timeline":"","work_mode":"","auth":"",
 "industries":[],"notes":"","resume":{}
})
st.session_state.setdefault("done",False)

# ===== SIDEBAR – RESUME =====
with st.sidebar:
    st.subheader("Progress")
    st.progress(min(st.session_state.q_i,MAX_QUESTIONS)/MAX_QUESTIONS)
    st.caption(f"{st.session_state.q_i}/{MAX_QUESTIONS} questions")
    st.markdown("---")
    st.subheader("Upload Résumé")
    up=st.file_uploader("PDF/DOCX/TXT",type=["pdf","docx","txt"],label_visibility="collapsed")
    if up:
        txt=_resume_text(up)
        js=_llm_resume_json(txt) if st.session_state.have_llm else {}
        st.session_state.profile["resume"]=js
        (DATA_DIR/"resume.txt").write_text(txt)
        (DATA_DIR/"resume.json").write_text(json.dumps(js,indent=2))
        st.success("Résumé uploaded ✅")
        st.rerun()

# ===== MAIN FLOW =====
if st.session_state.done:
    st.success("✅ Interview complete – here’s your profile.")
    prof=st.session_state.profile
    st.markdown(_summarize(prof))
    st.download_button("📥 Download Profile JSON",
        data=json.dumps(prof,indent=2).encode(),
        file_name="student_profile.json",
        mime="application/json")
    if st.button("🔁 Restart"):
        for k in ["q_i","hist","done"]:
            st.session_state[k]=0 if k=="q_i" else ([] if k=="hist" else False)
        st.session_state.profile={k:v for k,v in st.session_state.profile.items()}
        st.session_state.profile.update({"resume":{}})
        st.rerun()
else:
    remain=MAX_QUESTIONS-st.session_state.q_i
    if remain<=0:
        st.session_state.done=True; st.rerun()
    q=_llm_next_q(st.session_state.hist,remain) if st.session_state.have_llm else ""
    if not q: q=BASE_Q[min(st.session_state.q_i,len(BASE_Q)-1)]
    st.subheader(f"Question {st.session_state.q_i+1} of {MAX_QUESTIONS}")
    st.write(q)
    ans=st.text_input("Your answer",key=f"a{st.session_state.q_i}")
    c1,c2=st.columns([1,1])
    with c1:
        if st.button("Skip"):
            st.session_state.hist.append({"q":q,"a":"(skipped)"})
            st.session_state.q_i+=1; st.rerun()
    with c2:
        if st.button("Next"):
            if not ans.strip(): st.warning("Please enter an answer or skip.")
            else:
                st.session_state.hist.append({"q":q,"a":ans})
                _normalize(q,ans,st.session_state.profile)
                st.session_state.q_i+=1; st.rerun()
    st.markdown("---")
    st.caption("Live summary (so far):")
    st.info(_summarize(st.session_state.profile))
