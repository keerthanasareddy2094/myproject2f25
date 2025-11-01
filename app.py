import os, re, json, time
from pathlib import Path
from typing import Dict, List
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document

# ========= CONFIG =========
APP_TITLE = "LLM Internship Assistant — Phase 1 + Phase 2"
DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME  = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
MAX_Q = 10
st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")
if Path("styles.css").exists():
    st.markdown(Path("styles.css").read_text(), unsafe_allow_html=True)
st.title(APP_TITLE)

# ========= IMPORT SCRAPER =========
try:
    from scraper import scrape_csusb_listings, quick_company_links_playwright, CSUSB_CSE_URL
except Exception as e:
    st.error(f"Deep scraper not found: {e}")
    st.stop()

# ========= OLLAMA HELPERS =========
@st.cache_resource
def have_llm() -> bool:
    import urllib.request, json as j
    try:
        with urllib.request.urlopen(OLLAMA_HOST.rstrip("/")+"/api/tags", timeout=2) as r:
            d = j.loads(r.read().decode() or "{}")
            names=[m.get("name","") for m in d.get("models",[])]
            return any(MODEL_NAME in n for n in names)
    except Exception: return False

def llm_chat(msgs, temp=0.2, npred=160, timeout_s=12):
    try:
        import httpx
        payload={"model":MODEL_NAME,"messages":msgs,"stream":False,
                 "options":{"num_ctx":2048,"num_predict":npred,"temperature":temp}}
        with httpx.Client(timeout=timeout_s) as c:
            r=c.post(OLLAMA_HOST.rstrip("/")+"/api/chat",json=payload)
            r.raise_for_status()
            return ((r.json().get("message") or {}).get("content") or "").strip()
    except Exception: return ""

# ========= RESUME UTILITIES =========
def _read_pdf(b:bytes)->str:
    out=[]; pdf=PdfReader(os.BytesIO(b))
    for p in pdf.pages[:10]:
        try: out.append(p.extract_text() or "")
        except: pass
    return "\n".join(out)
def _read_docx(b:bytes)->str:
    doc=Document(os.BytesIO(b)); return "\n".join(p.text for p in doc.paragraphs)
def resume_text(f):
    n=f.name.lower(); b=f.getvalue()
    if n.endswith(".pdf"): return _read_pdf(b)
    if n.endswith(".docx"): return _read_docx(b)
    try: return b.decode("utf-8","ignore")
    except: return b.decode("latin-1","ignore")

def resume_json_llm(t:str)->Dict:
    if not have_llm(): return {}
    sys=("You are a resume parser. Return ONLY compact JSON: "
         '{"name":"","email":"","phone":"","skills":[],"education":[],"experience":[]}')
    out=llm_chat([{"role":"system","content":sys},{"role":"user","content":t[:15000]}],npred=300)
    m=re.search(r"\{[\s\S]*\}",out)
    try: return json.loads(m.group(0) if m else out)
    except: return {}

# ========= INTERVIEW HELPERS =========
def next_q(hist:List[Dict],rem:int)->str:
    sys=("Ask one short, precise question to learn a student’s internship goals. "
         "You have at most 10 questions total. Cover roles, companies, skills, location, work mode, "
         "timeline, visa, industries, and any final note. Return only the question text.")
    conv="\n".join([f"Q:{h['q']}\nA:{h['a']}" for h in hist])
    out=llm_chat([{"role":"system","content":sys},
                  {"role":"user","content":f\"Questions left:{rem}\n{conv}\nNext question:\"}],npred=80)
    for l in out.splitlines():
        l=l.strip(" -•:"); 
        if l: return l[:220]
    return "What internship roles interest you?"

def norm(q,a,p):
    s=a.strip(); ql=q.lower()
    def spl(x): return [i.strip() for i in re.split(r"[,;/\n]+",x) if i.strip()]
    if "role" in ql: p["roles"]=spl(s)
    elif "comp" in ql: p["companies"]=spl(s)
    elif "loc" in ql or "where" in ql: p["locations"]=spl(s)
    elif "skill" in ql: p["skills"]=spl(s)
    elif "mode" in ql or "remote" in ql or "hybrid" in ql: p["work_mode"]=s
    elif "when" in ql or "timeline" in ql: p["timeline"]=s
    elif "visa" in ql or "auth" in ql: p["auth"]=s
    elif "industr" in ql: p["industries"]=spl(s)
    else: p["notes"]=p.get("notes","")+" "+s

def summary(p): 
    def bl(l): return "".join([f"\n- {i}" for i in l]) if l else "—"
    return "\n\n".join([
        f"**Name:** {p.get('name','—')}",
        f"**Roles:**{bl(p.get('roles'))}",
        f"**Companies:**{bl(p.get('companies'))}",
        f"**Skills:**{bl(p.get('skills'))}",
        f"**Locations:**{bl(p.get('locations'))}",
        f"**Timeline:** {p.get('timeline','—')}",
        f"**Work Mode:** {p.get('work_mode','—')}",
        f"**Authorization:** {p.get('auth','—')}",
        f"**Industries:**{bl(p.get('industries'))}",
        f"**Notes:** {p.get('notes','—')}"
    ])

# ========= STATE =========
st.session_state.setdefault("hist",[])
st.session_state.setdefault("i",0)
st.session_state.setdefault("done",False)
st.session_state.setdefault("profile",
    {"name":"","roles":[],"companies":[],"skills":[],"locations":[],
     "work_mode":"","timeline":"","auth":"","industries":[],"notes":""})

# ========= SIDEBAR =========
with st.sidebar:
    st.subheader("Progress")
    st.progress(min(st.session_state.i,MAX_Q)/MAX_Q)
    st.caption(f"{st.session_state.i}/{MAX_Q} questions")
    st.markdown("---")
    st.subheader("Upload Résumé")
    up=st.file_uploader("PDF/DOCX/TXT",type=["pdf","docx","txt"],label_visibility="collapsed")
    if up:
        with st.spinner("Parsing résumé..."):
            t=resume_text(up)
            j=resume_json_llm(t)
            if j.get("name"): st.session_state.profile["name"]=j["name"]
            if j.get("skills"): st.session_state.profile["skills"]=j["skills"]
        st.success("Résumé uploaded ✅")
        st.rerun()

# ========= MAIN =========
if st.session_state.done:
    st.success("✅ Phase 1 complete")
    st.info(summary(st.session_state.profile))

    st.markdown("---")
    st.subheader("Phase 2 – Deep Internship Search")
    deep_csusb = st.toggle("Deep scrape CSUSB links", True)
    deep_company = st.toggle("Deep scrape companies from your answers", True)
    max_pages = st.slider("Max pages/company",10,100,40,10)
    if st.button("Run Deep Search"):
        with st.spinner("Scraping..."):
            df_csusb = scrape_csusb_listings(deep=deep_csusb, max_pages=max_pages)
            df_list=[df_csusb]
            if deep_company:
                for c in st.session_state.profile.get("companies",[]):
                    try:
                        df_list.append(quick_company_links_playwright(c, deep=True))
                    except Exception as e:
                        st.warning(f"{c}: {e}")
            df=pd.concat([d for d in df_list if d is not None and not d.empty],ignore_index=True)
        if df.empty: st.info("No results found.")
        else:
            st.write(f"**{len(df)} internships found**")
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.download_button("📥 Download CSV",
                df.to_csv(index=False).encode(),"internships.csv","text/csv")

    if st.button("🔁 Restart Interview"):
        st.session_state.clear(); st.rerun()

else:
    remain=MAX_Q-st.session_state.i
    if remain<=0: st.session_state.done=True; st.rerun()
    q=next_q(st.session_state.hist,remain)
    st.subheader(f"Question {st.session_state.i+1} of {MAX_Q}")
    st.write(q)
    ans=st.text_input("Your answer",key=f"ans{st.session_state.i}")
    c1,c2=st.columns(2)
    with c1:
        if st.button("Skip"):
            st.session_state.hist.append({"q":q,"a":"(skipped)"})
            st.session_state.i+=1; st.rerun()
    with c2:
        if st.button("Next"):
            if not ans.strip(): st.warning("Type an answer or click Skip.")
            else:
                st.session_state.hist.append({"q":q,"a":ans})
                norm(q,ans,st.session_state.profile)
                st.session_state.i+=1; st.rerun()
    st.markdown("---")
    st.caption("Profile so far:")
    st.info(summary(st.session_state.profile))
