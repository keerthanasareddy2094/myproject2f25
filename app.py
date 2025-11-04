import os, re, io, requests, pandas as pd, streamlit as st, nest_asyncio, asyncio
from urllib.parse import urljoin, urlparse
from datetime import datetime
from bs4 import BeautifulSoup
from auto_apply import auto_apply_batch

nest_asyncio.apply()

# -------------------- Config --------------------
CSUSB_URL = "https://www.csusb.edu/cse/internships-careers"
UA        = "Mozilla/5.0 (CSUSB Internship Assistant)"
OLLAMA    = os.getenv("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
MODEL     = os.getenv("MODEL_NAME", "qwen2:0.5b")
SYS       = os.getenv("SYSTEM_PROMPT", "You are a helpful, concise assistant.")
MAX_TOK   = int(os.getenv("MAX_TOKENS", "256"))
NUM_CTX   = int(os.getenv("NUM_CTX", "2048"))

# -------------------- Utils --------------------
BAD_LAST = {"careers","career","jobs","job","students","graduates","early-careers"}
JUNK_KEYWORDS = {
    "proposal form","evaluation form","student evaluation","supervisor evaluation",
    "report form","handbook","resume","cv","scholarship","scholarships","grant program",
    "career center","advising","policy","forms","pdf"
}

def _clean(s:str)->str:
    return re.sub(r"\s+"," ", (s or "")).strip()

def _path_is_specific(path:str)->bool:
    p = (path or "/").lower()
    if "intern" in p or "co-op" in p: return True
    seg = [s for s in p.split("/") if s]
    if any(re.search(r"\d{5,}", s) for s in seg): return True
    if seg and seg[-1] in BAD_LAST: return False
    return len(seg) >= 3

def _is_intern_link(text, url)->bool:
    low = f"{text} {url}".lower()
    if any(k in low for k in JUNK_KEYWORDS): return False
    if not ("intern" in low or "co-op" in low): return False
    try: return _path_is_specific(urlparse(url).path)
    except Exception: return False

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]","",(s or "").lower())

def scrape_csusb() -> pd.DataFrame:
    r = requests.get(CSUSB_URL, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
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
        rows.append({"title": t, "company": comp, "link": absu, "host": host,
                     "posted": datetime.utcnow().date().isoformat(),
                     "blob": _canon(f"{t} {comp} {host} {absu}")})
        seen.add(k)
    return pd.DataFrame(rows)

# -------------------- LLM (optional via Ollama) --------------------
def llm_answer(prompt: str, sys_msg: str = None) -> str:
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate
        llm = ChatOllama(base_url=OLLAMA, model=MODEL, temperature=0.2,
                         model_kwargs={"num_ctx": NUM_CTX, "num_predict": MAX_TOK})
        tmpl = ChatPromptTemplate.from_messages([("system", sys_msg or SYS), ("human", "{q}")])
        return (tmpl | llm).invoke({"q": prompt}).content.strip()
    except Exception:
        return ""

# -------------------- Résumé parsing --------------------
def extract_text_from_upload(upload):
    if not upload: return ""
    name = (upload.name or "").lower()
    try:
        if name.endswith(".pdf"):
            import pdfplumber
            txt = ""
            with pdfplumber.open(upload) as pdf:
                for p in pdf.pages:
                    txt += (p.extract_text() or "") + "\n"
            return txt
        elif name.endswith(".docx"):
            from docx import Document
            bio = io.BytesIO(upload.read())
            doc = Document(bio)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return upload.read().decode("utf-8", "ignore")
    except Exception:
        return ""

SKILL_LEXICON = {
    "python","java","c++","c#","javascript","typescript","go","rust","kotlin","swift","r","sql",
    "react","angular","vue","node","express","django","flask","fastapi","spring","spring boot",".net",
    "pandas","numpy","pytorch","tensorflow","sklearn","spark","hadoop","tableau","power bi",
    "selenium","cypress","playwright","pytest","junit","postman",
    "aws","azure","gcp","docker","kubernetes","terraform","linux","bash","git",
    "mysql","postgresql","mongodb","redis"
}
ROLE_LEXICON = {"software","data","ml","ai","security","cloud","devops","qa","sre","web","backend","frontend","mobile"}

def extract_profile(resume_text: str, role_hint: str, skills_hint: str):
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\+\.\-#]{1,}", (resume_text or "").lower())
    skills = sorted({t for t in toks if t in SKILL_LEXICON})
    roles  = sorted({t for t in toks if t in ROLE_LEXICON})
    if role_hint:
        for t in re.split(r"[,/; ]+", role_hint.lower()):
            t=t.strip()
            if t and t not in roles: roles.append(t)
    if skills_hint:
        for t in re.split(r"[,/; ]+", skills_hint.lower()):
            t=t.strip()
            if t and t not in skills: skills.append(t)
    exp_years = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years|yrs)", (resume_text or "").lower())
    if m:
        try: exp_years = float(m.group(1))
        except: pass
    return {"roles": roles, "skills": skills, "exp_years": exp_years}

# -------------------- Matching & scoring --------------------
def score_posting(row, profile, company_pref, location, remote):
    score = 0
    blob = row.get("blob","")
    for r in profile["roles"]:
        if _canon(r) in blob: score += 3
    for s in profile["skills"]:
        if _canon(s) in blob: score += 2
    if company_pref and _canon(company_pref) in blob: score += 4
    if location and _canon(location) in blob: score += 1
    if remote and remote.lower()!="any" and _canon(remote) in blob: score += 1
    return score

def filter_and_rank(df, profile, company, location, remote):
    if df.empty: return df
    df = df.copy()
    df["match_score"] = df.apply(lambda r: score_posting(r, profile, company, location, remote), axis=1)
    df = df.sort_values(["match_score","posted"], ascending=[False, False])
    good = df[df["match_score"] >= 3]
    return good.head(15) if not good.empty else df.head(15)

# -------------------- Application Kit generation --------------------
def build_application_kit(posting, profile, who):
    prompt = f"""
You are an internship application assistant. Create succinct, professional materials.

POSTING:
- Title: {posting.get('title','')}
- Company: {posting.get('company','')}
- URL: {posting.get('link','')}

CANDIDATE:
- Name: {who.get('name') or '—'}
- Email: {who.get('email') or '—'}
- Phone: {who.get('phone') or '—'}
- LinkedIn: {who.get('linkedin') or '—'}
- Roles: {', '.join(profile.get('roles') or []) or '—'}
- Skills: {', '.join(profile.get('skills') or []) or '—'}
- Experience (years): {profile.get('exp_years') or '—'}

Deliver exactly these sections:
1) Email Subject (one line)
2) Email Body (6–10 concise lines; include the posting title, link, and contact details)
3) Cover Note (one tight paragraph 90–130 words)
4) ATS Keywords (comma-separated, 12–20 items)
5) Resume Bullets (3–5 quantified bullets)
"""
    out = llm_answer(prompt, sys_msg="Be specific, accurate, concise, professional.") or \
          "(LLM unavailable) Draft your own email, short cover, ATS keywords, and 3 bullets using top skills."
    txt = f"""=== APPLICATION KIT ===
Company: {posting.get('company','')}
Role: {posting.get('title','')}
Link: {posting.get('link','')}

{out}
"""
    return out, txt

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="CSUSB Internship Assistant", page_icon="💬", layout="wide")
st.title("💬 CSUSB Internship Assistant")
st.caption("Greet → general chat • say “internships” to list • say “apply for internships” to personalize, upload résumé, generate Application Kits, and Auto-Apply (Greenhouse/Lever).")

# Session state
for k, v in {
    "mode": "greet",
    "applicant_info": None,
    "profile": None,
    "ranked_rows": None,
}.items():
    if k not in st.session_state: st.session_state[k] = v

# 1) Greeting / Router
if st.session_state["mode"] == "greet":
    st.markdown("**Hi! How can I help you today?**")
    user = st.text_input("Type here (e.g., 'Show internships', 'Apply for internships', or any question) 👇")
    if not user: st.stop()
    txt = user.lower()
    if re.search(r"\b(apply|find|match)\b.*\b(intern)", txt):
        st.session_state["mode"] = "apply"; st.rerun()
    elif re.search(r"\b(intern|internship|show internships|list internships)\b", txt):
        st.session_state["mode"] = "list"; st.rerun()
    else:
        ans = llm_answer(user) or "(LLM unavailable — enable/pair Ollama for richer answers.)"
        st.markdown("**Assistant:** " + ans)
        st.stop()

# 2) List internships (CSUSB)
if st.session_state["mode"] == "list":
    st.subheader("🔎 Internships from CSUSB – Internships & Careers")
    with st.spinner("Fetching internships…"):
        df = scrape_csusb()
    if df.empty:
        st.warning("No internship postings found on the CSUSB page right now.")
    else:
        st.dataframe(df[["title","company","link"]], use_container_width=True, hide_index=True)
    if st.button("Back ↩️"):
        st.session_state["mode"]="greet"; st.rerun()

# 3) Apply flow (form → results → kits → auto-apply)
if st.session_state["mode"] == "apply":
    st.subheader("🧭 Personalized Internship Finder & Application")

    with st.form("apply_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            name  = st.text_input("Your name")
            email = st.text_input("Email")
            phone = st.text_input("Phone")
        with c2:
            linkedin = st.text_input("LinkedIn URL")
            location = st.text_input("Preferred location (optional)")
            remote   = st.selectbox("Remote type", ["Any","Remote","Hybrid","Onsite"], index=0)

        role     = st.text_input("Desired role(s) (e.g., software, data, security)")
        company  = st.text_input("Preferred company (optional)")
        skills   = st.text_input("Top skills (comma-separated)")
        resume   = st.file_uploader("Upload résumé (PDF, DOCX, or TXT)", type=["pdf","docx","txt"])

        submitted = st.form_submit_button("Find Internships 🔍")

    if submitted:
        resume_text = extract_text_from_upload(resume)
        profile = extract_profile(resume_text, role, skills)
        with st.spinner("Searching CSUSB postings…"):
            df = scrape_csusb()
        ranked = filter_and_rank(df, profile, company, location, remote) if not df.empty else pd.DataFrame()
        st.session_state["applicant_info"] = {"name":name,"email":email,"phone":phone,"linkedin":linkedin}
        st.session_state["profile"] = profile
        st.session_state["ranked_rows"] = [] if ranked.empty else ranked.to_dict("records")
        st.success("Results ready below. Select postings and click **Generate Application Kit**.")

    ranked_rows = st.session_state["ranked_rows"]
    if ranked_rows:
        st.subheader("🎯 Best Matches")
        st.caption("Select the internships you want to apply to, then click **Generate Application Kit**.")
        for i, r in enumerate(ranked_rows, start=1):
            with st.expander(f"{i}. {r['title']} — {r.get('company','')}"):
                st.write(f"[Open posting / Apply]({r['link']})")
                st.checkbox("Select this internship", key=f"pick_{i}", value=False)

        # Generate kits
        if st.button("Generate Application Kit"):
            who = st.session_state.get("applicant_info") or {}
            prof = st.session_state.get("profile") or {"roles":[],"skills":[],"exp_years":None}
            picked = [r for i, r in enumerate(ranked_rows, start=1) if st.session_state.get(f"pick_{i}", False)]
            if not picked:
                st.warning("Select at least one internship above.")
            else:
                for r in picked:
                    kit_md, kit_txt = build_application_kit(
                        {"title":r.get("title"), "company":r.get("company"), "link":r.get("link")}, prof, who
                    )
                    st.markdown(f"**Application Kit – {r.get('title')} @ {r.get('company','')}**")
                    st.markdown(kit_md or "_(Enable Ollama for richer kits.)_")
                    st.download_button(
                        label=f"⬇️ Download Kit ({r.get('company','')[:20]} – {r.get('title','')[:30]})",
                        data=kit_txt.encode("utf-8"),
                        file_name=f"application_kit_{_canon(r.get('company',''))}_{_canon(r.get('title',''))}.txt",
                        mime="text/plain"
                    )

        # Auto-Apply (Greenhouse / Lever)
        st.markdown("---")
        st.subheader("⚡ Auto-Apply (beta) — Greenhouse & Lever only")
        st.caption("Fills common fields (name, email, phone, résumé, cover letter). Other portals will require manual apply.")
        resume_for_auto = st.file_uploader("Upload résumé for auto-apply (PDF preferred)", type=["pdf"], key="resume_for_auto")

        if st.button("Auto-apply to selected"):
            picked = [r for i, r in enumerate(ranked_rows, start=1) if st.session_state.get(f"pick_{i}", False)]
            if not picked:
                st.warning("Select at least one internship above.")
                st.stop()
            if not resume_for_auto:
                st.warning("Please upload your resume (PDF) for auto-apply.")
                st.stop()

            who = st.session_state.get("applicant_info") or {}
            prof = st.session_state.get("profile") or {"roles":[],"skills":[],"exp_years":None}
            cover = llm_answer(
                f"Write a concise 6–8 line cover letter for internships. Roles={prof.get('roles')}, "
                f"skills={prof.get('skills')}. Mention eagerness to learn and link to LinkedIn: {who.get('linkedin','')}.",
                sys_msg="Professional, succinct tone."
            ) or (
                f"Dear Hiring Team,\n\n"
                f"I am excited to apply for internship roles. My background includes {', '.join(prof.get('skills')[:6])} "
                f"and interests in {', '.join(prof.get('roles')[:3])}. I am eager to contribute and learn.\n"
                f"LinkedIn: {who.get('linkedin','')}\n\n"
                f"Thank you for your consideration.\n"
                f"{who.get('name','')}\n{who.get('email','')}\n{who.get('phone','')}"
            )

            with st.spinner("Submitting applications…"):
                resume_bytes = resume_for_auto.read()
                results = asyncio.run(auto_apply_batch(picked, who, resume_bytes, cover))

            st.success("Auto-apply finished. Review statuses below.")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.markdown(
                "Legend: **submitted_or_attempted** = form filled & submit click attempted • "
                "**manual_required** = unsupported portal; open link and apply manually."
            )

    if st.button("Back ↩️"):
        for k in ("applicant_info","profile","ranked_rows"):
            st.session_state[k]=None
        st.session_state["mode"]="greet"; st.rerun()
