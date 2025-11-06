# app.py — CSUSB Internship Assistant (deep-scrape + apply)
# Features:
#  - Greeting + simple Q&A (works without LLM; uses LLM if available)
#  - Deep-scrape: CSUSB page -> follow out to career platforms -> collect INTERNSHIP POSTINGS (not just portals)
#  - Filters by company keywords (e.g., "amazon internships")
#  - "Apply for internships" → interview → resume upload → profile extraction → ranking → Application Kits → Auto-Apply (Greenhouse/Lever)
#
# NOTES:
#  - Deep-scrape is conservative and capped to avoid heavy crawling. It targets common career platforms.
#  - Auto-Apply supports Greenhouse & Lever (best-effort form fill). Others marked "manual_required".

import os, re, io, json, time, asyncio, urllib.request, requests, pandas as pd, streamlit as st, nest_asyncio
from datetime import datetime
from urllib.parse import urljoin, urlparse
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

PLATFORMS = {
    "greenhouse": "greenhouse.io",
    "lever": "lever.co",
    "workday": "workday",
    "smartrecruiters": "smartrecruiters.com",
    "icims": "icims.com",
    "taleo": "taleo.net",
}

TIMEOUT = 18
MAX_FOLLOW_PER_SOURCE = 1          # follow up to N portal links per outbound source to keep it fast
MAX_POSTINGS_PER_PORTAL = 30       # collect up to N internship postings per portal root
MAX_TOTAL_POSTINGS = 150           # global cap

# -------------------- Helpers --------------------
def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+","",(s or "").lower())

def _read_url(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")

def _is_internish(text: str) -> bool:
    s = (text or "").lower()
    return ("intern" in s) or ("co-op" in s) or ("co op" in s) or ("internship" in s)

def _job_like_path(path: str) -> bool:
    """
    For portals that we fetch without JS, prefer detail pages /jobs/ /job/ /positions/ with an id-like segment.
    """
    p = (path or "/").lower()
    if re.search(r"/job[s]?/|/position[s]?/|/opportunit|/careers/|/opening|/vacanc", p):
        return True
    # numeric or req-id style segments
    if re.search(r"[/-](req|job|r[e]?q|gh_jid|gh_src|posting)[-_]?\d+", p):
        return True
    # many Greenhouse/Lever detail pages just use /jobs/<id> or /apply/<id>
    if re.search(r"/apply/|/jobs?/", p):
        return True
    return False

def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _platform_of(url: str) -> str:
    h = _host(url)
    for name, pat in PLATFORMS.items():
        if pat in h:
            return name
    return "other"

def ollama_ready(host: str) -> bool:
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

# Simple Q&A fallback when LLM not available
_SIMPLE_QA = {
    r"\b(hi|hello|hey)\b": "Hi! I’m the CSUSB Internship Assistant. I can list internships, answer quick questions, and help you apply.",
    r"\b(who are you|your name)\b": "I’m the CSUSB Internship Assistant. I help you find and apply to internships.",
    r"\b(what can you do|help|capabilities)\b": "I can: (1) list internships from CSUSB’s page, (2) match you to roles after a quick interview, (3) draft materials, (4) auto-apply on Greenhouse/Lever.",
    r"\b(how (do|to) apply|apply steps)\b": "Say “apply for internships”. I’ll ask a few questions, read your résumé (optional), then show matches and generate application materials.",
    r"\b(internship tips|resume tips|cover letter)\b": "Keep bullets quantified, mirror keywords from postings, keep cover letters ~120 words.",
}
def rule_based_answer(user: str) -> str:
    s = (user or "").lower().strip()
    for pat, ans in _SIMPLE_QA.items():
        if re.search(pat, s):
            return ans
    if "intern" in s:
        return "Say “internships” to list CSUSB postings, or “apply for internships” to start a short questionnaire."
    return "I can list internships, answer quick questions, and help you apply. Try “internships” or “apply for internships”."

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

# -------------------- Phase 1 Interview --------------------
INTERVIEW_QUESTIONS = [
    ("role",       "What roles are you targeting? (e.g., software, data, security)"),
    ("skills",     "List your top 5–8 skills (comma-separated)."),
    ("company",    "Any preferred companies? (optional)"),
    ("location",   "Preferred location(s)? (optional)"),
    ("remote",     "Remote type preference? (any/remote/hybrid/onsite)"),
    ("exp",        "Rough years of experience (0/0.5/1/2…)? (optional)"),
    ("courses",    "Relevant courses or projects? (optional, comma-separated)"),
    ("availability","When can you start? (e.g., immediately, Jan 2026)"),
    ("auth",       "Work authorization (e.g., US citizen, OPT, CPT, H1B)? (optional)"),
    ("extras",     "Anything else to prioritize? (optional)"),
]
def merge_interview_into_profile(answers: dict) -> dict:
    roles = [w.strip() for w in (answers.get("role") or "").split(",") if w.strip()]
    skills = [w.strip() for w in (answers.get("skills") or "").split(",") if w.strip()]
    try:
        exp_years = float((answers.get("exp") or "").replace("+","").strip())
    except Exception:
        exp_years = None
    return {"roles": roles, "skills": skills, "exp_years": exp_years}

# -------------------- Resume parse --------------------
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

def resume_to_profile(resume_text: str, role_hint: str, skills_hint: str):
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

# -------------------- Scraping: CSUSB + Deep to portals --------------------
def scrape_csusb_links() -> list[dict]:
    """Pull outbound links from CSUSB page (fast)."""
    html = _read_url(CSUSB_URL)
    soup = _soup(html)
    main = soup.find("main") or soup
    rows, seen = [], set()
    for a in main.find_all("a", href=True):
        t = re.sub(r"\s+"," ", a.get_text(" ", strip=True))
        if not t: continue
        absu = urljoin(CSUSB_URL, a["href"])
        k = (t.lower(), absu)
        if k in seen: continue
        if not _is_internish(t + " " + absu):  # keep only obviously internship-ish
            continue
        host = _host(absu)
        comp = host.split(".")[-2].capitalize() if host else ""
        rows.append({"title": t, "company_guess": comp, "link": absu, "host": host})
        seen.add(k)
    return rows

def collect_postings_from_portal(root_url: str, limit=MAX_POSTINGS_PER_PORTAL) -> list[dict]:
    """
    Heuristic deep-scrape for common portals using requests+bs4 (no JS).
    We only keep detail pages that look like job postings AND contain 'intern' text.
    """
    out, seen = [], set()
    try:
        html = _read_url(root_url)
    except Exception:
        return out
    soup = _soup(html)
    base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(root_url))
    for a in soup.find_all("a", href=True):
        href = a["href"]
        absu = urljoin(base, href)
        p = urlparse(absu).path
        text = a.get_text(" ", strip=True)
        if not _is_internish(text + " " + absu):
            continue
        if not _job_like_path(p):
            continue
        key = absu.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": text or "Internship", "link": absu})
        if len(out) >= limit:
            break
    # If we found nothing, also consider the full page text and look for A tags with internish in URL alone
    if not out:
        for a in soup.find_all("a", href=True):
            absu = urljoin(base, a["href"])
            if "intern" in absu.lower() and _job_like_path(urlparse(absu).path):
                key = absu.lower()
                if key in seen: continue
                seen.add(key)
                out.append({"title": a.get_text(" ", strip=True) or "Internship", "link": absu})
                if len(out) >= limit: break
    return out

@st.cache_data(ttl=60*20, show_spinner=False)
def deep_collect_postings() -> pd.DataFrame:
    """
    1) Get internship-ish outbound links from CSUSB page.
    2) For each link:
       - If it's a well-known job portal root/listing, collect individual internship postings.
       - Else if it already looks like a posting, keep it.
    """
    sources = scrape_csusb_links()
    all_rows, total = [], 0
    for src in sources:
        url = src["link"]
        plat = _platform_of(url)
        # If already looks like a posting (detail page), keep it directly
        if _job_like_path(urlparse(url).path) and _is_internish(url):
            all_rows.append({
                "title": src["title"],
                "company": src["company_guess"],
                "link": url,
                "platform": plat,
                "source": CSUSB_URL,
                "posted": datetime.utcnow().date().isoformat(),
                "blob": _canon(f"{src['title']} {src['company_guess']} {url} {plat}")
            })
            total += 1
            if total >= MAX_TOTAL_POSTINGS: break
            continue

        # If it's a known portal, follow it and pull detail postings
        if plat in PLATFORMS:
            # follow only a few unique portal roots to avoid hammering
            postings = collect_postings_from_portal(url, limit=MAX_POSTINGS_PER_PORTAL)
            for p in postings:
                all_rows.append({
                    "title": p["title"],
                    "company": src["company_guess"] or urlparse(p["link"]).netloc.split(".")[-2].capitalize(),
                    "link": p["link"],
                    "platform": plat,
                    "source": url,
                    "posted": datetime.utcnow().date().isoformat(),
                    "blob": _canon(f"{p['title']} {src['company_guess']} {p['link']} {plat}")
                })
                total += 1
                if total >= MAX_TOTAL_POSTINGS: break
        # else: ignore generic pages (not detail)

        if total >= MAX_TOTAL_POSTINGS:
            break

    # Deduplicate by link
    if not all_rows:
        return pd.DataFrame(columns=["title","company","link","platform","source","posted","blob"])
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["link"]).reset_index(drop=True)
    return df

# -------------------- Matching & scoring --------------------
def score_posting(row, profile, company_pref, location, remote):
    score = 0
    blob = row.get("blob","")
    for r in (profile or {}).get("roles", []):
        if _canon(r) in blob: score += 3
    for s in (profile or {}).get("skills", []):
        if _canon(s) in blob: score += 2
    if company_pref and _canon(company_pref) in blob: score += 4
    if location and _canon(location) in blob: score += 1
    if remote and remote.lower()!="any" and _canon(remote) in blob: score += 1
    return score

def filter_rank(df, profile, query_text: str):
    if df.empty: return df
    q = (query_text or "").strip().lower()
    comp_tok = None
    if "intern" in q:
        # crude company token extraction: last non-generic word before 'intern'
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-\.]{1,}", q)
        for i, t in enumerate(tokens):
            if "intern" in t.lower() and i > 0:
                comp_tok = tokens[i-1]
                break

    # textual filter by query
    if comp_tok:
        tok = _canon(comp_tok)
        df = df[df["blob"].str.contains(tok, na=False)]
        if df.empty:
            return df

    # score and rank
    df = df.copy()
    df["match_score"] = df.apply(lambda r: score_posting(r, profile, comp_tok, None, "any"), axis=1)
    df = df.sort_values(["match_score","posted"], ascending=[False, False])
    return df

# -------------------- Application Kit --------------------
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

# -------------------- UI --------------------
st.set_page_config(page_title="CSUSB Internship Assistant", page_icon="💬", layout="wide")
st.title("💬 CSUSB Internship Assistant")
st.caption("Greet → general chat • say “internships” to list (deep scrape to posting pages) • say “apply for internships” to personalize, upload résumé, generate Application Kits, and Auto-Apply (Greenhouse/Lever).")

# Session state init
for k, v in {
    "mode": "greet",
    "llm_ready": ollama_ready(OLLAMA),
    "interview_active": False,
    "interview_idx": 0,
    "interview_answers": {},
    "applicant_info": None,
    "profile": None,
    "results_df": None,
    "selected": set(),
}.items():
    if k not in st.session_state: st.session_state[k] = v

# Greet
if st.session_state["mode"] == "greet":
    st.markdown("**Hi! How can I help you today?**")
    user = st.text_input("Type here (e.g., 'Amazon internships', 'Apply for internships', or any question) 👇")
    if not user:
        st.stop()
    txt = user.lower()

    # Route: apply
    if re.search(r"\b(apply|application|apply for)\b.*\b(intern|internship)", txt) or txt.strip() in {"apply", "apply for internships"}:
        st.session_state["interview_active"] = True
        st.session_state["interview_idx"] = 0
        st.session_state["interview_answers"] = {}
        st.session_state["mode"] = "apply"
        st.rerun()

    # Route: list/search
    if "intern" in txt:
        with st.spinner("Deep-scraping internship postings…"):
            df = deep_collect_postings()
        if df.empty:
            st.warning("No internship postings found right now.")
        else:
            # filter & rank w/ minimal profile (none yet)
            df2 = filter_rank(df, None, user)
            st.session_state["results_df"] = df2
            st.session_state["mode"] = "list"
            st.rerun()

    # General Q&A
    ans = llm_answer(user) if st.session_state["llm_ready"] else ""
    if not ans:
        ans = rule_based_answer(user)
    st.markdown("**Assistant:** " + ans)
    st.stop()

# List results (from greet search or manual)
if st.session_state["mode"] == "list":
    st.subheader("🔎 Internship Postings (deep-scraped)")
    df = st.session_state.get("results_df")
    if df is None:
        with st.spinner("Deep-scraping internship postings…"):
            df = deep_collect_postings()
        st.session_state["results_df"] = df
    if df is None or df.empty:
        st.warning("No internship postings found at the moment.")
    else:
        view = df[["title","company","platform","link","posted"]].reset_index(drop=True)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption("Tip: Use the “Apply for internships” flow to personalize, upload résumé, rank, generate kits, and auto-apply.")

    if st.button("Back ↩️"):
        st.session_state["mode"] = "greet"; st.rerun()

# Apply flow (interview -> form -> rank -> kits -> auto-apply)
if st.session_state["mode"] == "apply":
    # 1) Interview
    if st.session_state.get("interview_active", False):
        idx = st.session_state.get("interview_idx", 0)
        if idx < len(INTERVIEW_QUESTIONS):
            key, qtext = INTERVIEW_QUESTIONS[idx]
            st.subheader("🧭 Internship Application – Quick Questions")
            st.write(f"**{idx+1}/{len(INTERVIEW_QUESTIONS)}**  {qtext}")
            ans = st.text_input("Your answer:", key=f"interview_{key}")
            c1, c2 = st.columns(2)
            if c1.button("Next ➡️"):
                st.session_state["interview_answers"][key] = ans
                st.session_state["interview_idx"] = idx + 1
                st.rerun()
            if c2.button("Skip"):
                st.session_state["interview_answers"][key] = ""
                st.session_state["interview_idx"] = idx + 1
                st.rerun()
            st.stop()
        else:
            # finish interview
            A = st.session_state.get("interview_answers", {})
            st.session_state["profile"] = merge_interview_into_profile(A)
            st.session_state["interview_active"] = False
            st.success("Thanks! I used your answers to prefill your preferences below.")

    # 2) Application form
    st.subheader("📄 Your Info & Preferences")
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

        pref_company = st.text_input("Preferred company (optional)")
        role_hint    = st.text_input("Desired role(s) (e.g., software, data, security)",
                                     value=", ".join(st.session_state.get("profile",{}).get("roles",[])))
        skills_hint  = st.text_input("Top skills (comma-separated)",
                                     value=", ".join(st.session_state.get("profile",{}).get("skills",[])))
        resume       = st.file_uploader("Upload résumé (PDF, DOCX, or TXT)", type=["pdf","docx","txt"])

        submitted = st.form_submit_button("Find & Rank Internships 🔍")

    # 3) Ranking
    ranked = None
    if submitted:
        resume_text = extract_text_from_upload(resume)
        prof = resume_to_profile(resume_text, role_hint, skills_hint)
        st.session_state["profile"] = prof
        with st.spinner("Deep-scraping internship postings…"):
            df = deep_collect_postings()
        if df.empty:
            st.warning("No internship postings found.")
        else:
            df2 = df.copy()
            # simple textual company filter first if provided
            if pref_company:
                df2 = df2[df2["blob"].str.contains(_canon(pref_company), na=False)]
                if df2.empty:
                    df2 = df.copy()  # fallback
            # rank
            df2["match_score"] = df2.apply(lambda r: score_posting(r, prof, pref_company, location, remote), axis=1)
            df2 = df2.sort_values(["match_score","posted"], ascending=[False, False]).reset_index(drop=True)
            ranked = df2.head(30)
            st.session_state["results_df"] = ranked

    df = st.session_state.get("results_df")
    if df is not None and not df.empty:
        st.subheader("🎯 Best Matches")
        st.caption("Select internships you want to apply to, then generate Application Kits or Auto-Apply (Greenhouse/Lever).")
        picks = []
        for i, r in df.iterrows():
            key = f"pick_{i}"
            with st.expander(f"{i+1}. {r['title']} — {r.get('company','')} ({r.get('platform','')})"):
                st.write(f"[Open posting / Apply]({r['link']})")
                chosen = st.checkbox("Select this internship", key=key, value=False)
                if chosen:
                    picks.append(r.to_dict())

        # 4) Application Kits
        if st.button("Generate Application Kits"):
            who = {
                "name": name, "email": email, "phone": phone, "linkedin": linkedin
            }
            prof = st.session_state.get("profile") or {"roles":[],"skills":[],"exp_years":None}
            if not picks:
                st.warning("Select at least one internship above.")
            else:
                for item in picks:
                    kit_md, kit_txt = build_application_kit(item, prof, who)
                    st.markdown(f"**Application Kit – {item.get('title')} @ {item.get('company','')}**")
                    st.markdown(kit_md or "_(Enable Ollama for richer kits.)_")
                    st.download_button(
                        label=f"⬇️ Download Kit ({item.get('company','')[:18]} – {item.get('title','')[:26]})",
                        data=kit_txt.encode("utf-8"),
                        file_name=f"application_kit_{_canon(item.get('company',''))}_{_canon(item.get('title',''))}.txt",
                        mime="text/plain"
                    )

        # 5) Auto-Apply
        st.markdown("---")
        st.subheader("⚡ Auto-Apply (beta) — Greenhouse & Lever only")
        st.caption("Fills common fields (name, email, phone, résumé, cover note). Other portals will be marked 'manual_required'.")
        resume_for_auto = st.file_uploader("Upload résumé for auto-apply (PDF preferred)", type=["pdf"], key="resume_for_auto")
        if st.button("Auto-apply to selected"):
            if not picks:
                st.warning("Select at least one internship above.")
                st.stop()
            if not resume_for_auto:
                st.warning("Please upload your resume (PDF) for auto-apply.")
                st.stop()
            who = {"name": name, "email": email, "phone": phone, "linkedin": linkedin}
            prof = st.session_state.get("profile") or {"roles":[],"skills":[],"exp_years":None}
            cover = llm_answer(
                f"Write a concise 6–8 line cover letter for internships. Roles={prof.get('roles')}, "
                f"skills={prof.get('skills')}. Mention eagerness to learn and link to LinkedIn: {linkedin}.",
                sys_msg="Professional, succinct tone."
            ) or (
                f"Dear Hiring Team,\n\n"
                f"I am excited to apply for internship roles. My background includes {', '.join(prof.get('skills')[:6])} "
                f"and interests in {', '.join(prof.get('roles')[:3])}. I am eager to contribute and learn.\n"
                f"LinkedIn: {linkedin}\n\n"
                f"Thank you for your consideration.\n"
                f"{name}\n{email}\n{phone}"
            )
            with st.spinner("Submitting applications…"):
                results = asyncio.run(auto_apply_batch(picks, who, resume_for_auto.read(), cover))
            st.success("Auto-apply finished. Review statuses below.")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.markdown(
                "Legend: **submitted_or_attempted** = form filled & submit click attempted • "
                "**manual_required** = unsupported portal; open link and apply manually."
            )

    if st.button("Back ↩️"):
        st.session_state["mode"] = "greet"
        st.session_state["results_df"] = None
        st.rerun()
