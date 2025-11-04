# app.py — minimal Streamlit app so the Docker image runs
import re, requests, pandas as pd, streamlit as st
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from datetime import datetime

CSUSB_URL = "https://www.csusb.edu/cse/internships-careers"
UA = "Mozilla/5.0 (CSUSB Internship Assistant)"

st.set_page_config(page_title="CSUSB Internship Assistant", page_icon="💬", layout="wide")
st.title("💬 CSUSB Internship Assistant")
st.caption("Type 'internships' to list CSUSB postings. This is a minimal starter app.py so Docker can run.")

def _clean(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", (s or "")).strip()

BAD_LAST = {"careers","career","jobs","job","students","graduates","early-careers"}
JUNK = {"proposal form","evaluation form","student evaluation","supervisor evaluation",
        "report form","handbook","resume","cv","scholarship","scholarships","grant program",
        "career center","advising","policy","forms","pdf"}

def _path_is_specific(path: str) -> bool:
    p = (path or "/").lower()
    if "intern" in p or "co-op" in p: return True
    seg = [s for s in p.split("/") if s]
    if any(re.search(r"\d{5,}", s) for s in seg): return True
    if seg and seg[-1] in BAD_LAST: return False
    return len(seg) >= 3

def _is_intern_link(text, url) -> bool:
    low = f"{text} {url}".lower()
    if any(k in low for k in JUNK): return False
    if not ("intern" in low or "co-op" in low): return False
    try: return _path_is_specific(urlparse(url).path)
    except: return False

@st.cache_data(ttl=60*30)
def fetch_csusb() -> pd.DataFrame:
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
        rows.append({
            "title": t, "company": comp, "link": absu, "host": host,
            "posted": datetime.utcnow().date().isoformat()
        })
        seen.add(k)
    return pd.DataFrame(rows)

q = st.text_input("Ask here (e.g., 'internships'):")
if q:
    if "intern" in q.lower():
        with st.spinner("Fetching internships from CSUSB…"):
            df = fetch_csusb()
        if df.empty:
            st.warning("No internship postings found on the CSUSB page right now.")
        else:
            st.success(f"Found {len(df)} postings:")
            st.dataframe(df[["title","company","link"]], use_container_width=True, hide_index=True)
    else:
        st.write("This minimal app only lists internships. Type **internships** to see the list.")
else:
    st.info("Tip: type **internships** and press Enter.")
