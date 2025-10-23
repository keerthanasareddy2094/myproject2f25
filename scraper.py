# scraper.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable, Tuple
from urllib.parse import urljoin, urlparse
import asyncio, os, re, time, html
import httpx
from bs4 import BeautifulSoup

# =========================
# Config
# =========================
CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
CONCURRENCY = int(os.getenv("SCRAPE_CONCURRENCY", "8"))
USE_PLAYWRIGHT = os.getenv("USE_PLAYWRIGHT", "0") in {"1", "true", "yes"}

JUNK_HOSTS = {"youtube.com", "youtu.be", "facebook.com", "twitter.com", "linkedin.com"}
JUNK_KEYWORDS = {
    "proposal form", "evaluation form", "student evaluation", "supervisor evaluation",
    "report form", "handbook", "resume", "cv", "scholarship", "career center", "advising",
}
ALLOW_HOST_HINTS = {"workday", "myworkdayjobs", "greenhouse", "lever", "taleo", "icims", "smartrecruiters",
                    "jobs", "careers", "career", "opportunities"}
INTERNSHIP_TERMS = re.compile(
    r"\b(intern|internship|co-?op|summer\s+analyst|student|early\s+career|graduate\s+program|apprentice)\b",
    re.I
)

# =========================
# Data model
# =========================
@dataclass
class Posting:
    title: str
    company: str
    link: str
    location: Optional[str] = None
    remote: Optional[str] = None
    posted_date: Optional[str] = None
    source: Optional[str] = None
    host: Optional[str] = None
    details: Optional[str] = None
    salary: Optional[str] = None
    education: Optional[str] = None
    deadline: Optional[str] = None
    tags: Optional[str] = None

def _domain(u: str) -> str:
    try: return urlparse(u).netloc.lower()
    except: return ""

def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _is_candidate_link(text: str, url: str) -> bool:
    low = f"{text} {url}".lower()
    if any(k in low for k in JUNK_KEYWORDS): return False
    host = _domain(url)
    if any(h in host for h in JUNK_HOSTS): return False
    if "intern" in low or "co-op" in low: return True
    return any(h in host or h in low for h in ALLOW_HOST_HINTS)

def _infer_company_from_url(abs_url: str) -> Optional[str]:
    host = _domain(abs_url)
    host = host.replace("www.", "")
    parts = host.split(".")
    # workday: brand.wdX.myworkdayjobs.com
    if "myworkdayjobs" in host and len(parts) >= 4:
        return parts[0].capitalize()
    if len(parts) >= 2:
        core = parts[-2]
        core = re.sub(r"(jobs?|careers?|recruit(ing)?|hire)$", "", core)
        if len(core) > 2:
            return core.capitalize()
    return None

# =========================
# Shallow: CSUSB page only
# =========================
async def _fetch(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        r = await client.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def _collect_from_html(page_html: str, base: str) -> List[Posting]:
    soup = BeautifulSoup(page_html, "lxml")
    main = soup.find("main") or soup
    rows, seen = [], set()
    for a in main.find_all("a", href=True):
        text = _clean(a.get_text(" ", strip=True))
        if not text:
            continue
        abs_url = urljoin(base, a["href"])
        if (text.lower(), abs_url) in seen:
            continue
        if not _is_candidate_link(text, abs_url):
            continue
        rows.append(Posting(
            title=text[:200],
            company=_infer_company_from_url(abs_url) or "",
            link=abs_url,
            source=base,
            host=_domain(abs_url),
        ))
        seen.add((text.lower(), abs_url))
    return rows

async def shallow_csusb_links() -> List[Posting]:
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": UA}) as client:
        html = await _fetch(client, CSUSB_CSE_URL)
        if not html:
            return []
        return _collect_from_html(html, CSUSB_CSE_URL)

# =========================
# Career URL discovery per company
# =========================
def guess_company_career_urls(company: str) -> List[str]:
    c = re.sub(r"[^a-z0-9 ]", "", (company or "").lower()).strip().replace(" ", "")
    if not c: return []
    guesses = [
        f"https://careers.{c}.com",
        f"https://www.{c}.com/careers",
        f"https://www.{c}.com/careers/students",
        f"https://jobs.{c}.com",
        f"https://{c}.com/careers",
        f"https://{c}.wd1.myworkdayjobs.com",
        f"https://{c}.wd5.myworkdayjobs.com",
    ]
    return guesses

async def discover_company_career_urls(company: str, include_csusb_page=True) -> List[str]:
    urls: List[str] = []
    # 1) Heuristics / typical career endpoints
    urls += guess_company_career_urls(company)
    # 2) Scan CSUSB page anchors for host or company name matches (optional)
    if include_csusb_page:
        links = await shallow_csusb_links()
        token = re.sub(r"[^a-z0-9]", "", company.lower())
        for p in links:
            blob = re.sub(r"[^a-z0-9]", "", (p.title or "") + " " + (p.host or "") + " " + (p.link or "")).lower()
            if token and token in blob:
                urls.append(p.link)
    # De-dup + keep only http(s)
    out, seen = [], set()
    for u in urls:
        if u and u.startswith("http") and u not in seen:
            out.append(u); seen.add(u)
    return out[:10]

# =========================
# Deep scrape (requests + bs4)
#   NOTE: works for static / light JS pages. For heavy JS,
#         set USE_PLAYWRIGHT=1 and use the optional path below.
# =========================
def _extract_cards(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    """Return list of (title, href) candidates containing internship cues."""
    out = []
    for a in soup.find_all("a", href=True, limit=500):
        txt = _clean(a.get_text(" ", strip=True))
        href = a["href"]
        blob = " ".join([txt.lower(), href.lower()])
        if INTERNSHIP_TERMS.search(blob):
            abs_u = href
            out.append((txt, abs_u))
    return out

async def deep_scrape_company_requests(company: str, seeds: Iterable[str], max_pages: int = 40) -> List[Posting]:
    results: List[Posting] = []
    visited: set[str] = set()

    async def visit(u: str, client: httpx.AsyncClient):
        if len(visited) >= max_pages:
            return
        if u in visited:
            return
        visited.add(u)
        try:
            r = await client.get(u, timeout=REQUEST_TIMEOUT)
            if r.status_code >= 400:
                return
            soup = BeautifulSoup(r.text, "lxml")
            cards = _extract_cards(soup)
            for title, href in cards:
                link = urljoin(u, href)
                if not link.startswith("http"): 
                    continue
                if link in {p.link for p in results}:
                    continue
                if not INTERNSHIP_TERMS.search(title.lower() + " " + link.lower()):
                    continue
                results.append(Posting(
                    title=title[:180] or "Internship",
                    company=company,
                    link=link,
                    host=_domain(link),
                    source=u,
                ))
            # discover more pages to crawl (within same host or ATS)
            for a in soup.find_all("a", href=True):
                nxt = urljoin(u, a["href"])
                d = _domain(nxt)
                if not nxt.startswith("http"): 
                    continue
                # keep crawl small: only same brand host or known ATS hosts
                if company.lower().replace(" ", "") in d or any(h in d for h in ALLOW_HOST_HINTS):
                    if len(visited) + 1 < max_pages:
                        await visit(nxt, client)
        except Exception:
            return

    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": UA}) as client:
        tasks = []
        for s in seeds:
            if len(visited) >= max_pages: break
            if s.startswith("http"):
                tasks.append(visit(s, client))
        # run sequentially but cooperatively to keep RAM low
        for t in tasks:
            await t

    # Deduplicate by link
    dedup, seen = [], set()
    for p in results:
        if p.link not in seen:
            dedup.append(p); seen.add(p.link)

    return dedup

# =========================
# Optional: Playwright path (JS-heavy sites)
# Toggle with USE_PLAYWRIGHT=1
# =========================
def deep_scrape_company_playwright(company: str, seeds: Iterable[str], max_pages: int = 40) -> List[Posting]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return []

    visited: set[str] = set()
    found: List[Posting] = []

    def should_visit(u: str) -> bool:
        if len(visited) >= max_pages: return False
        d = _domain(u)
        return company.lower().replace(" ", "") in d or any(h in d for h in ALLOW_HOST_HINTS)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        queue = [u for u in seeds if u.startswith("http")]
        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited: continue
            visited.add(url)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                html = page.content()
                soup = BeautifulSoup(html, "lxml")
                for title, href in _extract_cards(soup):
                    link = urljoin(url, href)
                    blob = (title + " " + link).lower()
                    if INTERNSHIP_TERMS.search(blob) and link not in {p.link for p in found}:
                        found.append(Posting(
                            title=title[:180] or "Internship",
                            company=company,
                            link=link,
                            host=_domain(link),
                            source=url,
                        ))
                # discover next
                for a in soup.find_all("a", href=True):
                    nxt = urljoin(url, a["href"])
                    if nxt not in visited and should_visit(nxt):
                        queue.append(nxt)
            except Exception:
                continue
        browser.close()
    # de-dup
    unique, seen = [], set()
    for p in found:
        if p.link not in seen:
            unique.append(p); seen.add(p.link)
    return unique

# =========================
# Public API
# =========================
async def shallow_search() -> List[Dict]:
    """Return CSUSB page links as dicts."""
    posts = await shallow_csusb_links()
    return [asdict(p) for p in posts]

async def deep_search_for_company(company: str, max_pages: int = 40) -> List[Dict]:
    seeds = await discover_company_career_urls(company, include_csusb_page=True)
    if not seeds:
        return []
    if USE_PLAYWRIGHT:
        results = deep_scrape_company_playwright(company, seeds, max_pages=max_pages)
    else:
        results = await deep_scrape_company_requests(company, seeds, max_pages=max_pages)
    return [asdict(p) for p in results]
