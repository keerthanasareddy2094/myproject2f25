from __future__ import annotations
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime
import os, re, time, requests, pandas as pd
from bs4 import BeautifulSoup

# ---------- constants ----------
CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

TIMEOUT = int(os.getenv("TIMEOUT_MS", "10000")) // 1000
MAX_PAGES  = int(os.getenv("MAX_PAGES", "30"))
USE_PLAYWRIGHT = os.getenv("USE_PLAYWRIGHT", "0") in {"1", "true", "True"}

# --- filters ---
JUNK_HOSTS = {"youtube.com", "youtu.be", "facebook.com", "twitter.com", "linkedin.com"}
JUNK_KEYWORDS = {"form", "evaluation", "handbook", "scholarship", "resume", "career center", "advising"}
ALLOW_HOST_HINTS = {
    "myworkdayjobs", "workday", "greenhouse", "lever", "taleo", "icims",
    "smartrecruiters", "jobs", "careers", "career"
}

# ---------- helpers ----------
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _infer_company(abs_url: str) -> Optional[str]:
    try:
        host = urlparse(abs_url).netloc.lower()
        parts = host.split(".")
        if len(parts) >= 2:
            core = parts[-2]
            return core.capitalize()
        return host
    except Exception:
        return None

def _is_candidate_link(text: str, url: str) -> bool:
    low = f"{text} {url}".lower()
    if any(k in low for k in JUNK_KEYWORDS):
        return False
    host = urlparse(url).netloc.lower()
    if any(h in host for h in JUNK_HOSTS):
        return False
    return "intern" in low or any(h in host or h in low for h in ALLOW_HOST_HINTS)

def _collect_links(page_html: str, base: str) -> List[Dict]:
    """Extract all potential internship/career links from a page."""
    soup = BeautifulSoup(page_html, "lxml")
    main = soup.find("main") or soup
    rows, seen = [], set()
    for a in main.find_all("a", href=True):
        text = _clean(a.get_text(" ", strip=True))
        if not text:
            continue
        abs_url = urljoin(base, a["href"])
        host = urlparse(abs_url).netloc.lower()
        key = (text.lower(), abs_url)
        if key in seen:
            continue
        if not _is_candidate_link(text, abs_url):
            continue
        rows.append({
            "title": text,
            "company": _infer_company(abs_url),
            "location": None,
            "posted_date": datetime.utcnow().date().isoformat(),
            "tags": None,
            "link": abs_url,
            "host": host,
            "source": base,
            "deadline": None,
            "requirements": None,
            "salary": None,
            "education": None,
            "remote": None,
            "details": None,
        })
        seen.add(key)
    return rows

# ---------- shallow scraper ----------
def scrape_csusb_listings(
    url: str = CSUSB_CSE_URL,
    deep: bool = True,
    max_pages: int = MAX_PAGES
) -> pd.DataFrame:
    """
    Scrape CSUSB Internship & Careers page.
    If deep=True, it will also attempt to follow each company link
    and collect additional internship postings.
    """
    print(f"🔍 Scraping CSUSB page: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        print(f"✗ Error fetching CSUSB page: {e}")
        return pd.DataFrame()

    base_links = _collect_links(html, url)
    df = pd.DataFrame(base_links)
    if df.empty or not deep:
        return df

    print(f"Found {len(df)} candidate links. Deep scraping enabled ({'Playwright' if USE_PLAYWRIGHT else 'Requests'} mode).")
    all_results = []

    if USE_PLAYWRIGHT:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
                ctx = browser.new_context(user_agent=UA)
                page = ctx.new_page()
                for i, link in enumerate(df["link"].tolist()[:max_pages], 1):
                    print(f"[{i}/{max_pages}] Visiting {link}")
                    try:
                        page.goto(link, wait_until="domcontentloaded", timeout=15000)
                        time.sleep(1.5)
                        html2 = page.content()
                        rows = _collect_links(html2, link)
                        for r in rows:
                            r["source"] = link
                        all_results.extend(rows)
                    except Exception as e:
                        print(f"  ✗ {link[:60]}... {e}")
                browser.close()
        except Exception as e:
            print(f"⚠️ Playwright error: {e}. Falling back to requests.")
            for i, link in enumerate(df["link"].tolist()[:max_pages], 1):
                try:
                    r = requests.get(link, headers={"User-Agent": UA}, timeout=TIMEOUT)
                    rows = _collect_links(r.text, link)
                    for r_ in rows:
                        r_["source"] = link
                    all_results.extend(rows)
                except Exception as e:
                    print(f"  ✗ {link[:60]}... {e}")
    else:
        # Simple requests-based deep scrape
        for i, link in enumerate(df["link"].tolist()[:max_pages], 1):
            print(f"[{i}/{max_pages}] Visiting {link}")
            try:
                r = requests.get(link, headers={"User-Agent": UA}, timeout=TIMEOUT)
                rows = _collect_links(r.text, link)
                for r_ in rows:
                    r_["source"] = link
                all_results.extend(rows)
            except Exception as e:
                print(f"  ✗ {link[:60]}... {e}")

    if all_results:
        df2 = pd.DataFrame(all_results)
        df = pd.concat([df, df2], ignore_index=True)
    df.drop_duplicates(subset=["link"], inplace=True)
    print(f"✅ Total {len(df)} unique internship/career links collected.")
    return df

# ---------- company-specific quick search ----------
def quick_company_links_playwright(
    company_token: str,
    url: str = CSUSB_CSE_URL,
    deep: bool = True
) -> pd.DataFrame:
    """
    Quickly find internship/career links for a given company name.
    If deep=True, will visit the matched link(s) to collect more postings.
    """
    token = (company_token or "").strip().lower()
    if not token:
        return pd.DataFrame()
    print(f"🔎 Searching for company: {company_token}")

    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        soup = BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"✗ Error fetching CSUSB page: {e}")
        return pd.DataFrame()

    main = soup.find("main") or soup
    rows, seen = [], set()
    for a in main.find_all("a", href=True):
        text = _clean(a.get_text(" ", strip=True))
        if not text:
            continue
        abs_url = urljoin(url, a["href"])
        host = urlparse(abs_url).netloc.lower()
        key = (text.lower(), abs_url)
        if key in seen:
            continue
        if token in text.lower() or token in host:
            rows.append({
                "title": text,
                "company": _infer_company(abs_url) or company_token,
                "location": None,
                "posted_date": datetime.utcnow().date().isoformat(),
                "tags": None,
                "link": abs_url,
                "host": host,
                "source": url,
                "deadline": None,
                "requirements": None,
                "salary": None,
                "education": None,
                "remote": None,
                "details": None,
            })
            seen.add(key)

    df = pd.DataFrame(rows)
    if df.empty or not deep:
        print(f"✗ No links found for {company_token}")
        return df

    # Deep scrape the first few matched links
    deep_rows = []
    for link in df["link"].tolist()[:5]:
        try:
            print(f"  Visiting {link}")
            r = requests.get(link, headers={"User-Agent": UA}, timeout=TIMEOUT)
            rows2 = _collect_links(r.text, link)
            for r_ in rows2:
                r_["source"] = link
            deep_rows.extend(rows2)
        except Exception as e:
            print(f"  ✗ {link[:60]}... {e}")
    if deep_rows:
        df2 = pd.DataFrame(deep_rows)
        df = pd.concat([df, df2], ignore_index=True)
    df.drop_duplicates(subset=["link"], inplace=True)
    print(f"✅ Found {len(df)} total links for {company_token}")
    return df
