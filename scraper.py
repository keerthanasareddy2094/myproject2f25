
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ---------- constants ----------
CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

# Kept for compatibility; we don't deep crawl anymore
MAX_PAGES = int(os.getenv("MAX_PAGES", "30"))
TIMEOUT_MS = int(os.getenv("TIMEOUT_MS", "15000"))

# Host/keyword filters to avoid junk or social links
JUNK_HOSTS = {"youtube.com", "youtu.be", "facebook.com", "twitter.com", "linkedin.com"}
JUNK_KEYWORDS = {
    "proposal form", "evaluation form", "student evaluation", "supervisor evaluation",
    "report form", "handbook", "resume", "cv", "smartscholarship", "scholarship",
    "faculty & staff", "career center", "advising", "contact us", "about us"
}
# Hints that a link is jobs/careers-related
ALLOW_HOST_HINTS = {
    "myworkdayjobs", "workday", "greenhouse", "lever", "taleo", "icims", "smartrecruiters",
    "jobs", "careers", "career", "intern"
}

# Text hints on the CSUSB page that likely indicate relevant destinations
INTERNSHIP_INDICATORS = [
    "intern", "internship", "co-op", "coop", "summer program",
    "student position", "campus hire", "university", "graduate program",
    "summer analyst", "early insight", "industrial placement", "apprentice", "apprenticeship"
]

# ---------- helpers ----------
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _domain(u: str) -> str:
    try:
        return urlparse(u).netloc
    except Exception:
        return ""

def _infer_company(abs_url: str, text: str = "") -> Optional[str]:
    """Infer company name from URL or link text."""
    try:
        host = urlparse(abs_url).netloc.lower().replace("www.", "")
        parts = host.split(".")

        # e.g., <company>.myworkdayjobs.com
        if "myworkdayjobs" in host and len(parts) > 2:
            return parts[0].capitalize()

        if len(parts) >= 2:
            core = parts[-2]
            core = re.sub(r"(jobs?|careers?|hire|recruiting)$", "", core)
            if len(core) > 2:
                return core.capitalize()

        if text:
            match = re.search(
                r"^([A-Z][a-zA-Z\s&\.]+?)(?:\s*[-—–]\s*(?:Careers?|Jobs?|Internships?))?$",
                text
            )
            if match:
                return match.group(1).strip()
    except Exception:
        pass
    return None

def _is_candidate_link(text: str, url: str) -> bool:
    """Decide if a link from the CSUSB CSE page looks like a career/internship destination."""
    low = f"{text} {url}".lower()

    # filter obvious non-job resources
    if any(k in low for k in JUNK_KEYWORDS):
        return False

    host = urlparse(url).netloc.lower()
    if any(h in host for h in JUNK_HOSTS):
        return False

    # include if internship-relevant words show up
    if any(ind in low for ind in INTERNSHIP_INDICATORS):
        return True

    # or if the host/text looks like jobs/careers
    return any(h in host or h in low for h in ALLOW_HOST_HINTS)

def _collect_links(page_html: str, base: str) -> List[Dict]:
    """Collect candidate career/internship links from the CSUSB page."""
    soup = BeautifulSoup(page_html, "lxml")
    main = soup.find("main") or soup
    rows, seen = [], set()

    for a in main.find_all("a", href=True):
        text = _clean(a.get_text(" ", strip=True))
        if not text or len(text) < 3:
            continue

        href = a["href"]
        abs_url = urljoin(base, href)
        host = urlparse(abs_url).netloc.lower()

        key = (text.lower(), abs_url)
        if key in seen:
            continue

        if not _is_candidate_link(text, abs_url):
            continue

        rows.append({
            "title": text,
            "company": _infer_company(abs_url, text),
            "link": abs_url,
            "host": host,
            "source": base,
            "posted_date": datetime.utcnow().date().isoformat(),
        })
        seen.add(key)

    return rows

# ---------- MAIN (CSUSB-only) SCRAPER ----------
def scrape_csusb_listings(
    url: str = CSUSB_CSE_URL,
    timeout_ms: int = TIMEOUT_MS,
    deep: bool = False,          # kept only for backward compatibility; ignored
    max_pages: int = MAX_PAGES,  # kept only for backward compatibility; ignored
) -> pd.DataFrame:
    """
    Scrape CSUSB CSE internships page and return candidate destination links.
    No deep crawling of external company sites is performed.
    Returns a DataFrame with columns: link, title, company, host, source, posted_date.
    """
    print(f"Scraping CSUSB page (links only): {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox"]
        )
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 720})

        # Block heavy/irrelevant resources for speed/reliability
        def _should_block(u: str, rtype: str) -> bool:
            if rtype in {"image", "media", "font", "stylesheet"}:
                return True
            return any(b in u for b in ["analytics", "doubleclick", "tracking"])

        ctx.route(
            "**/*",
            lambda route, req: route.abort()
            if _should_block(req.url, req.resource_type)
            else route.continue_()
        )

        page = ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=2000)
            except Exception:
                pass
            html_src = page.content()
        except PlaywrightTimeout:
            browser.close()
            print(f"✗ Timeout loading {url}")
            return pd.DataFrame(columns=["link", "title", "company", "host", "source", "posted_date"])
        except Exception as e:
            browser.close()
            print(f"✗ Error loading {url}: {e}")
            return pd.DataFrame(columns=["link", "title", "company", "host", "source", "posted_date"])

        browser.close()

    # Extract and normalize links from the CSUSB page only
    rows = _collect_links(html_src, base=url)
    print(f"Found {len(rows)} candidate links on CSUSB CSE page")

    # Build final DataFrame (only the columns the UI needs)
    cols = ["link", "title", "company", "host", "source", "posted_date"]
    df = pd.DataFrame(rows, columns=cols).drop_duplicates(subset=["link"], keep="first")

    # Sort by most recently scraped (posted_date is the scrape date here)
    try:
        df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    print(f"✓ Total unique links collected: {len(df)}")
    return df
