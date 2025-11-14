"""
playwright_fetcher.py
CSUSB-focused Playwright utilities to fetch HTML and extract links.
- No LLM usage.
- No deep navigation of external company career sites.
- Includes an internal crawler that stays on CSUSB CSE hosts only.
"""

from __future__ import annotations

import asyncio
from typing import Optional, List, Dict, Tuple, Set, Iterable
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup


# A stable, “normal” desktop UA
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# Allowed CSUSB CSE hosts
CSUSB_ALLOWED_HOSTS = {"cse.csusb.edu", "sec.cse.csusb.edu"}

# Skip patterns that will never be “real” content pages
SKIP_URL_PATTERNS = (
    "mailto:", "tel:", "javascript:", "#",
)

# Skip obvious external/social/irrelevant domains
SKIP_DOMAINS = {
    "facebook.com", "twitter.com", "instagram.com", "youtube.com", "youtu.be",
    "linkedin.com", "tiktok.com", "snapchat.com",
}

# Resource types to block (faster, less flaky)
BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}


def _is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _same_host(url: str, allowed_hosts: Set[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return host in allowed_hosts
    except Exception:
        return False


def _should_skip_url(href: str) -> bool:
    low = (href or "").strip().lower()
    if not low:
        return True
    if any(low.startswith(p) for p in SKIP_URL_PATTERNS):
        return True
    return False


def _is_skippable_domain(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(skip in host for skip in SKIP_DOMAINS)


def _extract_links(html: str, base_url: str) -> List[Dict]:
    """Extract (text, url, domain) from HTML, aggressively but safely."""
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content nodes
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    links: List[Dict] = []
    seen: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if _should_skip_url(href):
            continue

        # Absolute URL
        try:
            abs_url = urljoin(base_url, href)
        except Exception:
            continue

        if not _is_http_url(abs_url):
            continue

        # De-dup
        if abs_url in seen:
            continue
        seen.add(abs_url)

        # Text content (fallback to aria-label/title if empty)
        txt = (a.get_text(" ", strip=True) or a.get("aria-label") or a.get("title") or "Link").strip()
        txt = txt[:150]

        links.append({
            "text": txt,
            "url": abs_url,
            "domain": urlparse(abs_url).netloc.lower(),
        })

    return links


class PlaywrightFetcher:
    """
    Async Playwright fetcher for HTML/links.
    Includes a CSUSB-only internal crawler (BFS, bounded by max_pages).
    """

    def __init__(self, timeout_ms: int = 15000, wait_ms: int = 1500):
        self.timeout_ms = timeout_ms
        self.wait_ms = wait_ms
        self.visited_urls: Set[str] = set()

    async def _create_context(self, p):
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(user_agent=UA)

        # Block heavy/irrelevant resources
        async def _route_handler(route):
            req = route.request
            try:
                if req.resource_type in BLOCKED_RESOURCE_TYPES:
                    await route.abort()
                    return
                # quick heuristic: block common trackers
                url_low = req.url.lower()
                if "doubleclick" in url_low or "analytics" in url_low or "tracking" in url_low:
                    await route.abort()
                    return
            except Exception:
                pass
            await route.continue_()

        await context.route("**/*", _route_handler)
        return browser, context

    async def fetch_html(self, url: str) -> Optional[str]:
        """
        Fetch HTML for a single page (no navigation beyond this page).
        """
        if url in self.visited_urls:
            return None
        self.visited_urls.add(url)

        try:
            async with async_playwright() as p:
                browser, context = await self._create_context(p)
                page = await context.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    # allow minimal time for lazy content
                    await page.wait_for_timeout(self.wait_ms)
                    try:
                        # optional: small idle wait; ignore if not reached
                        await page.wait_for_load_state("networkidle", timeout=1000)
                    except Exception:
                        pass
                    html = await page.content()
                    return html
                finally:
                    await context.close()
                    await browser.close()
        except PlaywrightTimeout:
            return None
        except Exception:
            return None

    async def extract_text_and_links(self, url: str) -> Tuple[str, List[Dict]]:
        """
        Convenience: fetch a page then parse it into (text, links).
        """
        html = await self.fetch_html(url)
        if not html:
            return "", []
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Compact text
        text = " ".join(soup.get_text(" ", strip=True).split())
        text = text[:3000]
        links = _extract_links(html, base_url=url)
        return text, links

    async def crawl_csusb_links(
        self,
        start_url: str,
        allowed_hosts: Iterable[str] = CSUSB_ALLOWED_HOSTS,
        max_pages: int = 30,
    ) -> List[Dict]:
        """
        Crawl **only** within CSUSB CSE hosts (breadth-first), up to `max_pages`.
        Returns a de-duplicated list of link dicts: {text, url, domain}.
        """
        allowed_hosts_set = set(allowed_hosts)
        queue: List[str] = [start_url]
        seen_pages: Set[str] = set()
        collected: List[Dict] = []
        seen_links: Set[str] = set()

        async with async_playwright() as p:
            browser, context = await self._create_context(p)
            try:
                while queue and len(seen_pages) < max_pages:
                    current = queue.pop(0)
                    if current in seen_pages:
                        continue
                    if not _same_host(current, allowed_hosts_set):
                        continue

                    page = await context.new_page()
                    try:
                        await page.goto(current, wait_until="domcontentloaded", timeout=self.timeout_ms)
                        await page.wait_for_timeout(self.wait_ms)
                        try:
                            await page.wait_for_load_state("networkidle", timeout=1000)
                        except Exception:
                            pass

                        html = await page.content()
                        seen_pages.add(current)

                        # Extract links from this page
                        links = _extract_links(html, base_url=current)

                        # Keep only links within the allowed hosts; enqueue HTML pages we haven't seen
                        for link in links:
                            url = link["url"]
                            if not _same_host(url, allowed_hosts_set):
                                # external: skip (no deep external navigation)
                                continue
                            if _is_skippable_domain(url):
                                continue

                            # Collect link (dedup by URL)
                            if url not in seen_links:
                                collected.append(link)
                                seen_links.add(url)

                            # Enqueue additional pages to crawl (same host)
                            # Only enqueue HTML-like pages (avoid PDFs/images)
                            path = urlparse(url).path.lower()
                            if any(path.endswith(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".zip")):
                                continue
                            if url not in seen_pages and url not in queue:
                                queue.append(url)
                    finally:
                        await page.close()
            finally:
                await context.close()
                await browser.close()

        # De-duplicate collected links by URL, preserve first text seen
        unique: Dict[str, Dict] = {}
        for item in collected:
            unique.setdefault(item["url"], item)
        return list(unique.values())


# ----------------------
# Synchronous wrappers
# ----------------------

def fetch_html_sync(url: str, timeout_ms: int = 15000, wait_ms: int = 1500) -> Optional[str]:
    try:
        fetcher = PlaywrightFetcher(timeout_ms=timeout_ms, wait_ms=wait_ms)
        return asyncio.run(fetcher.fetch_html(url))
    except Exception:
        return None


def extract_text_and_links_sync(url: str, timeout_ms: int = 15000, wait_ms: int = 1500) -> Tuple[str, List[Dict]]:
    try:
        fetcher = PlaywrightFetcher(timeout_ms=timeout_ms, wait_ms=wait_ms)
        return asyncio.run(fetcher.extract_text_and_links(url))
    except Exception:
        return "", []


def crawl_csusb_links_sync(
    start_url: str,
    allowed_hosts: Iterable[str] = CSUSB_ALLOWED_HOSTS,
    max_pages: int = 30,
    timeout_ms: int = 15000,
    wait_ms: int = 1500,
) -> List[Dict]:
    try:
        fetcher = PlaywrightFetcher(timeout_ms=timeout_ms, wait_ms=wait_ms)
        return asyncio.run(fetcher.crawl_csusb_links(start_url, allowed_hosts=allowed_hosts, max_pages=max_pages))
    except Exception:
        return []
