# search.py
from __future__ import annotations
import asyncio
from typing import Dict, List
from scraper import shallow_search, deep_search_for_company

async def discover_from_profile(profile: Dict, max_pages_per_company: int = 40) -> Dict[str, List[Dict]]:
    """
    Given the Phase-1 profile JSON, run discovery:
    - shallow CSUSB scan
    - deep company scraping for each target company (if any)
    """
    out: Dict[str, List[Dict]] = {"csusb_links": [], "companies": []}

    # 1) shallow CSUSB page
    out["csusb_links"] = await shallow_search()

    # 2) deep per company
    companies = [c for c in (profile.get("companies") or []) if c]
    for c in companies[:6]:  # keep sane cap
        postings = await deep_search_for_company(c, max_pages=max_pages_per_company)
        out["companies"].append({"company": c, "postings": postings})

    return out

def discover_from_profile_sync(profile: Dict, max_pages_per_company: int = 40) -> Dict:
    return asyncio.run(discover_from_profile(profile, max_pages_per_company))
