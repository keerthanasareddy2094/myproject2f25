"""
main.py — Minimal FastAPI server for CSUSB CSE links only.

What it does:
- GET /healthz        -> simple health info (and cached link count if available)
- GET /csusb/links    -> returns cached links from scrape_csusb_listings(deep=False)

Removed:
- All LLM/Ollama/langchain-related endpoints (/chat, /chat/complete, /model/info)
- Rate limiting, streaming, and any navigator/LLM logic
"""

from __future__ import annotations

import os
import time
import asyncio
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Local scraper (must exist in the same project directory)
from scraper import scrape_csusb_listings


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="CSUSB CSE Links API",
    description="Returns links extracted from CSUSB CSE pages (no deep search).",
    version="1.0.0",
)

# In production, constrain origins as needed
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:5002",
    "http://127.0.0.1:5002",
    "https://sec.cse.csusb.edu",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Simple in-memory cache for the dataframe
# -----------------------------------------------------------------------------
CACHE_TTL = int(os.getenv("CSUSB_CACHE_TTL", "3600"))  # seconds
_cache: Dict[str, Any] = {"at": 0.0, "df": None}


def _scrape_df() -> pd.DataFrame:
    """
    Synchronous scrape for CSUSB CSE links only.
    scraper.scrape_csusb_listings() should already be configured to avoid deep crawling.
    """
    df = scrape_csusb_listings(deep=False, max_pages=1)
    # Ensure expected columns exist
    expected = ["link", "title", "company", "host", "source", "posted_date"]
    for c in expected:
        if c not in df.columns:
            df[c] = None
    return df[expected].drop_duplicates(subset=["link"], keep="first")


async def _get_df(force: bool = False) -> pd.DataFrame:
    """
    Get cached DataFrame; refresh if stale or force=True.
    """
    now = time.time()
    if not force and _cache["df"] is not None and (now - _cache["at"] < CACHE_TTL):
        return _cache["df"]

    # Run the sync scrape in a worker thread (so we don't block the event loop)
    df = await asyncio.to_thread(_scrape_df)
    _cache["df"] = df
    _cache["at"] = time.time()
    return df


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/healthz", tags=["Health"])
async def healthz():
    """
    Basic health with cache info.
    """
    try:
        df = _cache["df"]
        count = int(len(df)) if df is not None else 0
        age = time.time() - float(_cache["at"]) if _cache["at"] else None
        return {
            "status": "ok",
            "cached_count": count,
            "cache_age_seconds": age,
            "cache_ttl_seconds": CACHE_TTL,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health error: {e}")


@app.get("/csusb/links", tags=["Links"])
async def csusb_links(
    refresh: bool = Query(False, description="If true, bypass cache and rescrape now."),
):
    """
    Return CSUSB CSE links as JSON (cached).
    """
    try:
        df = await _get_df(force=refresh)
        items = df.to_dict(orient="records")
        return {
            "count": len(items),
            "cached_at_unix": _cache["at"],
            "cache_ttl_seconds": CACHE_TTL,
            "items": items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scrape error: {e}")


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "CSUSB CSE Links API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/healthz",
            "links": "/csusb/links",
        },
        "notes": "This API only returns links from CSUSB CSE pages (no deep external navigation).",
    }
