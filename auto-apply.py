# auto_apply.py — Greenhouse/Lever auto-apply helper
import asyncio, os, tempfile
from typing import Dict, List
from playwright.async_api import async_playwright

COMMON_TIMEOUT = 20000  # 20s

def _save_temp_file(file_or_bytes, suffix=".pdf") -> str:
    if not file_or_bytes:
        return ""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if hasattr(file_or_bytes, "read"):
        data = file_or_bytes.read()
    else:
        data = file_or_bytes
    with open(path, "wb") as f:
        f.write(data)
    return path

async def _apply_greenhouse(page, url: str, who: Dict, resume_path: str, cover_text: str) -> str:
    await page.goto(url, wait_until="domcontentloaded", timeout=COMMON_TIMEOUT)
    for sel in ['input[name="resume"]','input[type="file"][name*="resume"]','input[type="file"]']:
        try:
            up = page.locator(sel)
            if await up.count():
                await up.set_input_files(resume_path); break
        except: pass
    fields = [
        ("first_name", ['input[name="first_name"]','input[name="firstName"]','input[name~="first"]']),
        ("last_name",  ['input[name="last_name"]','input[name="lastName"]','input[name~="last"]']),
        ("email",      ['input[type="email"]','input[name="email"]','input[name="email_address"]']),
        ("phone",      ['input[type="tel"]','input[name="phone"]','input[name="phone_number"]']),
    ]
    for key, sels in fields:
        val = who.get(key, "") or who.get(key.replace("_name",""))
        for sel in sels:
            try:
                el = page.locator(sel)
                if await el.count():
                    await el.first.fill(str(val)); break
            except: pass
    for sel in ['textarea[name*="cover"]','textarea[name="cover_letter"]','textarea']:
        try:
            t = page.locator(sel)
            if await t.count():
                await t.first.fill(cover_text[:4000]); break
        except: pass
    for sel in ['button[type="submit"]','input[type="submit"]','button:has-text("Submit")','button:has-text("Apply")']:
        try:
            b = page.locator(sel)
            if await b.count():
                await b.first.click(); await page.wait_for_timeout(2000); break
        except: pass
    return page.url

async def _apply_lever(page, url: str, who: Dict, resume_path: str, cover_text: str) -> str:
    await page.goto(url, wait_until="domcontentloaded", timeout=COMMON_TIMEOUT)
    for sel in ['input[type="file"][name="resume"]','input[type="file"]']:
        try:
            up = page.locator(sel)
            if await up.count():
                await up.set_input_files(resume_path); break
        except: pass
    fields = [
        ("first_name", ['input[name="name.first"]','input[name="firstName"]','input[placeholder*="First"]']),
        ("last_name",  ['input[name="name.last"]','input[name="lastName"]','input[placeholder*="Last"]']),
        ("email",      ['input[type="email"]','input[name="email"]']),
        ("phone",      ['input[type="tel"]','input[name="phone"]']),
    ]
    for key, sels in fields:
        val = who.get(key, "")
        for sel in sels:
            try:
                el = page.locator(sel)
                if await el.count():
                    await el.first.fill(str(val)); break
            except: pass
    for sel in ['textarea[name="coverLetterText"]','textarea[name*="cover"]','textarea']:
        try:
            t = page.locator(sel)
            if await t.count():
                await t.first.fill(cover_text[:4000]); break
        except: pass
    for sel in ['button[type="submit"]','button:has-text("Submit")','button:has-text("Apply")']:
        try:
            b = page.locator(sel)
            if await b.count():
                await b.first.click(); await page.wait_for_timeout(2000); break
        except: pass
    return page.url

def _platform_for(url: str) -> str:
    u = url.lower()
    if "greenhouse.io" in u: return "greenhouse"
    if "lever.co" in u: return "lever"
    return "other"

async def _apply_one(browser, url: str, who: Dict, resume_path: str, cover_text: str):
    ctx = await browser.new_context()
    page = await ctx.new_page()
    platform = _platform_for(url)
    try:
        if platform == "greenhouse":
            dest = await _apply_greenhouse(page, url, who, resume_path, cover_text)
            status = "submitted_or_attempted"
        elif platform == "lever":
            dest = await _apply_lever(page, url, who, resume_path, cover_text)
            status = "submitted_or_attempted"
        else:
            status = "manual_required"; dest = url
    except Exception as e:
        status = f"error: {e}"; dest = url
    finally:
        await ctx.close()
    return {"platform": platform, "status": status, "final_url": dest}

async def auto_apply_batch(selected: List[Dict], who: Dict, resume_bytes: bytes, cover_text: str) -> List[Dict]:
    resume_path = _save_temp_file(resume_bytes, suffix=".pdf")
    out = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
        for item in selected:
            url = item.get("link","")
            res = await _apply_one(browser, url, who, resume_path, cover_text)
            res.update({"title": item.get("title",""), "company": item.get("company",""), "url": url})
            out.append(res)
        await browser.close()
    try:
        if resume_path and os.path.exists(resume_path): os.remove(resume_path)
    except: pass
    return out
