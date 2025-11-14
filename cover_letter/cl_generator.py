# cl_generator.py
# Generates a tailored cover letter using an LLM (Ollama via LangChain).
# Robust job-text extraction: Playwright first, then requests+BeautifulSoup fallback.
# If LLM is unavailable, falls back to a clean, ATS-friendly template letter.

from __future__ import annotations
from typing import Dict, Optional
import os
import json
import streamlit as st  # for accessing session resume_json (optional)

def _fetch_job_text_via_playwright(url: str) -> str:
    """Try to pull readable text from the job page using PlaywrightFetcher."""
    if not url:
        return ""
    try:
        from playwright_fetcher import PlaywrightFetcher
        fetcher = PlaywrightFetcher()
        html = fetcher.fetch_html(url) or ""
        text, _ = fetcher.extract_text_and_links(html, url)
        return (text or "")[:8000]
    except Exception:
        return ""

def _fetch_job_text_fallback(url: str) -> str:
    """
    Fallback when Playwright isn't available.
    Uses requests + BeautifulSoup (if installed). Safe to no-op if libs missing.
    """
    if not url:
        return ""
    try:
        import re
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)

        # Basic cleanup
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text[:8000]
    except Exception:
        return ""

def _ollama_cover_letter(profile: Dict[str, str], resume_text: str, job_text: str) -> Optional[str]:
    """
    Try to generate with LangChain + Ollama. Return None if unavailable or fails.
    Includes resume_json from session state if present.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
    except Exception:
        return None

    sys_msg = (
    "You are a professional career assistant. Write a one-page, ATS-friendly cover letter for a job application. "
    "The letter should:\n"
    "- Address the hiring manager and mention the company/organization by name (if 'company' is provided or can be inferred from the job link or description).\n"
    "- Clearly state the applicant's interest in the specific role at the company and explain why the applicant is a strong fit (drawing only from the provided profile, resume, and job description fields).\n"
    "- Summarize the candidate’s most relevant experience and skills in a narrative style (do NOT list resume sections, bullet points, or copy section headers).\n"
    "- Reference the company's mission, values, or notable projects ONLY IF the job description or provided input includes them.\n"
    "- Maintain a positive, concise, and confident tone throughout (no generic clichés, no filler).\n"
    "- Close politely and with enthusiasm for the opportunity.\n"
    "Strict rules:\n"
    "- Do NOT copy or list raw resume text or section names (e.g., 'EXPERIENCE', 'PROJECTS').\n"
    "- Do NOT invent or use placeholder text for missing fields (just omit them).\n"
    "- Only mention factual elements supported by the user's real data or the job/company description.\n"
    "- Return ONLY the actual letter as continuous text (no markdown, no bullet points, no JSON, no YAML, no extra explanations).\n"
    )


    # --- Enhancement: include parsed resume JSON if available ---
    user_blob = {
        "profile": profile,
        "resume_excerpt": (resume_text or "")[:8000],
        "resume_json": st.session_state.get("resume_json") or {},  # <--- parsed JSON from your pipeline
        "job_excerpt": job_text,
    }

    model = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
    base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    try:
        llm = ChatOllama(base_url=base_url, model=model, temperature=0.2)
        prompt = ChatPromptTemplate.from_messages([("system", sys_msg), ("human", "{data}")])
        chain = prompt | llm
        out = chain.invoke({"data": json.dumps(user_blob, ensure_ascii=False)})
        content = getattr(out, "content", None)
        content = (content or "").strip()
        return content if content else None
    except Exception:
        return None

def _template_fallback(profile: Dict[str, str], resume_text: str, job_text: str) -> str:
    """Simple, clean fallback letter if the LLM path isn't available."""
    name = profile.get("full_name")
    email = profile.get("email")
    phone = profile.get("phone")
    city = profile.get("city", "")
    target = profile.get("role_interest")
    highlights = profile.get("highlights")
    extras = profile.get("extras")

    bullets = ""
    if highlights:
        items = [x.strip(" •-") for x in highlights.replace("\n", ";").split(";") if x.strip()]
        if items:
            bullets = "\n".join(f"• {it}" for it in items[:4])

    lines = []
    lines.append(f"{name}")
    if city:
        lines.append(city)
    lines.append(f"{email} | {phone}")
    lines.append("")
    lines.append("Dear Hiring Manager,")
    lines.append("")
    lines.append(f"I am excited to apply for {target}. With my background and experience, I can contribute immediately to your team.")
    if job_text:
        lines.append("From the job description, several requirements align with my experience.")
    if bullets:
        lines.append("")
        lines.append("Highlights:")
        lines.append(bullets)
    if extras:
        lines.append("")
        lines.append(extras)
    lines.append("")
    lines.append("Thank you for your time and consideration. I would welcome the opportunity to discuss how my skills align with your needs.")
    lines.append("")
    lines.append("Sincerely,")
    lines.append(name)
    return "\n".join(lines)

def make_cover_letter(profile: Dict[str, str], resume_text: str, target_url: str) -> str:
    """
    Fetch job text (Playwright -> requests/BS fallback),
    then attempt LLM generation, fallback to template.
    """
    # --- JOB TEXT ---
    job_text = _fetch_job_text_via_playwright(target_url.strip() if target_url else "")
    if not job_text and target_url:
        job_text = _fetch_job_text_fallback(target_url)

    # --- LLM ---
    generated = _ollama_cover_letter(profile, resume_text or "", job_text)
    if generated:
        return generated

    # --- Fallback template ---
    return _template_fallback(profile, resume_text or "", job_text)
