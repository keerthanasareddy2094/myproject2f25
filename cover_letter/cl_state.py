# cl_state.py
# Manages Streamlit session state and question list for the cover-letter flow.

from __future__ import annotations
from typing import Dict, List, Tuple
import streamlit as st

# Questions the LLM will ask to collect cover-letter details.
COVER_QUESTIONS: List[Tuple[str, str]] = [
    ("full_name", "What’s your full name?"),
    ("email", "What’s your email address?"),
    ("phone", "What’s your phone number?"),
    ("city", "Which city are you based in (or applying from)?"),
    ("role_interest", "Which role/link should I target (paste the URL, the title, or just the company name)?"),
    ("highlights", "Share 2–4 bullet highlights you want emphasized."),
    ("extras", "Anything else I should include (e.g., relocation, graduation date)?"),
]

def init_cover_state() -> None:
    """Initialize Streamlit session state for cover-letter flow if missing."""
    st.session_state.setdefault("want_cover_letter", None)         # None | True | False
    st.session_state.setdefault("collecting_cover_profile", False)
    st.session_state.setdefault("cover_profile", {})               # Dict[str, str]
    st.session_state.setdefault("cover_target_url", "")
    st.session_state.setdefault("resume_text", "")                 # Filled via your resume manager/parser
    st.session_state.setdefault("resume_json", {})                 # Optional structured JSON from your parser
    st.session_state.setdefault("generated_cover_letters", [])     # List[dict]
    st.session_state.setdefault("last_results_df", None)           # DataFrame from search flow

    # NEW: single-convo orchestration flags
    st.session_state.setdefault("asked_for_resume", False)
    st.session_state.setdefault("pending_company_query", "")   # holds company text while we fetch links
    st.session_state.setdefault("auto_pick_first_match", True) # auto-pick first row when we just need one

def get_profile() -> Dict[str, str]:
    """Return the current cover-letter profile answers."""
    return st.session_state.get("cover_profile", {})

def set_profile_field(key: str, value: str) -> None:
    """Set a single profile field."""
    st.session_state["cover_profile"][key] = (value or "").strip()

def set_target_url(url: str) -> None:
    """Store the target job URL and mirror it into role_interest for LLM context."""
    st.session_state["cover_target_url"] = (url or "").strip()
    st.session_state["cover_profile"]["role_interest"] = st.session_state["cover_target_url"]

def next_unanswered_key() -> str | None:
    """Return the next missing profile key, or None if all answered."""
    profile = get_profile()
    for key, _q in COVER_QUESTIONS:
        if not (profile.get(key) or "").strip():
            return key
    return None

def reset_cover_state(clear_profile: bool = True) -> None:
    """Reset cover-letter conversation state."""
    st.session_state["want_cover_letter"] = None
    st.session_state["collecting_cover_profile"] = False
    if clear_profile:
        st.session_state["cover_profile"] = {}
    st.session_state["cover_target_url"] = ""
    st.session_state["asked_for_resume"] = False
    st.session_state["pending_company_query"] = ""
