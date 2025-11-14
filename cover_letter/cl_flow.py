from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional
import json
import os
import time
import re
import streamlit as st
import langchain_ollama as ChatOllama


from .cl_state import (
    init_cover_state, get_profile, set_profile_field,
    set_target_url
)
from .cl_generator import make_cover_letter

def _results_preview(df) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        if df is None or df.empty:
            return out
        cols = [c for c in ["title", "company", "link", "url"] if c in df.columns]
        for _, row in df.head(12).iterrows():
            item = {k: str(row.get(k, "")) for k in cols}
            if "url" in item and not item.get("link"):
                item["link"] = item.pop("url")
            out.append(item)
    except Exception:
        pass
    return out

def _llm():
    try:
        from langchain_ollama import ChatOllama
    except Exception:
        return None
    base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
    try:
        return ChatOllama(base_url=base, model=model, temperature=0.2, streaming=False)
    except Exception:
        return None

def _default_render(role: str, content: str) -> None:
    with st.chat_message(role):
        st.write(content)

def _plan_next_step(user_msg: str) -> Dict[str, Any]:
    profile = get_profile()
    resume_text = st.session_state.get("resume_text", "")
    resume_json = st.session_state.get("resume_json", {})
    target_url = st.session_state.get("cover_target_url", "")
    results_df = st.session_state.get("last_results_df")
    results = _results_preview(results_df)
    collecting = bool(st.session_state.get("collecting_cover_profile"))
    print("DEBUG: CL PLANNER profile:", profile)
    print("DEBUG: CL PLANNER target_url:", target_url)


    # PROACTIVE: auto-match company/title to link if not already set
    patched = False
    if not target_url and user_msg:
        search_name = user_msg.lower().strip()
        if results_df is not None and len(results_df) > 0:
            matches = results_df[
                results_df['company'].astype(str).str.lower().str.contains(search_name, na=False) |
                results_df['title'].astype(str).str.lower().str.contains(search_name, na=False)
            ]
            if len(matches) == 1:
                url_col = "link" if "link" in matches.columns else ("url" if "url" in matches.columns else None)
                if url_col:
                    url = matches.iloc[0][url_col]
                    set_target_url(str(url))
                    profile = get_profile() # update
                    target_url = url
                    patched = True

    planner = _llm()
    def _fallback() -> Dict[str, Any]:
        if not target_url:
            return {"action": "ask", "field": "role_interest",
                    "question": "Please paste the job link, or tell me a company/title to target."}
        # If user disables LLM or error: ask for each non-filled profile field (else generate)
        for k in ["full_name", "email", "phone", "city"]:
            if not (profile.get(k) or "").strip():
                return {"action": "ask", "field": k, "question": f"Please share your {k.replace('_', ' ')}."}
        return {"action": "generate"}

    sys = (
        "You are a proactive assistant that manages a cover-letter workflow. "
        "You must return ONLY a single compact JSON object with an action (no prose). "
        "Available actions:\n"
        '  {"action":"ask","field":"<full_name|email|phone|city|highlights|extras|role_interest>","question":"..."}\n'
        '  {"action":"set","field":"...","value":"..."}\n'
        '  {"action":"set_url","url":"https://..."}\n'
        '  {"action":"fetch_company","company":"google"}\n'
        '  {"action":"answer","text":"..."}\n'
        '  {"action":"generate"}\n'
        "If the user mentioned a company/title, and a unique result exists in 'results', select that as the job target. "
        "Only ask about other profile details if missing from resume/profile. "
        "Whenever ready, proceed directly to generation in minimal steps without repeats."
    )
    blob = {
        "last_user_msg": user_msg or "",
        "collecting": collecting,
        "profile": profile,
        "resume_present": bool((resume_text or "").strip()),
        "resume_json_keys": list(resume_json.keys()) if isinstance(resume_json, dict) else [],
        "results": results,
        "target_url": target_url,
    }
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys),
        ("human", "{blob}")
    ])
    try:
        out = (prompt | planner).invoke({"blob": json.dumps(blob, ensure_ascii=False)})
        txt = (getattr(out, "content", "") or "").strip()
        m = re.search(r"\{[\s\S]*\}", txt)
        js = json.loads(m.group(0) if m else txt)
        if not isinstance(js, dict):
            raise ValueError("Planner did not return a JSON object")
        if js.get("action") not in {"ask", "set", "set_url", "fetch_company", "answer", "generate"}:
            raise ValueError("Planner returned unknown action")
        return js
    except Exception:
        return _fallback()

def offer_cover_letter(render: Callable[[str, str], None] = _default_render) -> None:
    init_cover_state()
    if st.session_state.get("want_cover_letter") is None:
        st.session_state["want_cover_letter"] = True
        render("assistant", "Want me to draft a tailored cover letter? I’ll ask a couple of quick questions, use your resume, and generate a download.")

def start_collection(render: Callable[[str, str], None] = _default_render) -> None:
    init_cover_state()
    st.session_state["collecting_cover_profile"] = True

    if not (st.session_state.get("resume_text") or "").strip() and not st.session_state.get("asked_for_resume"):
        st.session_state["asked_for_resume"] = True
        render("assistant", "Please upload your resume (PDF/DOCX/TXT) in the left sidebar, then say “done”.")
        return

    df = st.session_state.get("last_results_df")
    if df is not None and len(df) == 1:
        url_col = "link" if "link" in df.columns else ("url" if "url" in df.columns else None)
        if url_col:
            set_target_url(str(df.iloc[0][url_col]))

    _drive_once("", render)

def ask_next_question(render: Callable[[str, str], None] = _default_render) -> None:
    _drive_once("", render)

def handle_user_message(message_text: str, render: Callable[[str, str], None] = _default_render) -> bool:
    init_cover_state()
    msg = (message_text or "").strip()
    low = msg.lower()

    # Allow row selection directly
    if low.startswith("select row ") or low.startswith("row "):
        try:
            idx = int(low.split()[-1])
            df = st.session_state.get("last_results_df")
            if df is not None and 0 <= idx < len(df):
                url_col = "link" if "link" in df.columns else ("url" if "url" in df.columns else None)
                if url_col:
                    set_target_url(str(df.iloc[idx][url_col]))
                    st.session_state["collecting_cover_profile"] = True
                    _drive_once("", render)
                    return True
        except Exception:
            pass

    if st.session_state.get("want_cover_letter") and not st.session_state.get("collecting_cover_profile"):
        if any(w in low for w in ["yes", "yep", "sure", "ok", "okay", "please", "start", "begin", "create", "make one", "draft"]):
            st.session_state["collecting_cover_profile"] = True
            _drive_once(msg, render)
            return True
        if low.startswith("http://") or low.startswith("https://"):
            set_target_url(msg)
            st.session_state["collecting_cover_profile"] = True
            _drive_once("", render)
            return True

    if st.session_state.get("collecting_cover_profile"):
        if low in {"done", "uploaded", "i uploaded", "resume uploaded"}:
            if (st.session_state.get("resume_text") or "").strip():
                _drive_once("", render)
            else:
                render("assistant", "I still don’t see a resume. Please upload it in the left sidebar and then say “done”.")
            return True
        _drive_once(msg, render)
        return True

    return False

def _drive_once(user_msg: str, render: Callable[[str, str], None]) -> None:
    if not (st.session_state.get("resume_text") or "").strip() and st.session_state.get("asked_for_resume"):
        render("assistant", "Once your resume is uploaded in the sidebar, just type “done”.")
        return

    step = _plan_next_step(user_msg)
    act = step.get("action")

    if act == "ask" and step.get("field") == "city":
        msg = user_msg.strip()
        if msg and not get_profile().get("city"):
            set_profile_field("city", msg)
            print("DEBUG: Emergency fallback: city set to", msg)
            print("DEBUG: profile after city fallback:", get_profile())
            # Now re-invoke planner to progress
            step = _plan_next_step("")
            act = step.get("action")
    if act == "answer":
        txt = (step.get("text") or "").strip() or "Here’s what I recommend."
        render("assistant", txt)
        st.session_state.messages.append({"role": "assistant", "content": txt})

        step = _plan_next_step("")

        act = step.get("action")

    if act == "ask":
        q = (step.get("question") or "").strip() or "Please share that detail."
        render("assistant", q)
        st.session_state.messages.append({"role": "assistant", "content": q})
        return

    if act == "set":
        field = (step.get("field") or "").strip()
        value = (step.get("value") or "").strip()
        if field and value:
            set_profile_field(field, value)
            print("DEBUG: set field", field, "=", value)
        _drive_once("", render)
        return

    if act == "set_url":
        url = (step.get("url") or "").strip()
        if url:
            set_target_url(url)
        _drive_once("", render)
        return

    if act == "fetch_company":
        company = (step.get("company") or "").strip()
        if len(company) >= 3:
            st.session_state["pending_company_query"] = company
            render("assistant", f"Got it — I’ll pull roles for **{company}** and then continue.")
        else:
            render("assistant", "Please paste the job link or share a company/title (at least 3 characters).")
        return

    if act == "generate":
        _generate_and_show_letter(render)
        return

    render("assistant", "Please paste the job link, or tell me a company/title to target.")
    st.session_state.messages.append({"role": "assistant", "content": "Please paste the job link, or tell me a company/title to target."})

def _generate_and_show_letter(render: Callable[[str, str], None]) -> None:
    profile: Dict[str, str] = get_profile()
    target_url = profile.get("role_interest") or st.session_state.get("cover_target_url") or ""
    resume_text = st.session_state.get("resume_text", "")

    with st.spinner("💡 Generating your cover letter, please wait..."):
        letter = make_cover_letter(profile=profile, resume_text=resume_text, target_url=target_url)
        # Optional: add time.sleep(2) for demo, otherwise not needed

    record = {
        "ts": int(time.time()),
        "target": target_url,
        "text": letter,
        "profile": dict(profile),
    }
    st.session_state.setdefault("generated_cover_letters", []).append(record)

    render("assistant", "Here’s your tailored cover letter:\n\n" + letter)
    st.session_state.messages.append({"role": "assistant", "content": "Here’s your tailored cover letter:\n\n" + letter})

    _show_download(record)
    st.session_state["collecting_cover_profile"] = False


def _show_download(record: Dict[str, str]) -> None:
    try:
        import io
        buf = io.BytesIO(record["text"].encode("utf-8"))
        fname = f"cover_letter_{record['ts']}.txt"
        st.download_button(
            label="Download Cover Letter (.txt)",
            data=buf,
            file_name=fname,
            mime="text/plain",
            use_container_width=True,
        )
    except Exception:
        pass
