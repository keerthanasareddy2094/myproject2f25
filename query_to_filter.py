import os, re, json
from typing import Dict, Any

# ============================================================
# CONFIGURATION
# ============================================================
# Toggle whether to use Ollama for natural-language parsing.
USE_OLLAMA = True
# The Ollama API endpoint (by default local service).
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# Default model name to use for parsing/filter extraction.
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:0.5b")

# ============================================================
# BASIC KEYWORD SETS
# ============================================================

# Words to ignore when extracting tokens.
GENERIC_STOP = {
    # General function / pronouns / filler words
    "i","you","your","yours","me","my","mine","we","our","ours","they","them","their","theirs",
    "this","that","these","those","it","its","is","am","are","was","were","be","being","been",
    "do","does","did","a","an","the","and","or","but","if","then","else","than","not","no","yes",
    "please","hi","hello","hey","how","what","who","where","when","why","which","name","age",
    "u","yo","sup","thanks","thank","thankyou",

    # Common job-search filler terms (not useful as keywords)
    "intern","interns","internship","internships","job","jobs","career","careers",
    "opening","openings","position","positions","apply","application","role","roles",
    "only","strict","exact","just","show","list","give","find",
    "in","at","for","from","to","csusb","cse","website","site","listed"
}

# Common technical skills to recognize directly (used for fallback extraction)
TECH_SKILLS = {
    "java","python","c++","c#","javascript","typescript","go","rust","kotlin","swift","r","matlab","sql",
    "react","angular","vue","node","express","django","flask","fastapi","spring","spring boot",".net","asp.net",
    "pandas","numpy","pytorch","tensorflow","scikit-learn","spark","hadoop","tableau","power bi",
    "selenium","cypress","playwright","pytest","junit","postman",
    "aws","azure","gcp","docker","kubernetes","terraform","linux","bash","git","jira",
    "mysql","postgresql","mongodb","redis"
}

# Simple greetings used to detect small-talk vs internship queries.
GREETINGS = {
    "hi","hello","hey","how are you","good morning","good afternoon","good evening",
    "what is your name","your name","who are you","help","thanks","thank you",
    "your age","how old are you"
}

# ============================================================
# LLM HELPER FUNCTION
# ============================================================

def _llm_json(sys_msg: str, user: str, num_ctx=2048, num_predict=160, temp=0.1) -> Dict[str, Any]:
    """
    Calls a local Ollama model via LangChain to extract structured JSON.
    Returns {} on any failure so the rest of the app can continue.

    Parameters:
        sys_msg:  The system prompt that defines what to extract.
        user:     The user query text.
        num_ctx:  Context window.
        num_predict: Max tokens to predict.
        temp:     Sampling temperature (low = deterministic).
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate

        # Compose a prompt template with both system and user messages
        tmpl = ChatPromptTemplate.from_messages([("system", sys_msg), ("human", "{q}")])

        # Initialize the Ollama LLM
        llm = ChatOllama(
            base_url=OLLAMA_HOST, model=MODEL_NAME,
            temperature=temp, streaming=False,
            model_kwargs={"num_ctx": num_ctx, "num_predict": num_predict},
        )

        # Invoke the chain with the user text
        out = (tmpl | llm).invoke({"q": user}).content

        # Extract JSON from the response (regex for { ... } structure)
        m = re.search(r"\{[\s\S]*\}", out)
        return json.loads(m.group(0) if m else out)
    except Exception:
        # Fallback to empty dict if LLM fails
        return {}

# ============================================================
# TOKENIZATION (used for non-LLM fallback)
# ============================================================

def _extract_skills_and_keywords(s: str) -> tuple[list[str], list[str]]:
    """
    Extract basic 'skills' and 'keywords' by tokenizing locally.
    This is used when LLM is off or unavailable.
    """
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\.\+#\-]{1,}", s)]
    skills, keywords = [], []
    for t in tokens:
        if t in GENERIC_STOP:
            continue
        if t in TECH_SKILLS:
            skills.append(t)
        else:
            keywords.append(t)
    return skills[:6], keywords[:6]

# ============================================================
# MAIN: QUERY PARSER
# ============================================================
def parse_query_to_filter(q: str) -> Dict[str, Any]:
    """
    Turn a free-text user query into a structured filter dictionary.
    Returns a compact dict, e.g.:
    {
      "intent": "internship_search" | "general_question" | "resume_question",
      "company_name": "string",
      "title_keywords": ["token", ...],
      "skills": ["token", ...],
      "show_all": true|false,
      "role_match": "broad" | "strict",
      # location fields are included ONLY if explicitly typed:
      # "city": "...", "state": "...", "country": "...", "zipcode": "...",
      "remote_type": "remote|hybrid|onsite"
    }
    """
    if not q:
        return {}

    s = q.strip()

    # ---- LLM instruction ----
    sys = (
        "You extract job-search filters from a short user query.\n"
        "Return ONLY compact JSON with these keys (omit keys that don't apply):\n"
        "{\n"
        '  "intent": "internship_search|general_question|resume_question",\n'
        '  "company_name": "string",\n'
        '  "title_keywords": ["token", ...],\n'
        '  "skills": ["token", ...],\n'
        '  "show_all": true|false,\n'
        '  "role_match": "broad|strict",\n'
        '  "city": "string",\n'
        '  "state": "string",\n'
        '  "country": "string",\n'
        '  "zipcode": "string",\n'
        '  "remote_type": "remote|hybrid|onsite"\n'
        "}\n"
        "Rules:\n"
        "- Never infer or guess any location fields; include them ONLY if the user explicitly typed them.\n"
        "- Lower-case tokens; keep arrays ≤ 6; avoid nulls; do not invent values."
    )

    # ---- Step 1: LLM extraction (optional) ----
    data: Dict[str, Any] = {}
    if USE_OLLAMA:
        data = _llm_json(sys, s, num_ctx=2048, num_predict=160, temp=0.2)
        if not isinstance(data, dict):
            data = {}

    # ---- Step 2: Local fallbacks / defaults ----
    role_match = "strict" if re.search(r"\b(strict|exact|only|just)\b", s, re.I) else "broad"
    show_all = bool(
        re.search(r"\b(show|list|give|display|fetch|see)\b.*\b(all|every)\b.*\binternship[s]?\b", s, re.I) or
        re.search(r"\b(all|every)\b.*\binternship[s]?\b", s, re.I) or
        re.search(r"\binternship[s]?\b.*\b(all|every|listed)\b", s, re.I) or
        re.search(r"\ball\b.*\blisted\b.*\binternship[s]?\b", s, re.I)
    )
    skills, keywords = _extract_skills_and_keywords(s)

    # Merge fallback into LLM result
    data.setdefault("role_match", role_match)
    data.setdefault("show_all", show_all)
    data.setdefault("title_keywords", keywords)
    data.setdefault("skills", skills)

    # Normalize arrays
    data["title_keywords"] = [t.strip().lower() for t in data.get("title_keywords", [])][:6]
    data["skills"] = [t.strip().lower() for t in data.get("skills", [])][:6]

    # Intent fallback (resume > internship > general)
        # Intent fallback (cover letter > resume > internship > general)
    if not data.get("intent"):
        s_lo = s.lower()
        if re.search(r"\b(cover letter|coverletter|make cover letter|write cover letter|draft cover letter|create cover letter)\b", s_lo):
            data["intent"] = "cover_letter"
        elif re.search(r"\b(resume|résumé|cv|gpa|projects?|experience|education)\b", s_lo):
            data["intent"] = "resume_question"
        elif re.search(r"\bintern(ship|ships)?\b", s_lo) or re.search(
            r"\b(find|show|list|apply|search|available|display|get)\b.*\b(intern|job|role|position|career|opening)\b", s_lo
        ):
            data["intent"] = "internship_search"
        else:
            data["intent"] = "general_question"
    else:
        # clamp to allowed set
        if data["intent"] not in {"internship_search", "general_question", "resume_question", "cover_letter"}:
            data["intent"] = "general_question"


    # Drop location fields unless explicitly present
    explicit_loc = re.search(
        r"\b(remote|onsite|on-site|hybrid|usa|united states|uk|england|canada|india|"
        r"london|new york|ny|ca|tx|\d{5})\b",
        s, re.I
    )
    if not explicit_loc:
        for k in ("city", "state", "country", "zipcode"):
            data.pop(k, None)

    return data

# ============================================================
# INTENT CLASSIFIER
# ============================================================

def classify_intent(q: str) -> str:
    """
    Determine 'internship_search' vs 'resume_question' vs 'general_question'
    with LLM as the source of truth. Falls back only if LLM is unavailable.
    """
    s = (q or "").strip()
    if not s:
        return "general_question"

    if USE_OLLAMA:
        d = _llm_json(
            (
                "Classify the user's message into exactly one of these labels:\n"
                "- internship_search: user wants internships, companies, roles, skills, or terms\n"
                "- resume_question: question about user's résumé content, skills, experiences, education, formatting\n"
                "- general_question: greetings, app usage, anything not related to internships or résumé content\n"
                'Return JSON exactly like {"intent":"internship_search"} or {"intent":"resume_question"} or {"intent":"general_question"}'
            ),
            s,
            num_ctx=512, num_predict=30, temp=0.0
        )
        label = isinstance(d, dict) and d.get("intent")
        if label in {"internship_search", "resume_question", "general_question"}:
            return label

    # Emergency fallback if LLM is off/unreachable
        # Emergency fallback if LLM is off/unreachable
    sl = s.lower()
    if any(k in sl for k in ["cover letter", "coverletter", "make cover letter", "write cover letter", "draft cover letter", "create cover letter"]):
        return "cover_letter"
    if any(k in sl for k in ["resume", "résumé", "cv", "gpa", "projects", "experience"]):
        return "resume_question"
    if "intern" in sl or "career" in sl or "job" in sl:
        return "internship_search"
    return "general_question"
