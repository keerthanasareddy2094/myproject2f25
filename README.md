# Internship Onboarding (Phase 1) – Student Requirements Collector

This Streamlit app runs a short, adaptive interview (up to **10 questions**) to understand the student’s interests, target roles/companies, skills, location, and (optionally) parse their résumé.  
It builds a structured **Student Profile JSON** that can be saved and used in later phases.

## Features
- LLM-guided questions (uses **Ollama** if available) or smart fallback list
- Up to **10** questions total (progress tracked)
- Résumé upload (PDF/DOCX/TXT) + optional LLM parsing
- Instant **Profile JSON** summary at the end
- Clean, consistent UI

---

## Run locally (no Docker)

### 1) Install Python deps
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
