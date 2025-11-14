
"""
backend_navigator.py
LLM-guided web navigation to find internship listings.
Run with: uvicorn backend_navigator:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import re
import json
import asyncio
from typing import Optional, Dict, List
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from playwright_fetcher import PlaywrightFetcher


# ============================================================================
# FASTAPI APP SETUP
# ============================================================================
app = FastAPI(
    title="Internship Navigator",
    description="LLM-guided web navigator to find internship job listings",
    version="1.0.0"
)


# ============================================================================
# NAVIGATOR CLASS
# ============================================================================
class LLMNavigator:
    """
    LLM-directed browser that navigates career pages to find internship links.
    Uses Playwright for JavaScript-heavy sites.
    """
    
    def __init__(self, max_hops: int = 5):
        self.max_hops = max_hops
        self.fetcher = PlaywrightFetcher(timeout_ms=15000, wait_ms=2000)
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model_name = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
        self.visited_urls = set()
        print(f"Backend initialized - Ollama host: {self.ollama_host}, Model: {self.model_name}")
        
    async def fetch_url_async(self, url: str) -> Optional[str]:
        """Fetch URL using Playwright asynchronously."""
        html = await self.fetcher.fetch_html(url)
        return html
    
    def fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL using Playwright (sync wrapper)."""
        try:
            return asyncio.run(self.fetch_url_async(url))
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_text_and_links(self, html: str, base_url: str):
        """Extract readable text and links from HTML using PlaywrightFetcher."""
        return self.fetcher.extract_text_and_links(html, base_url)
    
    def get_llm_navigation_decision(self, 
                                   page_content: str, 
                                   available_links: List[Dict],
                                   user_query: str,
                                   current_url: str,
                                   hop_count: int) -> Dict:
        """
        Ask LLM which URL to visit next based on page content.
        Returns: {"action": "visit_url" | "found_jobs" | "stop", "url": "...", "reasoning": "..."}
        """
        
        llm = ChatOllama(
            base_url=self.ollama_host,
            model=self.model_name,
            temperature=0.3,
            streaming=False,
            model_kwargs={"num_ctx": 4096, "num_predict": 300}
        )
        
        # Format links with better context
        links_text = "\n".join([
            f"{i+1}. {link.get('text', 'No text')[:80]} -> {link.get('url', 'No URL')}"
            for i, link in enumerate(available_links[:20])
        ])
        
        sys_prompt = """You are a web navigator helping find internship job listings.

CURRENT TASK: Analyze the page and decide the next action.

ACTIONS:
1. "found_jobs" - Use this if you see ACTUAL job/internship listings on the current page
   - Signs: multiple job titles, apply buttons, job descriptions, search results with positions
   
2. "visit_url" - Use this to navigate to a more specific page
   - Look for links with: "careers", "jobs", "internships", "students", "university", "positions", "search jobs"
   - Prefer links that seem to lead to job listings, not general info pages
   
3. "stop" - Use this if stuck or no relevant links found

IMPORTANT:
- If you see job/internship listings on THIS page, say "found_jobs"
- The current page content preview will show you what's on the page
- Choose the MOST SPECIFIC link available (e.g., "Internships" over "Careers")

Return ONLY valid JSON:
{{"action": "found_jobs|visit_url|stop", "url": "exact_url_from_list", "reasoning": "brief explanation"}}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", """Current URL: {current_url}
Hop: {hop_count}/{max_hops}
Query: {query}

PAGE CONTENT PREVIEW (first 2000 chars):
{content}

AVAILABLE LINKS:
{links}

Decision (JSON only):""")
        ])
        
        chain = prompt | llm
        
        try:
            response = chain.invoke({
                "current_url": current_url,
                "hop_count": hop_count,
                "max_hops": self.max_hops,
                "query": user_query,
                "content": page_content[:2000],
                "links": links_text
            })
            
            print(f"  LLM raw response: {response.content[:200]}")
            
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*?\}', response.content)
            if json_match:
                decision = json.loads(json_match.group(0))
                print(f"  Parsed decision: {decision}")
                return decision
        except Exception as e:
            print(f"LLM error: {e}")
        
        return {"action": "stop", "reasoning": "LLM decision error"}
    
    def has_job_listings(self, html: str, text: str) -> bool:
        """
        Heuristic check if page has job listings.
        Returns True if page appears to have job postings.
        """
        text_lower = text.lower()
        
        # Strong indicators of job listings
        job_indicators = [
            'apply now', 'apply for', 'view job', 'job description',
            'posted date', 'job id', 'requisition', 'position id',
            'salary range', 'job type', 'employment type',
            'intern - ', 'internship - ', 'software engineer intern',
            'data analyst intern', 'product manager intern',
            'search jobs', 'current openings', 'view all jobs',
            'filter jobs', 'job search', 'careers at'
        ]
        
        indicator_count = sum(1 for ind in job_indicators if ind in text_lower)
        
        # Check HTML structure
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        
        # Look for job listing patterns
        job_cards = soup.find_all(['div', 'article', 'li'], class_=re.compile(r'job|position|listing|opening|career', re.I))
        
        # Look for links with job/internship keywords
        job_links = soup.find_all('a', href=re.compile(r'job|career|position|intern|opening', re.I))
        
        print(f"  Heuristic check:")
        print(f"    - Job indicators in text: {indicator_count}")
        print(f"    - Job card elements: {len(job_cards)}")
        print(f"    - Job-related links: {len(job_links)}")
        
        # More lenient - if we have ANY indication this might have jobs
        return indicator_count >= 1 or len(job_cards) >= 2 or len(job_links) >= 3
    
    def navigate_to_jobs(self, start_url: str, user_query: str) -> tuple[Optional[str], List[Dict]]:
        """
        Navigate from start URL to find job listings.
        Returns: (final_url, list_of_found_links)
        """
        print(f"\n{'='*60}")
        print(f"Starting navigation from: {start_url}")
        print(f"Query: {user_query}")
        print(f"{'='*60}")
        
        current_url = start_url
        hop_count = 0
        self.visited_urls.add(current_url)
        
        while hop_count < self.max_hops:
            hop_count += 1
            print(f"\n[Hop {hop_count}/{self.max_hops}] Visiting: {current_url}")
            
            # Fetch page
            html = self.fetch_url(current_url)
            if not html:
                print("  ✗ Failed to fetch page")
                return None, []
            
            print(f"  ✓ Fetched {len(html)} bytes")
            
            # Extract content and links
            page_text, links = self.extract_text_and_links(html, current_url)
            print(f"  ✓ Extracted {len(page_text)} chars text, {len(links)} links")
            
            if not links:
                print("  ✗ No links found on page")
                # But check if THIS page has job listings
                if self.has_job_listings(html, page_text):
                    print("  ✓ Current page appears to have job listings!")
                    return current_url, links
                return None, []
            
            # Quick heuristic check before asking LLM
            has_listings = self.has_job_listings(html, page_text)
            if has_listings:
                print("  ✓ Heuristic check: Page has job listings!")
                # Make sure we return links even if the list is empty
                if not links:
                    print("  ! Warning: Heuristic found jobs but no links extracted")
                    # Extract links more aggressively
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'lxml')
                    links = []
                    for a in soup.find_all('a', href=True, limit=50):
                        href = a.get('href', '').strip()
                        text = a.get_text(strip=True)[:100]
                        if href and text:
                            full_url = urljoin(current_url, href)
                            links.append({'text': text, 'url': full_url})
                    print(f"  → Extracted {len(links)} links aggressively")
                return current_url, links
            
            # Ask LLM what to do
            decision = self.get_llm_navigation_decision(
                page_text,
                links,
                user_query,
                current_url,
                hop_count
            )
            
            action = decision.get('action', 'stop')
            reasoning = decision.get('reasoning', 'No reason provided')
            print(f"  Decision: {action} - {reasoning}")
            
            if action == "found_jobs":
                print("  ✓ LLM confirmed: Job listings found!")
                return current_url, links
            
            elif action == "visit_url":
                next_url = decision.get("url", "").strip()
                
                if not next_url:
                    print("  ✗ No URL provided in decision")
                    return None, []
                
                # Normalize URL
                if not next_url.startswith('http'):
                    next_url = urljoin(current_url, next_url)
                
                if next_url in self.visited_urls:
                    print(f"  ✗ Already visited: {next_url}")
                    # Try to find an alternative link
                    for link in links[:5]:
                        alt_url = link.get('url', '')
                        if alt_url and alt_url not in self.visited_urls:
                            next_url = alt_url
                            print(f"  → Trying alternative: {next_url}")
                            break
                    else:
                        return None, []
                
                self.visited_urls.add(next_url)
                current_url = next_url
            
            else:  # stop
                print("  ✗ Navigation stopped by LLM")
                return None, []
        
        print(f"\n✗ Max hops ({self.max_hops}) reached")
        # Return last page's links anyway
        return current_url, links


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class NavigationRequest(BaseModel):
    start_url: str
    query: str
    max_hops: int = 5


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Internship Navigator"}


@app.post("/navigate")
async def navigate(request: NavigationRequest):
    """
    Navigate to find internship jobs from a starting URL.
    
    The LLM will intelligently follow links on career pages until it finds
    job listings or reaches the maximum hop limit.
    """
    try:
        navigator = LLMNavigator(max_hops=request.max_hops)
        
        final_url, found_links = navigator.navigate_to_jobs(
            request.start_url,
            request.query
        )
        
        success = final_url is not None
        
        print(f"\n{'='*60}")
        print(f"Navigation complete:")
        print(f"  Success: {success}")
        print(f"  Final URL: {final_url}")
        print(f"  Links found: {len(found_links) if found_links else 0}")
        print(f"  Visited URLs: {len(navigator.visited_urls)}")
        print(f"{'='*60}\n")
        
        return {
            "success": success,
            "final_url": final_url or request.start_url,  # Return original URL if navigation failed
            "visited_urls": list(navigator.visited_urls),
            "found_links": found_links[:50] if found_links else [],
            "total_hops": len(navigator.visited_urls)
        }
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"Navigation error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'!'*60}\n")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch")
async def fetch_url(url: str):
    """
    Fetch and return structured data from a URL.
    
    Returns: text preview, extracted links, and metadata.
    """
    try:
        navigator = LLMNavigator()
        html = navigator.fetch_url(url)
        
        if not html:
            raise HTTPException(status_code=400, detail="Could not fetch URL")
        
        # Return summary instead of full HTML
        text, links = navigator.extract_text_and_links(html, url)
        
        return {
            "url": url,
            "text_preview": text[:1000],
            "links_count": len(links),
            "links": links[:30]
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API information."""
    return {
        "service": "Internship Navigator Backend",
        "version": "1.0.0",
        "description": "LLM-guided web navigation for finding internship job listings",
        "endpoints": {
            "health": "GET /health",
            "navigate": "POST /navigate",
            "fetch": "POST /fetch"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
