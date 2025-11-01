from __future__ import annotations
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime
import os, re, time, requests, pandas as pd
from bs4 import BeautifulSoup

CSUSB_CSE_URL="https://www.csusb.edu/cse/internships-careers"
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
USE_PLAYWRIGHT=os.getenv("USE_PLAYWRIGHT","0").lower() in {"1","true","yes"}
TIMEOUT=10

def _clean(s:str)->str: return re.sub(r"\s+"," ",s or "").strip()
def _infer_company(u:str)->str:
    try: host=urlparse(u).netloc.lower(); return host.split(".")[-2].capitalize()
    except: return ""

def _collect(html:str,base:str)->List[Dict]:
    soup=BeautifulSoup(html,"lxml"); main=soup.find("main") or soup
    out=[]; seen=set()
    for a in main.find_all("a",href=True):
        t=_clean(a.get_text(" ",strip=True)); 
        if not t: continue
        link=urljoin(base,a["href"]); host=urlparse(link).netloc.lower()
        if (t.lower(),link) in seen: continue
        if "intern" not in t.lower() and "career" not in link.lower(): continue
        out.append({"title":t,"company":_infer_company(link),"link":link,
                    "posted_date":datetime.utcnow().date().isoformat(),
                    "source":base,"host":host})
        seen.add((t.lower(),link))
    return out

def scrape_csusb_listings(url:str=CSUSB_CSE_URL,deep:bool=True,max_pages:int=30)->pd.DataFrame:
    r=requests.get(url,headers={"User-Agent":UA},timeout=TIMEOUT); r.raise_for_status()
    rows=_collect(r.text,url); df=pd.DataFrame(rows)
    if not deep: return df
    extra=[]
    links=df["link"].tolist()[:max_pages]
    if USE_PLAYWRIGHT:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b=p.chromium.launch(headless=True,args=["--no-sandbox"]); ctx=b.new_context(user_agent=UA); pg=ctx.new_page()
            for i,l in enumerate(links,1):
                try:
                    pg.goto(l,timeout=15000); time.sleep(1)
                    extra+=_collect(pg.content(),l)
                except Exception as e: print(i,"err",e)
            b.close()
    else:
        for i,l in enumerate(links,1):
            try:
                rr=requests.get(l,headers={"User-Agent":UA},timeout=TIMEOUT)
                extra+=_collect(rr.text,l)
            except Exception as e: print(i,"err",e)
    if extra: df=pd.concat([df,pd.DataFrame(extra)],ignore_index=True)
    return df.drop_duplicates("link")

def quick_company_links_playwright(company:str,url:str=CSUSB_CSE_URL,deep:bool=True)->pd.DataFrame:
    token=company.lower().strip()
    r=requests.get(url,headers={"User-Agent":UA},timeout=TIMEOUT); soup=BeautifulSoup(r.text,"lxml")
    matches=[]
    for a in soup.find_all("a",href=True):
        t=_clean(a.get_text(" ",strip=True)); link=urljoin(url,a["href"])
        if token in t.lower() or token in link.lower():
            matches.append({"title":t,"company":_infer_company(link),"link":link,
                            "posted_date":datetime.utcnow().date().isoformat(),
                            "source":url,"host":urlparse(link).netloc})
    df=pd.DataFrame(matches)
    if not deep or df.empty: return df
    extra=[]
    for l in df["link"].tolist()[:5]:
        try:
            rr=requests.get(l,headers={"User-Agent":UA},timeout=TIMEOUT)
            extra+=_collect(rr.text,l)
        except Exception as e: print(e)
    if extra: df=pd.concat([df,pd.DataFrame(extra)],ignore_index=True)
    return df.drop_duplicates("link")
