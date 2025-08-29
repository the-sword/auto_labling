import os
import re
import time
import random
import threading
from typing import List, Tuple, Dict

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
TIMEOUT = 12


# 全局停止事件，供外部控制停止爬取
STOP_EVENT = threading.Event()

def request_stop():
    """请求停止当前爬取任务"""
    STOP_EVENT.set()

def clear_stop():
    """清除停止标记，在新任务开始前调用"""
    STOP_EVENT.clear()

def sanitize_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return re.sub(r"[^\w\-_.]", "", name)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _download(url: str, out_path: str) -> bool:
    try:
        with requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


def _bing_search(query: str, count: int) -> List[str]:
    # Bing images: https://www.bing.com/images/search?q=...
    q = requests.utils.quote(query)
    url = f"https://www.bing.com/images/search?q={q}&form=HDRSC2&first=1&tsc=ImageBasicHover"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    imgs = []
    # Bing often stores image urls in m attribute of <a class="iusc"> or murl in JSON-like string
    for a in soup.select("a.iusc"):
        m = a.get("m") or ""
        # m contains something like: {murl:"...",turl:"..."}
        murl = None
        murl_match = re.search(r'"murl"\s*:\s*"(.*?)"', m)
        if murl_match:
            murl = murl_match.group(1)
        if murl:
            imgs.append(murl)
        if len(imgs) >= count:
            break

    # fallback: direct img tags
    if len(imgs) < count:
        for img in soup.select("img.mimg"):
            src = img.get("data-src") or img.get("src")
            if src and src.startswith("http"):
                imgs.append(src)
            if len(imgs) >= count:
                break
    return imgs[:count]


def _google_search(query: str, count: int) -> List[str]:
    """Lightweight Google Images scrape (public results).
    Note: Google may change markup frequently and apply rate-limits. This is best-effort only.
    """
    q = requests.utils.quote(query)
    # Use images vertical (tbm=isch). hl to stabilize markup; safe=off for raw results
    url = f"https://www.google.com/search?tbm=isch&q={q}&hl=en&safe=off"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    imgs: List[str] = []

    # Primary: grid images under #islrg img (avoid base64 data thumbnails)
    for img in soup.select("#islrg img"):
        src = img.get("data-src") or img.get("src")
        if not src:
            continue
        if src.startswith("http") and not src.startswith("https://encrypted-tbn0.gstatic.com"):
            imgs.append(src)
        # Accept gstatic (thumbnail) only if nothing else is available later
        if len(imgs) >= count:
            break

    # Fallback: any <img> in page with http URL (skip data URIs)
    if len(imgs) < count:
        for img in soup.select("img"):
            src = img.get("data-src") or img.get("src")
            if not src:
                continue
            if src.startswith("http") and not src.startswith("data:"):
                imgs.append(src)
            if len(imgs) >= count:
                break

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in imgs:
        if u not in seen:
            seen.add(u)
            unique.append(u)
        if len(unique) >= count:
            break
    return unique[:count]

essential_headers = {
    "Referer": "https://image.baidu.com/",
    "User-Agent": USER_AGENT,
}


def _baidu_search(query: str, count: int) -> List[str]:
    # Simple Baidu image search page scrape (public results), note it may change frequently.
    q = requests.utils.quote(query)
    url = f"https://image.baidu.com/search/index?tn=baiduimage&word={q}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    imgs = []
    # Common layout: img src or data-imgurl
    for img in soup.select("img"):
        src = img.get("data-imgurl") or img.get("data-src") or img.get("src")
        if src and src.startswith("http"):
            imgs.append(src)
        if len(imgs) >= count:
            break
    return imgs[:count]


SUPPORTED_ENGINES = {
    "bing": _bing_search,
    "baidu": _baidu_search,
    "google": _google_search,
}


def crawl_images(engine: str, query: str, limit: int, out_dir: str) -> Dict:
    engine = (engine or "").lower()
    if engine not in SUPPORTED_ENGINES:
        return {"success": False, "error": f"Unsupported engine: {engine}"}

    ensure_dir(out_dir)
    fetcher = SUPPORTED_ENGINES[engine]

    urls = fetcher(query, limit)
    if not urls:
        return {"success": False, "error": "No images found or search failed"}

    saved = []
    base = sanitize_filename(query) or "images"
    for idx, u in enumerate(urls, start=1):
        # 支持外部停止
        if STOP_EVENT.is_set():
            break
        ext = os.path.splitext(u.split("?")[0])[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
            ext = ".jpg"
        fname = f"{base}_{idx:04d}{ext}"
        path = os.path.join(out_dir, fname)
        ok = _download(u, path)
        if ok:
            saved.append({"url": u, "path": path})
        # small polite delay
        # 停止检查 + 延时
        for _ in range(3):
            if STOP_EVENT.is_set():
                break
            time.sleep(random.uniform(0.05, 0.1))

    return {
        "success": True,
        "engine": engine,
        "query": query,
        "requested": limit,
        "found": len(urls),
        "saved_count": len(saved),
        "saved": saved,
        "stopped": STOP_EVENT.is_set(),
    }
