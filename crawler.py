import os
import re
import time
import random
from typing import List, Tuple, Dict

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
TIMEOUT = 12


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
    # "google": ...  # Google often requires additional measures; omitted for now.
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
        ext = os.path.splitext(u.split("?")[0])[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
            ext = ".jpg"
        fname = f"{base}_{idx:04d}{ext}"
        path = os.path.join(out_dir, fname)
        ok = _download(u, path)
        if ok:
            saved.append({"url": u, "path": path})
        # small polite delay
        time.sleep(random.uniform(0.2, 0.6))

    return {
        "success": True,
        "engine": engine,
        "query": query,
        "requested": limit,
        "found": len(urls),
        "saved_count": len(saved),
        "saved": saved,
    }
