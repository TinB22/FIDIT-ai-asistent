import json
import os
import hashlib
from typing import Any, Dict, Optional, Tuple

CACHE_DIR = os.path.join("data", "cache")
ANSWERS_CACHE_PATH = os.path.join(CACHE_DIR, "answers_cache.json")

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def stable_key(*parts: str) -> str:
    raw = "||".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def load_answers_cache() -> Dict[str, Any]:
    ensure_cache_dir()
    if not os.path.exists(ANSWERS_CACHE_PATH):
        return {}
    try:
        with open(ANSWERS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_answers_cache(cache: Dict[str, Any]) -> None:
    ensure_cache_dir()
    with open(ANSWERS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_cached_answer(key: str) -> Optional[Dict[str, Any]]:
    cache = load_answers_cache()
    return cache.get(key)

def set_cached_answer(key: str, value: Dict[str, Any]) -> None:
    cache = load_answers_cache()
    cache[key] = value
    save_answers_cache(cache)