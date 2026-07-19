import requests

_WEB_SEARCH_TIMEOUT = 15
_BRAVE_API_KEY = None


def configure(brave_api_key: str | None = None):
    global _BRAVE_API_KEY
    _BRAVE_API_KEY = (brave_api_key or "").strip() or None


_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web via Brave Search for current events, docs, or facts. "
            "Summarize findings; post to chat only if relevant to others."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (1-10, default 5)",
                },
            },
            "required": ["query"],
        },
    },
}


def execute_web_search(query: str, max_results: int = 5) -> str:
    if not _BRAVE_API_KEY:
        return "Error: web search not configured (set brave_api_key in mind config)"
    q = (query or "").strip()
    if not q:
        return "Error: empty query"
    try:
        n = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        n = 5
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": _BRAVE_API_KEY,
            },
            params={"q": q, "count": n},
            timeout=_WEB_SEARCH_TIMEOUT,
        )
        if r.status_code != 200:
            detail = (r.text or "")[:500]
            return f"Error: Brave Search HTTP {r.status_code}" + (f" — {detail}" if detail else "")
        data = r.json()
        results = (data.get("web") or {}).get("results") or []
        if not results:
            return f"No results for: {q}"
        parts = [f"Query: {q}", f"Results ({len(results)}):"]
        for i, item in enumerate(results, 1):
            title = (item.get("title") or "(no title)").strip()
            url = (item.get("url") or "").strip()
            desc = (item.get("description") or "").strip()
            block = f"{i}. {title}"
            if desc:
                block += f"\n   {desc}"
            if url:
                block += f"\n   {url}"
            parts.append(block)
        out = "\n\n".join(parts)
        if len(out) > 7000:
            out = out[:7000] + "\n[truncated]"
        return out
    except requests.Timeout:
        return f"Error: timed out after {_WEB_SEARCH_TIMEOUT}s"
    except Exception as e:
        return f"Error: {str(e)}"
