import os
import json
import re
import datetime
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

from news_fetcher import fetch_all_news, format_articles_for_prompt

load_dotenv()

app = FastAPI(title="The MBA Brief API")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

HTML_FILE = Path(__file__).parent / "the-mba-brief.html"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Used when we have real news to ground the brief
BRIEF_PROMPT_WITH_NEWS = """You are the editorial AI behind "The MBA Brief" — a premium daily business newsletter for MBA students at India's top B-Schools (IIM-A, IIM-B, ISB, XLRI, etc.).

Today's date is {today} (India Standard Time).

Below are REAL news articles fetched TODAY from top business news outlets. You MUST base your brief on these real articles. Do NOT fabricate or invent news. If an article does not have enough detail in its summary, you may expand on the known facts about that real topic — but the core event must be real and from the list below.

=== REAL NEWS ARTICLES ===
{articles_text}
=== END OF NEWS ARTICLES ===

Using the above real articles as your source material, generate today's complete daily brief as a single valid JSON object.

The JSON must exactly follow this schema:

{{
  "edition": {{
    "number": <integer, increment logically based on date, start from 200 for April 2026>,
    "date": "{today}",
    "readTime": <integer, total estimated minutes to read all content>,
    "sector": "<sector name for today's spotlight, pick one relevant to today's top news, e.g. Fintech, EV, Pharma, FMCG, etc.>"
  }},
  "markets": [
    {{"label": "SENSEX", "value": "<realistic value>", "change": "<+/- X (Y%)>", "dir": "up"|"down"}},
    {{"label": "NIFTY 50", "value": "<realistic value>", "change": "<+/- X (Y%)>", "dir": "up"|"down"}},
    {{"label": "USD / INR", "value": "₹XX.XX", "change": "<+/- X (Y%)>", "dir": "up"|"down"}},
    {{"label": "BRENT CRUDE", "value": "$XX.XX", "change": "<+/- X (Y%)>", "dir": "up"|"down"}}
  ],
  "stories": [
    {{
      "id": "s1",
      "lead": true,
      "source": "<exact source name from the article list, e.g. Economic Times, Reuters, Mint>",
      "headline": "<the real headline from the article, or a close paraphrase>",
      "gist": "<2-3 sentences factual summary based on the real article — include specific numbers if mentioned>",
      "sowhat": "<1-2 sentences: the MBA student takeaway — career, case study, or analytical relevance>",
      "context": ["<bullet 1>", "<bullet 2>", "<bullet 3>", "<bullet 4>", "<bullet 5>"],
      "readTime": "2 min"
    }},
    {{"id": "s2", "lead": false, "source": "...", "headline": "...", "gist": "...", "sowhat": "...", "context": ["...", "...", "...", "...", "..."], "readTime": "90 sec"}},
    {{"id": "s3", "lead": false, "source": "...", "headline": "...", "gist": "...", "sowhat": "...", "context": ["...", "...", "...", "...", "..."], "readTime": "90 sec"}},
    {{"id": "s4", "lead": false, "source": "...", "headline": "...", "gist": "...", "sowhat": "...", "context": ["...", "...", "...", "...", "..."], "readTime": "90 sec"}},
    {{"id": "s5", "lead": false, "source": "...", "headline": "...", "gist": "...", "sowhat": "...", "context": ["...", "...", "...", "...", "..."], "readTime": "90 sec"}}
  ],
  "policy": [
    {{
      "id": "p1",
      "source": "<exact source from the article list>",
      "headline": "<real policy/economy headline>",
      "gist": "<2 sentences based on real article>",
      "sowhat": "<MBA relevance>",
      "context": ["<bullet 1>", "<bullet 2>"],
      "readTime": "90 sec"
    }},
    {{"id": "p2", "source": "...", "headline": "...", "gist": "...", "sowhat": "...", "context": ["...", "..."], "readTime": "90 sec"}}
  ],
  "global": [
    {{
      "id": "g1",
      "source": "<exact source from the article list>",
      "headline": "<real global headline with India angle>",
      "gist": "<2 sentences based on real article>",
      "sowhat": "<why India MBA students should care>",
      "context": ["<bullet 1>", "<bullet 2>"],
      "readTime": "90 sec"
    }},
    {{"id": "g2", "source": "...", "headline": "...", "gist": "...", "sowhat": "...", "context": ["...", "..."], "readTime": "90 sec"}}
  ],
  "sector": {{
    "name": "<sector name — pick based on which real story dominates today's news>",
    "headline": "<bold, analytical headline about this sector based on today's real news>",
    "body": "<3-4 sentences of deep analysis grounded in today's real articles, with specific data points, company names, and metrics>",
    "players": ["<Company — description>", "<Company — description>", "<Company — description>", "<Company — description>"],
    "implications": ["<Implication for MBAs in consulting/finance>", "<Implication for MBAs in ops/supply chain>", "<Implication for MBAs in strategy>"],
    "metrics": ["<Key metric to track 1>", "<Key metric to track 2>", "<Key metric to track 3>"]
  }},
  "wordOfDay": {{
    "term": "<MBA/finance/economics term directly relevant to today's real news>",
    "definition": "<precise, educational definition>",
    "example": "<how this term specifically connects to one of today's real stories>"
  }},
  "vault": [
    {{"title": "<Background topic relevant to today's news 1>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}},
    {{"title": "<Background topic 2>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}},
    {{"title": "<Background topic 3>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}},
    {{"title": "<Background topic 4>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}},
    {{"title": "<Background topic 5>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}}
  ]
}}

CRITICAL RULES:
- Return ONLY the JSON object. No markdown, no code fences, no commentary.
- Every story, policy item, and global item MUST correspond to a real article from the list above.
- The 5 Daily stories should cover a diverse mix: finance/markets, corporate, policy, global, and a unique sector story.
- For the "context" bullets: provide real analytical context — industry background, historical comparison, key numbers, stakeholder impact. These CAN go beyond the article itself as educational MBA context.
- wordOfDay must directly connect to one of today's real stories.
- vault entries should provide background context that helps MBAs better understand the real stories covered.
- Make analysis premium, analytical, and opinionated — not generic.
- If the article list is thin in any category (policy, global), pick the best available story from any category and adapt it.
"""

# Fallback prompt used only if ALL RSS feeds fail
BRIEF_PROMPT_FALLBACK = """You are the editorial AI behind "The MBA Brief" — a premium daily business newsletter for MBA students at India's top B-Schools (IIM-A, IIM-B, ISB, XLRI, etc.).

Today's date is {today} (India Standard Time).

NOTE: Real-time news could not be fetched today due to connectivity issues. Generate a brief based on credible, known business developments and publicly known facts as of your training data. Clearly stay close to known reality — do not fabricate dramatic or speculative events.

Generate today's complete daily brief as a single valid JSON object following the exact same schema as always (edition, markets, stories, policy, global, sector, wordOfDay, vault). Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

def _yf_ticker(ticker: str) -> tuple[float, float] | None:
    try:
        fi = yf.Ticker(ticker).fast_info
        p, c = float(fi.last_price), float(fi.previous_close)
        if p and c:
            return p, c
    except Exception:
        pass
    return None


def _fetch_usd_inr() -> tuple[float, float] | None:
    try:
        import urllib.request, json as _json, datetime as _dt
        url_today = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
        with urllib.request.urlopen(url_today, timeout=6) as r:
            today_rate = _json.loads(r.read())["usd"]["inr"]
        for days_back in (1, 2, 3):
            try:
                date_str = (_dt.date.today() - _dt.timedelta(days=days_back)).isoformat()
                url_prev = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date_str}/v1/currencies/usd.json"
                with urllib.request.urlopen(url_prev, timeout=6) as r2:
                    prev_rate = _json.loads(r2.read())["usd"]["inr"]
                break
            except Exception:
                prev_rate = today_rate
        return float(today_rate), float(prev_rate)
    except Exception:
        return None


def _fetch_brent() -> tuple[float, float] | None:
    result = _yf_ticker("BZ=F")
    if result:
        return result
    return _yf_ticker("CL=F")


def fetch_live_markets() -> list[dict]:
    if not _YF_AVAILABLE:
        return []

    def make_entry(label: str, vals: tuple[float, float] | None,
                   prefix: str = "", suffix: str = "") -> dict | None:
        if not vals:
            return None
        today_val, prev_val = vals
        change = today_val - prev_val
        pct    = (change / prev_val) * 100
        sign   = "+" if change >= 0 else ""
        direction = "up" if change >= 0 else "down"
        if today_val >= 10_000:
            value_str = f"{prefix}{today_val:,.2f}{suffix}"
        else:
            value_str = f"{prefix}{today_val:.2f}{suffix}"
        change_str = f"{sign}{change:.2f} ({sign}{pct:.2f}%)"
        return {"label": label, "value": value_str, "change": change_str, "dir": direction}

    entries = [
        make_entry("SENSEX",      _yf_ticker("^BSESN")),
        make_entry("NIFTY 50",    _yf_ticker("^NSEI")),
        make_entry("USD / INR",   _fetch_usd_inr(), prefix="₹"),
        make_entry("BRENT CRUDE", _fetch_brent(),   prefix="$"),
    ]
    return [e for e in entries if e is not None]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_memory_cache: dict[str, dict] = {}
MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]


def get_cache_path(date_str: str) -> Path:
    return CACHE_DIR / f"brief_{date_str}.json"


def inject_live_markets(data: dict) -> dict:
    live = fetch_live_markets()
    if live:
        data["markets"] = live
    return data


# ---------------------------------------------------------------------------
# Core: fetch news → build prompt → call Gemini
# ---------------------------------------------------------------------------

def generate_brief(today: str) -> dict:
    """Fetch real news, pass to Gemini for analysis, return structured brief."""

    # 1. Fetch real articles from RSS feeds
    articles_by_cat = fetch_all_news()
    total_articles = sum(len(v) for v in articles_by_cat.values())

    if total_articles >= 5:
        articles_text = format_articles_for_prompt(articles_by_cat)
        prompt = BRIEF_PROMPT_WITH_NEWS.format(today=today, articles_text=articles_text)
        print(f"[MBA Brief] Using {total_articles} real articles from RSS feeds.")
    else:
        # All feeds failed — fall back to knowledge-based (but honest) generation
        prompt = BRIEF_PROMPT_FALLBACK.format(today=today)
        print(f"[MBA Brief] WARNING: Only {total_articles} articles fetched. Using fallback prompt.")

    # 2. Call Gemini (try models in order)
    last_err = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,          # lower temp = more faithful to real articles
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            data = json.loads(raw)
            return inject_live_markets(data)
        except Exception as e:
            last_err = e
            continue

    raise last_err


def get_brief(today: str) -> dict:
    """Return cached brief (memory → disk → generate)."""
    if today in _memory_cache:
        return inject_live_markets(_memory_cache[today])
    cache_path = get_cache_path(today)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _memory_cache[today] = data
        return inject_live_markets(data)
    # Generate fresh with real news
    data = generate_brief(today)
    _memory_cache[today] = data
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=HTML_FILE.read_text(encoding="utf-8"))


@app.get("/api/brief")
async def api_brief():
    today = datetime.date.today().isoformat()
    try:
        data = get_brief(today)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": str(e), "message": "Failed to generate brief. Check API key quota."},
        )


@app.delete("/api/cache")
async def clear_cache():
    """Dev endpoint: wipe today's cache to force regeneration."""
    today = datetime.date.today().isoformat()
    _memory_cache.pop(today, None)
    cache_path = get_cache_path(today)
    if cache_path.exists():
        cache_path.unlink()
        return {"message": f"Cache cleared for {today}"}
    return {"message": "No cache found for today (disk); memory cache cleared if present."}


@app.get("/api/news-sources")
async def news_sources():
    """Debug endpoint: show which RSS feeds are configured."""
    from news_fetcher import RSS_FEEDS
    return {"feeds": [{"source": f["source"], "category": f["category"]} for f in RSS_FEEDS]}
