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

load_dotenv()

app = FastAPI(title="The MBA Brief API")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

HTML_FILE = Path(__file__).parent / "the-mba-brief.html"

BRIEF_PROMPT = """You are the editorial AI behind "The MBA Brief" — a premium daily business newsletter for MBA students at India's top B-Schools (IIM-A, IIM-B, ISB, XLRI, etc.).

Today's date is {today} (India Standard Time). Generate today's complete daily brief as a single valid JSON object.

The JSON must exactly follow this schema:

{{
  "edition": {{
    "number": <integer, increment logically based on date, start from 200 for April 2026>,
    "date": "{today}",
    "readTime": <integer, total estimated minutes to read all content>,
    "sector": "<sector name for today's spotlight, e.g. Fintech, EV, Pharma, FMCG, etc.>"
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
      "source": "<e.g. MINT, ET, BS, BLOOMBERG, REUTERS>",
      "headline": "<compelling, specific headline>",
      "gist": "<2-3 sentences of factual summary, include specific numbers>",
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
      "source": "<e.g. PIB, MINT, ET>",
      "headline": "<specific policy headline>",
      "gist": "<2 sentences>",
      "sowhat": "<MBA relevance>",
      "context": ["<bullet 1>", "<bullet 2>"],
      "readTime": "90 sec"
    }},
    {{"id": "p2", "source": "...", "headline": "...", "gist": "...", "sowhat": "...", "context": ["...", "..."], "readTime": "90 sec"}}
  ],
  "global": [
    {{
      "id": "g1",
      "source": "<e.g. FT, BLOOMBERG, WSJ>",
      "headline": "<global headline with India angle>",
      "gist": "<2 sentences>",
      "sowhat": "<why India MBA students should care>",
      "context": ["<bullet 1>", "<bullet 2>"],
      "readTime": "90 sec"
    }},
    {{"id": "g2", "source": "...", "headline": "...", "gist": "...", "sowhat": "...", "context": ["...", "..."], "readTime": "90 sec"}}
  ],
  "sector": {{
    "name": "<sector name>",
    "headline": "<bold, analytical headline about the sector>",
    "body": "<3-4 sentences of deep analysis with specific data points, company names, and metrics>",
    "players": ["<Company — description>", "<Company — description>", "<Company — description>", "<Company — description>"],
    "implications": ["<Implication for MBAs in consulting/finance>", "<Implication for MBAs in ops/supply chain>", "<Implication for MBAs in strategy>"],
    "metrics": ["<Key metric to track 1>", "<Key metric to track 2>", "<Key metric to track 3>"]
  }},
  "wordOfDay": {{
    "term": "<MBA/finance/economics term>",
    "definition": "<precise, educational definition>",
    "example": "<today's context: how this term connects to today's news>"
  }},
  "vault": [
    {{"title": "<Background topic 1>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}},
    {{"title": "<Background topic 2>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}},
    {{"title": "<Background topic 3>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}},
    {{"title": "<Background topic 4>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}},
    {{"title": "<Background topic 5>", "points": ["<point 1>", "<point 2>", "<point 3>", "<point 4>", "<point 5>"]}}
  ]
}}

CRITICAL RULES:
- Return ONLY the JSON object. No markdown, no code fences, no commentary.
- All content must be relevant to India's business/economy context — but include global stories with India angle.
- Use real, plausible data points and company names appropriate for {today}.
- Stories should reflect events plausible for {today} — think about recent trends in India's economy: RBI policy, IPOs, PLI schemes, FDI, digital economy, global trade tensions, energy transition.
- The 5 Daily stories should cover a diverse mix: finance/markets, corporate, policy, global, and a unique sector story.
- wordOfDay should directly connect to one of today's stories.
- vault entries should provide background on storylines mentioned in today's brief.
- Make content premium, analytical, and opinionated — not generic.
"""


def _yf_ticker(ticker: str) -> tuple[float, float] | None:
    """Return (last_price, prev_close) for a Yahoo Finance ticker, or None."""
    try:
        fi = yf.Ticker(ticker).fast_info
        p, c = float(fi.last_price), float(fi.previous_close)
        if p and c:
            return p, c
    except Exception:
        pass
    return None


def _fetch_usd_inr() -> tuple[float, float] | None:
    """Fetch USD/INR from fawazahmed0 Currency API (free, no key, real-time + history)."""
    try:
        import urllib.request, json as _json, datetime as _dt
        # Today's rate
        url_today = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
        with urllib.request.urlopen(url_today, timeout=6) as r:
            today_rate = _json.loads(r.read())["usd"]["inr"]
        # Yesterday's rate (skip weekends by going back 3 days max)
        for days_back in (1, 2, 3):
            try:
                date_str = (_dt.date.today() - _dt.timedelta(days=days_back)).isoformat()
                url_prev = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date_str}/v1/currencies/usd.json"
                with urllib.request.urlopen(url_prev, timeout=6) as r2:
                    prev_rate = _json.loads(r2.read())["usd"]["inr"]
                break
            except Exception:
                prev_rate = today_rate  # fallback: show no change
        return float(today_rate), float(prev_rate)
    except Exception:
        return None


def _fetch_brent() -> tuple[float, float] | None:
    """Fetch Brent Crude via yfinance BZ=F — returns (last, prev) in USD."""
    result = _yf_ticker("BZ=F")
    if result:
        return result
    # Fallback: try CL=F (WTI, close enough as indicator)
    return _yf_ticker("CL=F")


def fetch_live_markets() -> list[dict]:
    """Fetch real-time SENSEX, NIFTY, USD/INR, Brent Crude from multiple sources."""
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




# In-memory cache as fallback when filesystem is ephemeral (e.g. cloud)
_memory_cache: dict[str, dict] = {}

# Models to try in order (primary → fallback)
MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]


def get_cache_path(date_str: str) -> Path:
    return CACHE_DIR / f"brief_{date_str}.json"


def inject_live_markets(data: dict) -> dict:
    """Replace the markets array with real-time Yahoo Finance prices."""
    live = fetch_live_markets()
    if live:  # only override if we got valid data
        data["markets"] = live
    return data


def generate_brief(today: str) -> dict:
    """Call Gemini to generate today's brief, trying models in order."""
    prompt = BRIEF_PROMPT.format(today=today)
    last_err = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.85,
                    response_mime_type="application/json",
                )
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
    # 1. Check memory cache — still refresh live markets on every request
    if today in _memory_cache:
        return inject_live_markets(_memory_cache[today])
    # 2. Check disk cache
    cache_path = get_cache_path(today)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _memory_cache[today] = data
        return data
    # 3. Generate fresh
    data = generate_brief(today)
    _memory_cache[today] = data
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # Cloud ephemeral filesystem — skip disk write
    return data


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=HTML_FILE.read_text(encoding="utf-8"))


@app.get("/api/brief")
async def api_brief():
    today = datetime.date.today().isoformat()  # YYYY-MM-DD
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
