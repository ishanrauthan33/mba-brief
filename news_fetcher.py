"""
news_fetcher.py
Fetches real news articles from RSS feeds across Indian & global business sources.
No API keys required — all feeds are publicly available.
"""

import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from typing import Optional
import re

try:
    import feedparser
    _FP_AVAILABLE = True
except ImportError:
    _FP_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS Feed registry — source name, URL, category
# ---------------------------------------------------------------------------
RSS_FEEDS = [
    # --- Indian Business & Markets ---
    {
        "source": "Economic Times",
        "url": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
        "category": "india",
    },
    {
        "source": "Economic Times Markets",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "category": "markets",
    },
    {
        "source": "Mint",
        "url": "https://www.livemint.com/rss/news",
        "category": "india",
    },
    {
        "source": "Business Standard",
        "url": "https://www.business-standard.com/rss/home_page_top_stories.rss",
        "category": "india",
    },
    # --- Policy / Government ---
    {
        "source": "PIB",
        "url": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
        "category": "policy",
    },
    {
        "source": "Economic Times Economy",
        "url": "https://economictimes.indiatimes.com/economy/rssfeeds/1373380680.cms",
        "category": "policy",
    },
    # --- Global ---
    {
        "source": "Reuters Business",
        "url": "https://feeds.reuters.com/reuters/businessNews",
        "category": "global",
    },
    {
        "source": "BBC Business",
        "url": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "category": "global",
    },
    {
        "source": "Financial Times",
        "url": "https://www.ft.com/rss/home/uk",
        "category": "global",
    },
]

# Max articles to collect per feed
MAX_PER_FEED = 5
# How many hours old an article can be before we skip it (0 = no filter)
MAX_AGE_HOURS = 36


def _clean_html(text: str) -> str:
    """Strip HTML tags from feed summaries."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"&[a-z]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:500]  # cap at 500 chars


def _parse_date(entry) -> Optional[datetime.datetime]:
    """Try to extract a timezone-aware datetime from a feedparser entry."""
    # feedparser sets published_parsed as a time.struct_time (UTC)
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            import calendar, time as _time
            ts = calendar.timegm(entry.published_parsed)
            return datetime.datetime.utcfromtimestamp(ts).replace(
                tzinfo=datetime.timezone.utc
            )
        except Exception:
            pass
    if hasattr(entry, "published") and entry.published:
        try:
            return parsedate_to_datetime(entry.published)
        except Exception:
            pass
    return None


def _fetch_feed(feed_info: dict) -> list[dict]:
    """Fetch a single RSS feed and return a list of article dicts."""
    if not _FP_AVAILABLE:
        return []

    articles = []
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=MAX_AGE_HOURS)
        if MAX_AGE_HOURS > 0
        else None
    )

    try:
        parsed = feedparser.parse(feed_info["url"])
        for entry in parsed.entries[:MAX_PER_FEED]:
            title = (entry.get("title") or "").strip()
            summary = _clean_html(entry.get("summary") or entry.get("description") or "")
            link = entry.get("link") or ""

            if not title:
                continue

            pub_dt = _parse_date(entry)
            if cutoff and pub_dt and pub_dt < cutoff:
                continue  # article too old

            articles.append(
                {
                    "source": feed_info["source"],
                    "category": feed_info["category"],
                    "headline": title,
                    "summary": summary,
                    "url": link,
                    "published": pub_dt.isoformat() if pub_dt else "",
                }
            )
    except Exception as e:
        logger.warning(f"Failed to fetch {feed_info['source']}: {e}")

    return articles


def fetch_all_news(max_workers: int = 6) -> dict[str, list[dict]]:
    """
    Fetch news from all configured RSS feeds in parallel.

    Returns a dict keyed by category:
      {
        "india":   [...articles],
        "markets": [...articles],
        "policy":  [...articles],
        "global":  [...articles],
      }
    """
    all_articles: dict[str, list[dict]] = {
        "india": [],
        "markets": [],
        "policy": [],
        "global": [],
    }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_feed, feed): feed for feed in RSS_FEEDS}
        for future in as_completed(futures):
            try:
                articles = future.result()
                for art in articles:
                    all_articles[art["category"]].append(art)
            except Exception as e:
                logger.warning(f"Feed future error: {e}")

    # Deduplicate within each category by headline (case-insensitive prefix match)
    for cat in all_articles:
        seen: set[str] = set()
        deduped = []
        for art in all_articles[cat]:
            key = art["headline"][:60].lower()
            if key not in seen:
                seen.add(key)
                deduped.append(art)
        all_articles[cat] = deduped

    total = sum(len(v) for v in all_articles.values())
    logger.info(
        f"Fetched {total} real articles: "
        f"india={len(all_articles['india'])}, "
        f"markets={len(all_articles['markets'])}, "
        f"policy={len(all_articles['policy'])}, "
        f"global={len(all_articles['global'])}"
    )

    return all_articles


def format_articles_for_prompt(articles_by_cat: dict[str, list[dict]]) -> str:
    """
    Render fetched articles as numbered bullet lines for Gemini's prompt.
    Returns a plain-text string the prompt can embed verbatim.
    """
    lines: list[str] = []
    category_labels = {
        "india": "INDIA BUSINESS & MARKETS",
        "markets": "FINANCIAL MARKETS",
        "policy": "POLICY & ECONOMY",
        "global": "GLOBAL BUSINESS",
    }

    for cat, label in category_labels.items():
        arts = articles_by_cat.get(cat, [])
        if not arts:
            continue
        lines.append(f"\n## {label}")
        for i, art in enumerate(arts, 1):
            pub = f" [{art['published'][:10]}]" if art.get("published") else ""
            lines.append(f"{i}. [{art['source']}{pub}] {art['headline']}")
            if art.get("summary"):
                lines.append(f"   Summary: {art['summary']}")

    return "\n".join(lines)
