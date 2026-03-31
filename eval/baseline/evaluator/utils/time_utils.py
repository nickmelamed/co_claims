from datetime import datetime
import time

# Common formats your RAG might return
DATE_FORMATS = [
    "%Y-%m-%d",        # 2024-01-15
    "%Y/%m/%d",        # 2024/01/15
    "%Y-%m",           # 2024-01
    "%Y",              # 2024
    "%b %Y",           # Jan 2024
    "%B %Y",           # January 2024
    "%b %d %Y",        # Jan 15 2024
    "%B %d %Y",        # January 15 2024
]


def normalize_timestamp(ts):
    """
    Converts timestamps into days since epoch.
    Handles:
    - ISO strings
    - partial dates (year/month)
    - numeric timestamps
    - bad/missing values
    """
    if ts is None:
        return None

    # already numeric (assume days)
    if isinstance(ts, (int, float)):
        return ts

    ts = str(ts).strip()

    # try ISO first (fast path)
    try:
        dt = datetime.fromisoformat(ts)
        return dt.timestamp() / (60 * 60 * 24)
    except:
        pass

    # try known formats
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.timestamp() / (60 * 60 * 24)
        except:
            continue

    return None  # failed parsing


def current_time_days():
    return time.time() / (60 * 60 * 24)