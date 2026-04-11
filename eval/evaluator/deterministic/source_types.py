from urllib.parse import urlparse

# type weights 
SOURCE_TYPE_WEIGHTS = {
    "peer_reviewed": 1.0,
    "financial_filing": 0.95,
    "benchmark": 0.8,
    "news_article": 0.7,
    "technical_blog": 0.6,
    "case_study": 0.4,
    "marketing": 0.2,
    "unknown": 0.5
}

# verifiability
VERIFIABLE_TYPES = {
    "peer_reviewed",
    "benchmark",
    "independent_reporting",
    "open_dataset",
    "news_article",
    "financial_filing"
}

# domain rules 

PEER_REVIEWED_DOMAINS = {
    "arxiv.org", "nature.com", "science.org", "ieee.org",
    "acm.org", "springer.com", "sciencedirect.com"
}

BENCHMARK_DOMAINS = {
    "paperswithcode.com", "huggingface.co", "kaggle.com"
}

BLOG_DOMAINS = {
    "medium.com", "towardsdatascience.com", "substack.com"
}

MARKETING_DOMAINS = {
    "openai.com", "google.com", "microsoft.com", "anthropic.com"
}

# NEW: News + Financial
NEWS_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "nytimes.com",
    "cnbc.com", "ft.com", "forbes.com", "yahoo.com"
}

FINANCIAL_FILING_DOMAINS = {
    "sec.gov"
}

# keyword rules 

BENCHMARK_KEYWORDS = {
    "benchmark", "evaluation", "leaderboard", "dataset"
}

CASE_STUDY_KEYWORDS = {
    "case study", "customer story", "use case"
}

MARKETING_KEYWORDS = {
    "introducing", "announcement", "launch", "product"
}


FINANCIAL_KEYWORDS = {
    "10-k", "10q", "10-q", "8-k", "sec filing",
    "earnings report", "annual report", "quarterly report",
    "edgar", "form 10"
}

NEWS_KEYWORDS = {
    "reported", "according to", "news", "press", "coverage"
}

# domain extraction 
def extract_domain(url: str):
    if not url:
        return ""

    netloc = urlparse(url).netloc.lower()

    if netloc.startswith("www."):
        netloc = netloc[4:]

    return netloc


# classification 
def classify_source(url: str, text: str = ""):
    domain = extract_domain(url)
    url_lower = (url or "").lower()
    text_lower = (text or "").lower()

    # high confidence domain rules 
    if domain in FINANCIAL_FILING_DOMAINS:
        return "financial_filing"

    if domain in NEWS_DOMAINS:
        return "news_article"

    if domain in PEER_REVIEWED_DOMAINS:
        return "peer_reviewed"

    if domain in BENCHMARK_DOMAINS:
        return "benchmark"

    if domain in BLOG_DOMAINS:
        return "technical_blog"

    # url structure rules 
    if any(x in url_lower for x in ["10-k", "10q", "10-q", "sec.gov", "edgar"]):
        return "financial_filing"

    if any(x in url_lower for x in ["news", "article"]):
        return "news_article"

    if any(x in url_lower for x in ["arxiv.org/abs", "pdf", "paper"]):
        return "peer_reviewed"

    if any(x in url_lower for x in ["benchmark", "leaderboard", "dataset"]):
        return "benchmark"

    if any(x in url_lower for x in ["blog", "post"]):
        return "technical_blog"

    if any(x in url_lower for x in ["case-study", "customer"]):
        return "case_study"

    # text signal rules 
    if any(k in text_lower for k in FINANCIAL_KEYWORDS):
        return "financial_filing"

    if any(k in text_lower for k in NEWS_KEYWORDS):
        return "news_article"

    if any(k in text_lower for k in BENCHMARK_KEYWORDS):
        return "benchmark"

    if any(k in text_lower for k in CASE_STUDY_KEYWORDS):
        return "case_study"

    if any(k in text_lower for k in MARKETING_KEYWORDS):
        return "marketing"

    # fallback domain logic 
    if domain in MARKETING_DOMAINS:
        return "marketing"

    # final fallback 
    return "unknown"


# helpers 
def get_type_weight(source_type: str):
    return SOURCE_TYPE_WEIGHTS.get(source_type, 0.5)


def is_verifiable(source_type: str):
    return int(source_type in VERIFIABLE_TYPES)