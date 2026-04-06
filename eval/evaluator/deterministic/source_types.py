from urllib.parse import urlparse

# Type Weights
SOURCE_TYPE_WEIGHTS = {
    "peer_reviewed": 1.0,
    "benchmark": 0.8,
    "technical_blog": 0.6,
    "case_study": 0.4,
    "marketing": 0.2,
    "unknown": 0.5
}

VERIFIABLE_TYPES = {
    "peer_reviewed",
    "benchmark",
    "independent_reporting",
    "open_dataset"
}


# Domain Rules
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


# Keyword Rules
BENCHMARK_KEYWORDS = {
    "benchmark", "evaluation", "leaderboard", "dataset"
}

CASE_STUDY_KEYWORDS = {
    "case study", "customer story", "use case"
}

MARKETING_KEYWORDS = {
    "introducing", "announcement", "launch", "product"
}


# Domain Extraction 
def extract_domain(url: str):
    netloc = urlparse(url).netloc.lower()

    if netloc.startswith("www."):
        netloc = netloc[4:]

    return netloc


# Classification 
def classify_source(url: str, text: str = ""):
    domain = extract_domain(url)
    url_lower = url.lower()
    text_lower = text.lower()

    # For domain rules
    if domain in PEER_REVIEWED_DOMAINS:
        return "peer_reviewed"

    if domain in BENCHMARK_DOMAINS:
        return "benchmark"

    if domain in BLOG_DOMAINS:
        return "technical_blog"

    if domain in MARKETING_DOMAINS:
        # fall through — could still be case study or benchmark
        pass

    # for URL structure 
    if any(x in url_lower for x in ["arxiv.org/abs", "pdf", "paper"]):
        return "peer_reviewed"

    if any(x in url_lower for x in ["benchmark", "leaderboard", "dataset"]):
        return "benchmark"

    if any(x in url_lower for x in ["blog", "post"]):
        return "technical_blog"

    if any(x in url_lower for x in ["case-study", "customer"]):
        return "case_study"

    # text signals 
    if any(k in text_lower for k in BENCHMARK_KEYWORDS):
        return "benchmark"

    if any(k in text_lower for k in CASE_STUDY_KEYWORDS):
        return "case_study"

    if any(k in text_lower for k in MARKETING_KEYWORDS):
        return "marketing"

    # fallback logic 
    if domain in MARKETING_DOMAINS:
        return "marketing"

    return "unknown"


# helpers 
def get_type_weight(source_type: str):
    return SOURCE_TYPE_WEIGHTS.get(source_type, 0.5)


def is_verifiable(source_type: str):
    return int(source_type in VERIFIABLE_TYPES)