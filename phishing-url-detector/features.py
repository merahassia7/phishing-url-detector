import math
import re
from collections import Counter

import tldextract


KNOWN_BRANDS = [
    "paypal", "apple", "microsoft", "amazon", "facebook",
    "instagram", "netflix", "ebay", "bank", "whatsapp", "google",
]

SHORTENERS = {
    "bit.ly", "goo.gl", "shorte.st", "x.co", "ow.ly",
    "t.co", "tinyurl.com", "rb.gy", "cutt.ly",
}

SUSPICIOUS_TLDS = {
    "ru", "cn", "tk", "ml", "ga", "cf", "gq", "pw",
    "top", "xyz", "club", "online", "site", "click", "link", "live",
}

TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in freq.values())


def extract_features(url: str) -> dict:
    features = {}
    url_lower = str(url).strip().lower()

    try:
        ext = TLD_EXTRACTOR(url_lower)
        domain = ext.domain or ""
        suffix = ext.suffix or ""
        subdomain = ext.subdomain or ""
        reg_domain = f"{domain}.{suffix}" if suffix else domain
    except Exception:
        domain = suffix = subdomain = reg_domain = ""

    features["url_length"] = len(url_lower)
    features["dot_count"] = url_lower.count(".")
    features["hyphen_count"] = url_lower.count("-")
    features["slash_count"] = url_lower.count("/")
    features["at_count"] = url_lower.count("@")
    features["question_count"] = url_lower.count("?")
    features["equal_count"] = url_lower.count("=")
    features["digit_ratio"] = sum(c.isdigit() for c in domain) / len(domain) if domain else 0
    features["letter_ratio"] = sum(c.isalpha() for c in domain) / len(domain) if domain else 0
    features["is_https"] = 1 if url_lower.startswith("https") else 0
    features["is_ip"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) else 0
    features["domain_entropy"] = shannon_entropy(domain)
    features["is_shortened"] = 1 if reg_domain in SHORTENERS else 0

    sub_parts = [s for s in subdomain.split(".") if s]
    features["subdomain_count"] = len(sub_parts)
    features["too_many_subdomains"] = 1 if len(sub_parts) >= 3 else 0
    features["has_double_slash"] = 1 if url_lower.rfind("//") > 7 else 0

    for word in ["login", "verify", "update", "secure", "account", "admin", "signin", "ebayisapi", "bank"]:
        features[f"has_{word}"] = 1 if word in domain else 0

    features["domain_length"] = len(domain)
    features["tld_length"] = len(suffix)
    features["has_port"] = 1 if re.search(r":\d{2,5}/", url_lower) else 0

    features["brand_in_subdomain"] = 1 if (
        subdomain
        and any(b in subdomain for b in KNOWN_BRANDS)
        and not any(b == domain for b in KNOWN_BRANDS)
    ) else 0

    features["brand_in_domain_not_tld"] = 1 if (
        any(b in domain for b in KNOWN_BRANDS)
        and not any(b == domain for b in KNOWN_BRANDS)
    ) else 0

    features["path_length"] = len(url_lower.split("/", 3)[-1]) if "/" in url_lower else 0
    features["param_count"] = url_lower.count("&")
    features["has_encoded_chars"] = 1 if re.search(r"%[0-9a-f]{2}", url_lower) else 0
    features["digit_count_in_domain"] = sum(c.isdigit() for c in domain)
    features["suspicious_tld"] = 1 if suffix in SUSPICIOUS_TLDS else 0
    features["special_in_domain"] = sum(1 for c in domain if not c.isalnum() and c != "-")

    return features
