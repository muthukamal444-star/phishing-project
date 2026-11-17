import re
from urllib.parse import urlparse

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"""https?://[^\s<>"']+|www\.[^\s<>"']+""", re.I)
URGENT_WORDS = {
    "urgent", "immediately", "verify", "suspend", "suspended", "password",
    "account", "login", "confirm", "payment", "invoice", "click", "update",
    "unusual", "security", "limited", "expire", "expired", "winner", "prize"
}

def extract_url_features(text):
    text = str(text or "")
    urls = URL_RE.findall(text)
    domains = []
    for url in urls:
        candidate = url if url.lower().startswith(("http://", "https://")) else "http://" + url
        try:
            host = urlparse(candidate).netloc.lower().split("@")[-1].split(":")[0]
        except Exception:
            host = ""
        domains.append(host)

    return [
        len(urls),
        sum(len(u) for u in urls) / max(len(urls), 1),
        int(any(re.search(r"\d+\.\d+\.\d+\.\d+", u) for u in urls)),
        int(any("@" in u for u in urls)),
        int(any(u.lower().startswith("http://") or u.lower().startswith("www.") for u in urls)),
        int(any(d.count(".") >= 3 for d in domains)),
        int(any("-" in d for d in domains)),
        int(any(d.endswith((".zip", ".click", ".top", ".xyz", ".work", ".gq")) for d in domains)),
    ]

def extract_text_features(text):
    text = str(text or "")
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    return [
        len(text),
        len(words),
        text.count("!"),
        text.count("$"),
        int(any(word in lower for word in URGENT_WORDS)),
        sum(lower.count(word) for word in URGENT_WORDS),
        int(bool(re.search(r"\b(dear customer|dear user|valued customer)\b", lower))),
        int(bool(re.search(r"\b(verify|confirm|reset).{0,30}\b(password|account|login)\b", lower))),
    ]

class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Combines TF-IDF text vectors with simple URL and language indicators."""
    def __init__(self, max_features=12000, ngram_range=(1, 2), min_df=1):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english",
            max_features=max_features, ngram_range=ngram_range, min_df=min_df,
            sublinear_tf=True
        )

    def fit(self, X, y=None):
        texts = [str(x or "") for x in X]
        self.vectorizer.fit(texts)
        return self

    def transform(self, X):
        texts = [str(x or "") for x in X]
        text_matrix = self.vectorizer.transform(texts)
        extra = np.asarray([extract_url_features(t) + extract_text_features(t) for t in texts], dtype=float)
        return hstack([text_matrix, csr_matrix(extra)], format="csr")
