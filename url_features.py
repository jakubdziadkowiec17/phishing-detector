import pandas as pd


def extract_basic_url_features(urls: pd.Series) -> pd.DataFrame:
    s = urls.astype("string").fillna("")
    out = pd.DataFrame(index=urls.index)

    out["url_len"] = s.str.len().astype("int64")
    out["dot_count"] = s.str.count(r"\.").astype("int64")
    out["has_https"] = s.str.startswith("https://").astype("int64")

    # Very simple IPv4 detection anywhere in URL
    ipv4_re = r"(?:\d{1,3}\.){3}\d{1,3}"
    out["has_ip"] = s.str.contains(ipv4_re, regex=True).astype("int64")

    out["digit_count"] = s.str.count(r"\d").astype("int64")
    out["special_count"] = s.str.count(r"[^A-Za-z0-9]").astype("int64")

    return out

