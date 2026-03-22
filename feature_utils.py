import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def combine_text(df):
    sample_io_text = df["sample_io"].apply(
        lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x)
    )

    return (
        df["title"].astype(str) + " " +
        df["description"].astype(str) + " " +
        df["input_description"].astype(str) + " " +
        df["output_description"].astype(str) + " " +
        sample_io_text.fillna("")
    )

class TextExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return combine_text(X)

class HandcraftedTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text = combine_text(X).str.lower()
        features = pd.DataFrame()

        features["char_len"] = text.str.len()
        features["word_count"] = text.str.split().apply(len)
        features["digit_count"] = text.str.count(r"\d")
        features["math_symbols"] = text.str.count(r"[\+\-\*/=%]")

        keywords = [
            "graph", "tree", "dp", "dynamic", "greedy",
            "dfs", "bfs", "binary", "search", "sort",
            "mod", "prime", "gcd", "lcm", "string"
        ]

        for kw in keywords:
            features[f"kw_{kw}"] = text.str.count(rf"\b{kw}\b")

        return features.fillna(0)


def explain_prediction(df):
    text = (
        df["title"].astype(str) + " " +
        df["description"].astype(str) + " " +
        df["input_description"].astype(str) + " " +
        df["output_description"].astype(str)
    ).str.lower().iloc[0]

    explanations = []

    keyword_groups = {
        "graph-related keywords": ["graph", "tree", "bfs", "dfs"],
        "dynamic programming keywords": ["dp", "dynamic"],
        "search-related keywords": ["binary search", "search"],
        "string processing keywords": ["string", "substring"]
    }

    for label, keywords in keyword_groups.items():
        found = [kw for kw in keywords if kw in text]
        if found:
            explanations.append(f"{label} detected ({', '.join(found)})")

    if len(text) > 1500:
        explanations.append("Long problem description (higher complexity)")

    digit_count = sum(c.isdigit() for c in text)
    if digit_count > 20:
        explanations.append("Many numeric values / constraints present")

    symbol_count = sum(text.count(s) for s in "+-*/%=")
    if symbol_count > 10:
        explanations.append("High number of mathematical operators")

    if not explanations:
        explanations.append("Simple language and limited constraints detected")

    return explanations
