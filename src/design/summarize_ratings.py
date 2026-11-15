"""Average the human ratings. Each rater fills the *_1to5 columns of a COPY of outputs/designs/manifest.csv.
Run: python -m src.design.summarize_ratings rater1.csv rater2.csv ...
Output: reports/design_ratings_summary.csv (mean per style)
"""
import sys
import pandas as pd
from src import config as C

COLS = ["style_match_1to5", "realism_1to5", "layout_kept_1to5"]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.design.summarize_ratings rater1.csv [rater2.csv ...]")
    df = pd.concat([pd.read_csv(p).assign(rater=p) for p in sys.argv[1:]])
    df[COLS] = df[COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=COLS, how="all")
    res = df.groupby("style")[COLS].agg(["mean", "count"]).round(2)
    print(res)
    print("\nOverall mean:\n", df[COLS].mean().round(2))
    res.to_csv(C.ROOT / "reports" / "design_ratings_summary.csv")


if __name__ == "__main__":
    main()
