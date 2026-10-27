"""Build basket_builder.html: embeds the BAML HY positions file into basket_builder.template.html.

Usage: python3 build_basket_builder.py <Positions_in_BAML_HY.xlsx> [as-of label]
"""
import json
import math
import sys

import pandas as pd

FIELDS = [
    ("Isin", "isin"), ("Issuer", "issuer"), ("Security", "sec"), ("KS GICS Sector", "sector"),
    ("IndustryLvl1Desc", "l1"), ("IndustryLvl2Desc", "l2"), ("IndustryLvl3Desc", "l3"), ("IndustryLvl4Desc", "l4"),
    ("Issuer in KS", "inKS"), ("In BAML HY excluding Fossil Fuels", "exFF"), ("Rating Bucket", "rating"),
    ("Iso Country Code", "country"), ("Quarter of Maturity Date", "qtr"), ("Beta", "beta"),
    ("Issuer Weight in BAML HY Index", "wIss"), ("Issuer Weight in BAML HY Index Exc. Fossil Fuels", "wIssX"),
    ("Security Weight in BAML HY Index", "wSec"), ("Security Weight in BAML HY Index Exc. Fossil Fuels", "wSecX"),
    ("Price", "price"), ("Effective Duration", "dur"), ("Effective Yield", "effy"),
    ("Option Adjusted Spread", "oas"), ("Yield To Maturity", "ytm"), ("Yield To Worst", "ytw"),
]


def clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, float):
        return float(f"{v:.10g}")
    if hasattr(v, "item"):
        return clean(v.item())
    return v


def main():
    src = sys.argv[1]
    asof = sys.argv[2] if len(sys.argv) > 2 else ""
    df = pd.read_excel(src)
    missing = [c for c, _ in FIELDS if c not in df.columns]
    if missing:
        sys.exit(f"missing columns: {missing}")
    rows = [[clean(v) for v in r] for r in df[[c for c, _ in FIELDS]].itertuples(index=False, name=None)]
    data = {"asof": asof, "src": src.split("/")[-1], "keys": [k for _, k in FIELDS], "rows": rows}
    tpl = open("basket_builder.template.html", encoding="utf-8").read()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    open("basket_builder.html", "w", encoding="utf-8").write(out)
    print(f"basket_builder.html: {len(rows)} bonds, {len(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
