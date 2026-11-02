"""Fetch the inputs for the HYG / USHY / H0A0 comparison.

Sources
  * ICE BofA US High Yield Index (H0A0): total return index, OAS and effective yield, from FRED
    (series BAMLHYH0A0HYM2TRIV, BAMLH0A0HYM2, BAMLH0A0HYM2EY). FRED currently serves roughly the
    last three years of these ICE series.
  * ETF market prices from Yahoo Finance (yfinance), unadjusted close.
  * HYG and USHY NAV per share, shares outstanding and distributions from the iShares fund download
    (the "Historical" and "Distributions" sheets).
  * SPHY NAV per share and shares outstanding from the State Street (SSGA) NAV history workbook;
    SPHY distributions from Yahoo Finance (these matched the iShares figures to rounding for HYG/USHY).

Usage: python3 fetch_data.py <output_dir>
"""
import os
import re
import subprocess
import sys

import numpy as np
import openpyxl
import pandas as pd
import yfinance as yf

FRED_IDS = {"tri": "BAMLHYH0A0HYM2TRIV", "oas": "BAMLH0A0HYM2", "yld": "BAMLH0A0HYM2EY"}
# iShares portfolio ids: HYG 239565, USHY 291299
ISHARES = {"HYG": 239565, "USHY": 291299}
SSGA_SPHY_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/navhist-us-en-sphy.xlsx"
ISHARES_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document"
    "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US"
    "&portfolioId={pid}&component=fundDownload&userType=individual"
)


def curl(url, out):
    subprocess.run(["curl", "-sS", "-m", "90", "-f", "-o", out, url], check=True)


def ishares_sheets(path):
    """The workbook is SpreadsheetML with unescaped characters in the holdings sheet, so use a regex."""
    text = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r'<ss:Worksheet ss:Name="([^"]*)">(.*?)</ss:Worksheet>', text, re.S):
        rows = []
        for r in re.findall(r"<ss:Row[^>]*>(.*?)</ss:Row>", m.group(2), re.S):
            rows.append([re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<ss:Cell[^>]*>(.*?)</ss:Cell>", r, re.S)])
        out[m.group(1)] = rows
    return out


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def fetch_sphy(out_dir, raw):
    path = os.path.join(raw, "sphy_navhist.xlsx")
    curl(SSGA_SPHY_URL, path)
    rows = [r for r in openpyxl.load_workbook(path, data_only=True).active.iter_rows(values_only=True)]
    hdr = next(i for i, r in enumerate(rows) if r[0] == "Date")
    h = pd.DataFrame([r[:4] for r in rows[hdr + 1:] if r[0]], columns=["date", "nav", "shs", "tna"])
    h["date"] = pd.to_datetime(h["date"], format="%d-%b-%Y", errors="coerce")  # trailing rows are disclaimer text
    h = h.dropna(subset=["date"]).set_index("date").sort_index()[["nav", "shs"]].astype(float)
    px = yf.Ticker("SPHY").history(period="max", auto_adjust=False)
    px.index = px.index.tz_localize(None).normalize()
    daily = h.join(px[["Close"]].rename(columns={"Close": "px"}), how="outer")
    daily.to_pickle(os.path.join(out_dir, "SPHY_daily.pkl"))
    d = px["Dividends"][px["Dividends"] > 0].rename("div").to_frame()
    # Yahoo lists some distributions twice on consecutive days (e.g. 2016-05-02 and 2016-05-03): keep the first
    dup = (d["div"] == d["div"].shift(1)) & (d.index.to_series().diff().dt.days <= 3)
    if dup.any():
        print("dropping duplicate SPHY distributions:", [x.date().isoformat() for x in d.index[dup]])
    d = d[~dup]
    d.to_pickle(os.path.join(out_dir, "SPHY_div.pkl"))
    print("SPHY rows", len(daily), daily.index.min().date(), daily.index.max().date(), "distributions", len(d))


def main(out_dir):
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)

    frames = []
    for col, sid in FRED_IDS.items():
        path = os.path.join(raw, f"{sid}.csv")
        curl(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=1996-01-01", path)
        frames.append(pd.read_csv(path, index_col=0, parse_dates=True, na_values=".").set_axis([col], axis=1))
    pd.concat(frames, axis=1).to_pickle(os.path.join(out_dir, "H0A0.pkl"))

    for t, pid in ISHARES.items():
        path = os.path.join(raw, f"{t.lower()}_fund.xml")
        curl(ISHARES_URL.format(pid=pid), path)
        sh = ishares_sheets(path)
        h = pd.DataFrame(sh["Historical"][1:], columns=sh["Historical"][0])
        h["date"] = pd.to_datetime(h["As Of"], format="%b %d, %Y")
        h = h.set_index("date").sort_index()
        h["nav"] = h["NAV per Share"].map(to_float)
        h["shs"] = h["Shares Outstanding"].map(to_float)
        d = pd.DataFrame(sh["Distributions"][1:], columns=sh["Distributions"][0])
        d["ex"] = pd.to_datetime(d["Ex-Date"], format="%b %d, %Y")
        d["div"] = d["Total Distribution"].map(to_float)
        d = d.set_index("ex").sort_index()[["div"]]
        px = yf.Ticker(t).history(period="max", auto_adjust=False)
        px.index = px.index.tz_localize(None).normalize()
        daily = h[["nav", "shs"]].join(px[["Close"]].rename(columns={"Close": "px"}), how="outer")
        daily.to_pickle(os.path.join(out_dir, f"{t}_daily.pkl"))
        d.to_pickle(os.path.join(out_dir, f"{t}_div.pkl"))
        print(t, "rows", len(daily), daily.index.min().date(), daily.index.max().date(), "distributions", len(d))
    fetch_sphy(out_dir, raw)


if __name__ == "__main__":
    main(sys.argv[1])
