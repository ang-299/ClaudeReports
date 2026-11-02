"""SPHY / HYG / USHY vs ICE BofA US High Yield Index (H0A0): correlation, NAV vs price regimes, distributions, volatility.

Return conventions
  * ETF total return: the distribution is added back on the ex-date, to both price and NAV.
  * Index-relative statistics use total-return levels restricted to dates on which the index and the
    ETF both have an observation, so each return covers the same horizon. (The FRED file includes
    weekends and bond-market holidays on which the ETFs do not trade.)
  * Price-versus-NAV statistics use each ETF's own trading days.

Usage: python3 analysis.py <data_dir>   (prints tables; build_report.py imports compute())
"""
import sys

import numpy as np
import pandas as pd

ETFS = ["SPHY", "HYG", "USHY"]
ANN = {"D": 252, "W": 52, "M": 12}


def load(data_dir):
    idx = pd.read_pickle(f"{data_dir}/H0A0.pkl").dropna()
    etf = {}
    for t in ETFS:
        d = pd.read_pickle(f"{data_dir}/{t}_daily.pkl")
        dv = pd.read_pickle(f"{data_dir}/{t}_div.pkl")["div"]
        last = min(d.px.last_valid_index(), d.nav.last_valid_index())
        d = d[d.index <= last].dropna(subset=["px", "nav"]).copy()  # drops non-trading NAV strikes (e.g. Good Friday)
        dv = dv[dv.index <= last]
        d["div"] = dv.reindex(d.index).fillna(0.0)
        d["pxr"] = (d.px + d["div"]) / d.px.shift(1) - 1
        d["navr"] = (d.nav + d["div"]) / d.nav.shift(1) - 1
        d["prem"] = d.px / d.nav - 1
        d["pxtr"] = (1 + d.pxr.fillna(0)).cumprod()
        d["navtr"] = (1 + d.navr.fillna(0)).cumprod()
        etf[t] = (d, dv)
    return idx, etf


def aligned(idx, d):
    """Total-return levels on the dates common to index and ETF, then returns on that grid."""
    j = idx[["tri", "oas", "yld"]].join(d[["pxtr", "navtr", "px", "nav", "prem"]], how="inner")
    out = pd.DataFrame({"h": j.tri.pct_change(), "p": j.pxtr.pct_change(), "n": j.navtr.pct_change()})
    out["oas"], out["yld"], out["prem"] = j.oas, j.yld, j.prem
    return out.dropna(subset=["h", "p", "n"]), j


def periodic(j, rule):
    z = j[["tri", "pxtr", "navtr"]].resample(rule).last().pct_change().dropna()
    return z.rename(columns={"tri": "h", "pxtr": "p", "navtr": "n"})


def maxdd(r):
    lv = (1 + r).cumprod()
    return (lv / lv.cummax() - 1).min()


def compute(data_dir):
    idx, etf = load(data_dir)
    R = {"idx": idx, "etf": etf}
    A = {t: aligned(idx, etf[t][0]) for t in ETFS}
    R["A"] = A
    R["window"] = (A["HYG"][0].index.min(), A["HYG"][0].index.max(), len(A["HYG"][0]))

    # 1. correlation, beta, tracking error versus H0A0 at three horizons
    rows = []
    for t in ETFS:
        r, j = A[t]
        for freq, rule in [("daily", None), ("weekly", "W-FRI"), ("monthly", "ME")]:
            z = r[["h", "p", "n"]] if rule is None else periodic(j, rule)
            k = ANN[{"daily": "D", "weekly": "W", "monthly": "M"}[freq]]
            for lab, c in [("price", "p"), ("NAV", "n")]:
                rows.append(dict(etf=t, freq=freq, series=lab, corr=z.h.corr(z[c]), beta=np.polyfit(z.h, z[c], 1)[0],
                                 te=(z[c] - z.h).std() * np.sqrt(k) * 100, n=len(z)))
    R["corr"] = pd.DataFrame(rows)

    # 2. lead/lag: corr(H0A0 return at t, ETF return at t+k); k>0 means the ETF moves after the index
    rows = []
    for t in ETFS:
        r, _ = A[t]
        for lab, c in [("price", "p"), ("NAV", "n")]:
            rows.append(dict(etf=t, series=lab, **{f"k={k:+d}": r.h.corr(r[c].shift(-k)) for k in range(-2, 3)}))
        rows.append(dict(etf=t, series="H0A0 own", **{f"k={k:+d}": r.h.corr(r.h.shift(-k)) for k in range(-2, 3)}))
    R["leadlag"] = pd.DataFrame(rows)

    # NAV catch-up regression: NAV return on same-day and prior-day price return (and index)
    import statsmodels.api as sm
    rows = []
    for t in ETFS:
        r, _ = A[t]
        x = pd.DataFrame({"p0": r.p, "p1": r.p.shift(1)}).dropna()
        m = sm.OLS(r.n.loc[x.index], sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        x2 = pd.DataFrame({"h0": r.h, "h1": r.h.shift(1)}).dropna()
        m2 = sm.OLS(r.p.loc[x2.index], sm.add_constant(x2)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        m3 = sm.OLS(r.n.loc[x2.index], sm.add_constant(x2)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append(dict(etf=t, nav_on_p0=m.params.p0, nav_on_p1=m.params.p1, nav_p1_t=m.tvalues.p1, nav_r2=m.rsquared,
                         px_on_h0=m2.params.h0, px_on_h1=m2.params.h1, px_h1_t=m2.tvalues.h1,
                         navh_on_h0=m3.params.h0, navh_on_h1=m3.params.h1, navh_h1_t=m3.tvalues.h1))
    R["catchup"] = pd.DataFrame(rows)

    # 3. tracking difference of NAV total return versus H0A0, by calendar year and over the window
    rows = []
    for t in ETFS:
        r, _ = A[t]
        for y, g in r.groupby(r.index.year):
            rows.append(dict(etf=t, period=str(y), days=len(g), h=((1 + g.h).prod() - 1) * 100, nav=((1 + g.n).prod() - 1) * 100,
                             px=((1 + g.p).prod() - 1) * 100))
        rows.append(dict(etf=t, period="window", days=len(r), h=((1 + r.h).prod() - 1) * 100, nav=((1 + r.n).prod() - 1) * 100,
                         px=((1 + r.p).prod() - 1) * 100))
    td = pd.DataFrame(rows)
    td["nav_minus_h"] = td.nav - td.h
    td["px_minus_h"] = td.px - td.h
    R["trackdiff"] = td
    R["trackdiff_daily"] = {t: (A[t][0].n - A[t][0].h) for t in ETFS}

    # 4. premium / discount (ETF's own trading days, full history and index window)
    rows = []
    for t in ETFS:
        d = etf[t][0]
        for lab, g in [("full history", d), ("H0A0 window", d[d.index >= A[t][0].index.min()])]:
            p = g.prem * 100
            rows.append(dict(etf=t, sample=lab, n=len(p), mean=p.mean(), median=p.median(), std=p.std(), p01=p.quantile(.01), p99=p.quantile(.99),
                             min=p.min(), min_date=p.idxmin().date(), max=p.max(), max_date=p.idxmax().date(),
                             ac1=p.autocorr(1), ac5=p.autocorr(5), ac21=p.autocorr(21), abs_gt_50bp=(p.abs() > .5).mean() * 100,
                             dprem_vol_bp=p.diff().std() * 100))
    R["prem"] = pd.DataFrame(rows)

    # 5. price-vs-NAV co-movement by regime
    def own(t):
        d = etf[t][0].dropna(subset=["pxr", "navr"]).copy()
        d["vol21"] = d.navr.rolling(21).std() * np.sqrt(252) * 100
        d["rc60"] = d.pxr.rolling(60).corr(d.navr)
        d["rdp60"] = d.prem.diff().rolling(60).corr(d.navr)
        return d
    R["own"] = {t: own(t) for t in ETFS}
    rows = []
    for t in ETFS:
        d = R["own"][t]
        for y, g in d.groupby(d.index.year):
            if len(g) > 100:
                rows.append(dict(etf=t, year=y, corr_px_nav=g.pxr.corr(g.navr), corr_dprem_nav=g.prem.diff().corr(g.navr),
                                 prem_mean=g.prem.mean() * 100, prem_std=g.prem.std() * 100, prem_min=g.prem.min() * 100, prem_max=g.prem.max() * 100,
                                 gap_sd_bp=(g.pxr - g.navr).std() * 1e4))
    R["year"] = pd.DataFrame(rows)
    eps = {"GFC (Sep 2008-Mar 2009)": ("2008-09-01", "2009-03-31"), "Taper tantrum (May-Jul 2013)": ("2013-05-01", "2013-07-31"), "Energy sell-off (Jun 2015-Feb 2016)": ("2015-06-01", "2016-02-29"),
           "Q4 2018": ("2018-10-01", "2018-12-31"), "Covid (Mar-Apr 2020)": ("2020-03-01", "2020-04-30"),
           "Rate hikes (Jan-Oct 2022)": ("2022-01-01", "2022-10-31"), "SVB (Mar-Apr 2023)": ("2023-03-01", "2023-04-30"),
           "Tariff shock (25 Mar-15 May 2025)": ("2025-03-25", "2025-05-15")}
    rows = []
    for t in ETFS:
        d = R["own"][t]
        for k, (a, b) in eps.items():
            g = d[a:b]
            if len(g) > 10:
                rows.append(dict(etf=t, episode=k, n=len(g), corr_px_nav=g.pxr.corr(g.navr), corr_dprem_nav=g.prem.diff().corr(g.navr),
                                 prem_min=g.prem.min() * 100, prem_max=g.prem.max() * 100, gap_sd_bp=(g.pxr - g.navr).std() * 1e4))
    R["episodes"] = pd.DataFrame(rows)
    rows = []
    for t in ETFS:
        d = R["own"][t].dropna(subset=["vol21"]).copy()
        d["reg"] = pd.qcut(d.vol21, 3, labels=["low", "mid", "high"])
        for r_, g in d.groupby("reg", observed=True):
            rows.append(dict(etf=t, regime=r_, vol_lo=g.vol21.min(), vol_hi=g.vol21.max(), n=len(g), corr_px_nav=g.pxr.corr(g.navr),
                             corr_dprem_nav=g.prem.diff().corr(g.navr), gap_sd_bp=(g.pxr - g.navr).std() * 1e4, prem_std=g.prem.std() * 100))
    R["volreg_own"] = pd.DataFrame(rows)

    # within the H0A0 window, regimes defined by the index itself
    rows = []
    for t in ETFS:
        r, j = A[t]
        own_d = etf[t][0]
        f = r.join(own_d[["pxr", "navr"]].rename(columns={"pxr": "p1", "navr": "n1"}), how="left")  # own-day returns for premium work
        f["hv"] = r.h.rolling(21).std() * np.sqrt(252) * 100
        f["oasreg"] = pd.qcut(f.oas, 3, labels=["tight", "mid", "wide"])
        f["hvreg"] = pd.qcut(f.hv, 3, labels=["low", "mid", "high"])
        pr = own_d.prem.reindex(f.index)
        for col, nm in [("oasreg", "H0A0 OAS tercile"), ("hvreg", "H0A0 21d vol tercile")]:
            for r_, g in f.dropna(subset=[col]).groupby(col, observed=True):
                rows.append(dict(etf=t, by=nm, regime=r_, n=len(g), corr_p_h=g.p.corr(g.h), corr_n_h=g.n.corr(g.h), corr_p_n=g.p.corr(g.n),
                                 te_nav_bp=(g.n - g.h).std() * 1e4, te_px_bp=(g.p - g.h).std() * 1e4, prem_mean=pr.loc[g.index].mean() * 100,
                                 prem_std=pr.loc[g.index].std() * 100, oas_lo=g.oas.min(), oas_hi=g.oas.max(), hv_lo=g.hv.min(), hv_hi=g.hv.max()))
    R["idxreg"] = pd.DataFrame(rows)

    # 6. distributions as % of spot (spot = last close before the ex-date)
    DV, rows = {}, []
    for t in ETFS:
        d, dv = etf[t]
        prior = d.px.shift(1)
        s = pd.DataFrame({"div": dv})
        s["spot"] = [prior.loc[:x].dropna().iloc[-1] for x in s.index]
        s["pct"] = s["div"] / s.spot * 100
        s["ann"] = s.pct * 12
        s["ttm"] = s["div"].rolling(12).sum() / s.spot * 100   # trailing twelve distributions over spot at the latest ex-date
        s["h_yld"] = idx.yld.reindex(s.index, method="ffill").where(s.index >= idx.index.min())
        DV[t] = s
        for lab, g in [("since May 2019", s[s.index >= "2019-05-01"]), ("2023-11 onward (H0A0 window)", s[s.index >= "2023-11-01"]),
                       ("last 12 distributions", s.iloc[-12:])]:
            rows.append(dict(etf=t, sample=lab, n=len(g), mean_pct=g.pct.mean(), sd_pct=g.pct.std(), cv=g.pct.std() / g.pct.mean(), min_pct=g.pct.min(),
                             max_pct=g.pct.max(), ann_mean=g.ann.mean(), ann_sd=g.ann.std(), mom_chg_sd=g["div"].pct_change().std() * 100))
    R["DV"] = DV
    R["divstats"] = pd.DataFrame(rows)
    rows = []
    for t in ETFS:
        s = DV[t]
        for y, g in s[s.index >= "2018-01-01"].groupby(s[s.index >= "2018-01-01"].index.year):
            rows.append(dict(etf=t, year=y, n=len(g), ann_mean=g.ann.mean(), sd_pct=g.pct.std(), div_ps=g["div"].sum(), spot_mean=g.spot.mean()))
    R["divyear"] = pd.DataFrame(rows)
    rows = []
    for t in ETFS:
        s = DV[t][DV[t].index >= "2023-11-01"].dropna(subset=["h_yld"])
        rows.append(dict(etf=t, n=len(s), corr_ann_vs_yld=s.ann.corr(s.h_yld), corr_chg=s.ann.diff().corr(s.h_yld.diff()), ratio_mean=(s.ann / s.h_yld).mean(),
                         ratio_sd=(s.ann / s.h_yld).std(), ttm_now=s.ttm.iloc[-1], yld_now=s.h_yld.iloc[-1], ttm_first=s.ttm.dropna().iloc[0], yld_first=s.h_yld.iloc[0]))
    R["divvsyld"] = pd.DataFrame(rows)

    # 7. volatility
    rows = []
    for t in ETFS:
        r, j = A[t]
        wk = periodic(j, "W-FRI")
        av = lambda x, k=252: x.std() * np.sqrt(k) * 100
        rows.append(dict(etf=t, vol_h=av(r.h), vol_p=av(r.p), vol_n=av(r.n), wk_h=av(wk.h, 52), wk_p=av(wk.p, 52), wk_n=av(wk.n, 52),
                         ac1_h=r.h.autocorr(1), ac1_p=r.p.autocorr(1), ac1_n=r.n.autocorr(1),
                         dd_h=maxdd(r.h) * 100, dd_p=maxdd(r.p) * 100, dd_n=maxdd(r.n) * 100,
                         ret_h=((1 + r.h).prod() ** (252 / len(r)) - 1) * 100, ret_p=((1 + r.p).prod() ** (252 / len(r)) - 1) * 100,
                         ret_n=((1 + r.n).prod() ** (252 / len(r)) - 1) * 100))
    R["vol"] = pd.DataFrame(rows)
    roll = {}
    for t in ETFS:
        r, _ = A[t]
        roll[t] = pd.DataFrame({"h": r.h.rolling(21).std(), "p": r.p.rolling(21).std(), "n": r.n.rolling(21).std()}) * np.sqrt(252) * 100
    R["rollvol"] = roll
    rows = []
    for t in ETFS:
        d, _ = etf[t]
        last = d.index[-1]
        m1 = d.index[d.index <= last - pd.Timedelta(days=30)][-1]
        rows.append(dict(etf=t, date=last.date(), px=d.px.iloc[-1], nav=d.nav.iloc[-1], prem=d.prem.iloc[-1] * 100,
                         px_1m=(d.pxtr.iloc[-1] / d.pxtr.loc[m1] - 1) * 100, nav_1m=(d.navtr.iloc[-1] / d.navtr.loc[m1] - 1) * 100))
    R["now"] = pd.DataFrame(rows)
    R["idx_recent"] = idx[["tri", "oas", "yld"]].iloc[-130:]
    R["idxstats"] = dict(oas_mean=idx.oas.mean(), oas_sd=idx.oas.std(), oas_min=idx.oas.min(), oas_min_d=idx.oas.idxmin().date(), oas_max=idx.oas.max(),
                         oas_max_d=idx.oas.idxmax().date(), yld_mean=idx.yld.mean(), yld_sd=idx.yld.std(), yld_min=idx.yld.min(), yld_max=idx.yld.max(),
                         yld_last=idx.yld.iloc[-1], oas_last=idx.oas.iloc[-1], doas_sd_bp=idx.oas.diff().std() * 100, dyld_sd_bp=idx.yld.diff().std() * 100,
                         start=idx.index.min().date(), end=idx.index.max().date())
    return R


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    R = compute(sys.argv[1])
    print("window", R["window"])
    for k in ["corr", "leadlag", "catchup", "trackdiff", "prem", "year", "episodes", "volreg_own", "idxreg", "divstats", "divyear", "divvsyld", "vol"]:
        print("\n==", k)
        print(R[k].round(3).to_string())
    print(R["idxstats"])
    for t in ETFS:
        d = R["own"][t]
        print(t, "rolling 60d corr px/NAV: min", round(d.rc60.min(), 3), d.rc60.idxmin().date(), "max", round(d.rc60.max(), 3), d.rc60.idxmax().date(),
              "share<0.5", round((d.rc60 < .5).mean(), 3), "share<0", round((d.rc60 < 0).mean(), 4))
        print(t, "rolling 60d corr dprem/NAV: min", round(d.rdp60.min(), 3), d.rdp60.idxmin().date(), "max", round(d.rdp60.max(), 3), d.rdp60.idxmax().date(),
              "share<0", round((d.rdp60 < 0).mean(), 3))
