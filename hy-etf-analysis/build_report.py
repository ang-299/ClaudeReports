"""Build report.html (charts + tables + prose) from analysis.compute().

Usage: python3 build_report.py <data_dir> <output_html>
"""
import base64
import io
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis as an

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", SPHY="#2a78d6", HYG="#eb6834", USHY="#1baf7a", H0A0="#52514e"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34332f", SPHY="#3987e5", HYG="#d95926", USHY="#199e70", H0A0="#c3c2b7"),
}
LABEL = {"SPHY": "SPHY", "HYG": "HYG", "USHY": "USHY", "H0A0": "H0A0"}


def style(ax, th, ylabel=None):
    ax.set_facecolor(th["bg"])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["grid"])
    ax.tick_params(colors=th["ink2"], labelsize=9, length=0)
    ax.grid(axis="y", color=th["grid"], lw=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=th["ink2"], fontsize=9)


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=color, fontsize=9, va="center", annotation_clip=False)


def fig_png(fig, th):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=th["bg"], bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def new_fig(th, rows=1, h=3.6, sharex=True):
    fig, axs = plt.subplots(rows, 1, figsize=(9, h), sharex=sharex, facecolor=th["bg"])
    return fig, (axs if rows > 1 else [axs])


def legend(ax, th, handles_labels=None, loc="upper left"):
    leg = ax.legend(*handles_labels, loc=loc, frameon=False, fontsize=9, ncol=4) if handles_labels else ax.legend(loc=loc, frameon=False, fontsize=9, ncol=4)
    for t in leg.get_texts():
        t.set_color(th["ink"])


def charts(R, th):
    out = {}
    A, etf = R["A"], R["etf"]
    # 1. cumulative NAV total return minus H0A0
    fig, (ax,) = new_fig(th)
    style(ax, th, "NAV total return minus H0A0 (percentage points)")
    for t in an.ETFS:
        r = A[t][0]
        cum = ((1 + r.n).cumprod() - (1 + r.h).cumprod()) * 100
        ax.plot(cum.index, cum, color=th[t], lw=2, label=t)
        end_label(ax, cum.index[-1], cum.iloc[-1], f"{t} {cum.iloc[-1]:+.2f}", th[t])
    ax.axhline(0, color=th["ink2"], lw=1)
    legend(ax, th)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    out["track"] = fig_png(fig, th)

    # 2. premium / discount, small multiples
    fig, axs = new_fig(th, rows=3, h=6.2)
    lo = min(etf[t][0].loc[R["window"][0]:].prem.min() for t in an.ETFS) * 100 - 0.1
    hi = max(etf[t][0].loc[R["window"][0]:].prem.max() for t in an.ETFS) * 100 + 0.1
    for ax, t in zip(axs, an.ETFS):
        style(ax, th)
        p = etf[t][0].loc[R["window"][0]:].prem * 100
        ax.plot(p.index, p, color=th[t], lw=1.1)
        ax.axhline(0, color=th["ink2"], lw=0.8)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{t} premium (+) / discount (-) to NAV, %", loc="left", color=th["ink"], fontsize=10)
        imax = p.idxmax()
        ax.annotate(f"{imax:%d %b %Y}: {p.max():+.2f}%", (imax, p.max()), xytext=(-8, -2), textcoords="offset points", ha="right", va="top", color=th["ink"], fontsize=8.5)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    out["prem"] = fig_png(fig, th)

    # 3. rolling 60-day correlations (two panels)
    fig, axs = new_fig(th, rows=2, h=6.4)
    for ax, col, ttl in [(axs[0], "rc60", "Rolling 60-day correlation: daily price return vs daily NAV return"),
                         (axs[1], "rdp60", "Rolling 60-day correlation: daily change in premium vs daily NAV return")]:
        style(ax, th)
        for t in an.ETFS:
            d = R["own"][t][col].dropna()
            ax.plot(d.index, d, color=th[t], lw=1.3, label=t)
        ax.axhline(0, color=th["ink2"], lw=1)
        ax.set_title(ttl, loc="left", color=th["ink"], fontsize=10)
    legend(axs[0], th, loc="lower right")
    axs[0].set_ylim(-0.3, 1.0)
    axs[1].set_ylim(-0.7, 0.7)
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    out["rolling"] = fig_png(fig, th)

    # 4. distributions as annualised % of spot, with H0A0 effective yield
    fig, (ax,) = new_fig(th)
    style(ax, th, "Annualised % (distribution x 12 / spot)")
    for t in an.ETFS:
        s = R["DV"][t]
        s = s[s.index >= "2019-05-01"]
        ax.plot(s.index, s.ann, color=th[t], lw=1.6, marker="o", ms=2.5, label=f"{t} distribution yield")
    y = R["idx"].yld
    ax.plot(y.index, y, color=th["H0A0"], lw=1.2, ls="--", label="H0A0 effective yield (daily)")
    legend(ax, th)
    ax.set_ylim(3, 10.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    out["div"] = fig_png(fig, th)

    # 5. rolling 21-day volatility, SPHY with H0A0
    fig, (ax,) = new_fig(th)
    style(ax, th, "21-day realised volatility, annualised %")
    rv = R["rollvol"]["SPHY"].dropna()
    ax.plot(rv.index, rv.h, color=th["H0A0"], lw=1.6, ls="--", label="H0A0")
    ax.plot(rv.index, rv.n, color=th["SPHY"], lw=1.8, label="SPHY NAV")
    ax.plot(rv.index, rv.p, color=th["HYG"], lw=1.2, alpha=0.9, label="SPHY price")
    legend(ax, th)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    out["vol"] = fig_png(fig, th)
    return out


def pic(name, charts_l, charts_d, alt):
    return (f'<figure><picture><source media="(prefers-color-scheme: dark)" srcset="data:image/png;base64,{charts_d[name]}">'
            f'<img alt="{alt}" src="data:image/png;base64,{charts_l[name]}"></picture></figure>')


def table(df, cols, heads, fmts, cls=""):
    h = "".join(f"<th>{x}</th>" for x in heads)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c, f in zip(cols, fmts):
            v = r[c]
            cells.append(f"<td>{f.format(v) if not isinstance(v, str) and pd.notna(v) else v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def q(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m].iloc[0]


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e6e5e1;--card:#f5f4f1;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#34332f;--card:#232321;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#34332f;--card:#232321;--accent:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.7rem;line-height:1.25;margin:0 0 6px}h2{font-size:1.2rem;margin:40px 0 8px;padding-top:16px;border-top:1px solid var(--rule)}
.sub{color:var(--ink2);margin:0 0 20px}p{margin:10px 0}ul{padding-left:1.2rem}li{margin:6px 0}
figure{margin:16px 0}figure img{width:100%;height:auto;display:block;border-radius:6px}
.tw{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
thead th{color:var(--ink2);font-weight:600;border-bottom:1px solid var(--ink2);white-space:normal;vertical-align:bottom;line-height:1.3}
table.l2 th:nth-child(2),table.l2 td:nth-child(2){text-align:left}
.note{color:var(--ink2);font-size:.88rem}.cap{color:var(--ink2);font-size:.85rem;margin:-6px 0 14px}
a{color:var(--accent)}code{font-size:.9em}
"""


def build(data_dir, out_path):
    R = an.compute(data_dir)
    CL, CD = charts(R, THEMES["light"]), charts(R, THEMES["dark"])
    C, LL, CU, TD, PR = R["corr"], R["leadlag"], R["catchup"], R["trackdiff"], R["prem"]
    YR, EP, VR, IR, DS, DY, DYL, VO, NOW = R["year"], R["episodes"], R["volreg_own"], R["idxreg"], R["divstats"], R["divyear"], R["divvsyld"], R["vol"], R["now"]
    IS, w0, w1, wn = R["idxstats"], R["window"][0], R["window"][1], R["window"][2]

    def c(t, f, s, k="corr"):
        return q(C, etf=t, freq=f, series=s)[k]
    sp = R["own"]["SPHY"]
    def rng(t, y0, y1, k="corr_px_nav"):
        v = YR[(YR.etf == t) & (YR.year >= y0) & (YR.year <= y1)][k]
        return v.min(), v.max()
    def span(freq, series):
        v = C[(C.freq == freq) & (C.series == series)]["corr"]
        return v.min(), v.max()
    e_lo, e_hi = rng("SPHY", 2012, 2017)
    l_lo, l_hi = rng("SPHY", 2019, 2026)
    p_lo, p_hi = rng("SPHY", 2020, 2026)
    d_lo, d_hi = rng("SPHY", 2012, 2018, "corr_dprem_nav")
    dp_lo, dp_hi = rng("SPHY", 2020, 2026, "corr_dprem_nav")
    def rr(lo_hi):
        a, b = f"{lo_hi[0]:.2f}", f"{lo_hi[1]:.2f}"
        return a if a == b else f"{a} to {b}"
    pd_, pw_, pm_ = rr(span("daily", "price")), rr(span("weekly", "price")), rr(span("monthly", "price"))
    sp_rc_min, sp_rc_min_d = sp.rc60.min(), sp.rc60.idxmin()
    sp_rc_max, sp_rc_max_d = sp.rc60.max(), sp.rc60.idxmax()
    sp_lt05, sp_lt0 = (sp.rc60 < 0.5).mean() * 100, (sp.rc60 < 0).mean() * 100
    sp_dp_neg = (sp.rdp60 < 0).mean() * 100
    hyg, ushy = R["own"]["HYG"], R["own"]["USHY"]
    # distribution statistics
    def ds(t, s):
        return q(DS, etf=t, sample=s)
    L12, WIN, SINCE = "last 12 distributions", "2023-11 onward (H0A0 window)", "since May 2019"
    # trailing twelve distributions over the latest close
    ttm = {}
    for t in an.ETFS:
        d, dv = R["etf"][t]
        ttm[t] = dv.iloc[-12:].sum() / d.px.iloc[-1] * 100
    ydsd = {t: R["DV"][t][R["DV"][t].index >= "2023-11-01"].h_yld.std() for t in an.ETFS}
    yr_tbl = DY[DY.year >= 2020].pivot(index="year", columns="etf", values="ann_mean").reset_index()
    yr_tbl["year"] = yr_tbl.year.astype(str).replace({"2026": "2026 YTD"})

    # ---------- tables ----------
    t_corr = C.copy()
    t_corr["etf_s"] = t_corr.etf + " " + t_corr.series
    t_corr = t_corr.pivot_table(index=["etf", "series", ], columns="freq", values=["corr", "beta", "te"]).reset_index()
    t_corr.columns = ["etf", "series"] + [f"{a}_{b}" for a, b in t_corr.columns[2:]]
    order = {t: i for i, t in enumerate(an.ETFS)}
    t_corr = t_corr.sort_values(by=["etf", "series"], key=lambda s: s.map(order) if s.name == "etf" else s.map({"price": 1, "NAV": 0}))
    t1 = table(t_corr, ["etf", "series", "corr_daily", "corr_weekly", "corr_monthly", "beta_daily", "beta_weekly", "te_daily", "te_weekly", "te_monthly"],
               ["ETF", "Series", "Corr daily", "Corr weekly", "Corr monthly", "Beta daily", "Beta weekly", "TE daily (% ann.)", "TE weekly", "TE monthly"],
               ["{}", "{}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}"])
    ll = LL[~((LL.series == "H0A0 own") & (LL.etf != "SPHY"))].copy()
    ll.loc[ll.series == "H0A0 own", ["etf", "series"]] = ["H0A0", "own return"]
    t2 = table(ll, ["etf", "series", "k=-2", "k=-1", "k=+0", "k=+1", "k=+2"], ["ETF", "Series", "k = -2", "k = -1", "k = 0", "k = +1", "k = +2"], ["{}", "{}"] + ["{:.3f}"] * 5)
    t3 = table(CU, ["etf", "nav_on_p0", "nav_on_p1", "nav_p1_t", "nav_r2", "navh_on_h0", "navh_on_h1", "navh_h1_t"],
               ["ETF", "NAV on price, day t", "NAV on price, day t-1", "t-stat (t-1)", "R²", "NAV on H0A0, day t", "NAV on H0A0, day t-1", "t-stat (t-1)"],
               ["{}", "{:.3f}", "{:.3f}", "{:.1f}", "{:.2f}", "{:.3f}", "{:.3f}", "{:.1f}"])
    td = TD[TD.period.isin(["2024", "2025", "2026", "window"])].copy()
    td["period"] = td.period.replace({"2026": "2026 YTD", "window": "Window total"})
    td["etf"] = pd.Categorical(td.etf, an.ETFS, ordered=True)
    td = td.sort_values(["etf", "period"], key=lambda s: s if s.name == "etf" else s.map(lambda x: {"2024": 0, "2025": 1, "2026 YTD": 2, "Window total": 3}[x]))
    t4 = table(td, ["etf", "period", "h", "nav", "px", "nav_minus_h", "px_minus_h"], ["ETF", "Period", "H0A0 %", "NAV %", "Price %", "NAV minus H0A0 (pp)", "Price minus H0A0 (pp)"],
               ["{}", "{}", "{:.2f}", "{:.2f}", "{:.2f}", "{:+.2f}", "{:+.2f}"])
    pr = PR.copy()
    t5 = table(pr, ["etf", "sample", "mean", "std", "p01", "p99", "min", "max", "ac1", "ac5", "ac21", "abs_gt_50bp"],
               ["ETF", "Sample", "Mean %", "SD %", "1st pct", "99th pct", "Min %", "Max %", "AC 1d", "AC 5d", "AC 21d", "Days with |premium| > 50bp (%)"],
               ["{}", "{}", "{:+.3f}", "{:.3f}", "{:+.2f}", "{:+.2f}", "{:+.2f}", "{:+.2f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.1f}"])
    ye = YR.pivot(index="year", columns="etf", values="corr_px_nav")[an.ETFS].reset_index()
    yd = YR.pivot(index="year", columns="etf", values="corr_dprem_nav")[an.ETFS].reset_index()
    yy = ye.merge(yd, on="year", suffixes=("_c", "_d"))
    yy["year"] = yy.year.astype(str)
    t6 = table(yy.fillna(np.nan).replace({np.nan: "-"}), ["year"] + [f"{t}_c" for t in an.ETFS] + [f"{t}_d" for t in an.ETFS],
               ["Year"] + [f"{t} price vs NAV" for t in an.ETFS] + [f"{t} Δprem. vs NAV" for t in an.ETFS], ["{}"] + ["{:.2f}"] * 6)
    ep = EP.copy()
    ep["etf"] = pd.Categorical(ep.etf, an.ETFS, ordered=True)
    ep = ep.sort_values(["etf"], kind="stable")
    t7 = table(ep, ["etf", "episode", "n", "corr_px_nav", "corr_dprem_nav", "prem_min", "prem_max", "gap_sd_bp"],
               ["ETF", "Episode", "Days", "Price vs NAV", "Δpremium vs NAV", "Min premium %", "Max premium %", "SD of price minus NAV return (bp)"],
               ["{}", "{}", "{:.0f}", "{:.2f}", "{:+.2f}", "{:+.2f}", "{:+.2f}", "{:.0f}"])
    vr = VR.copy()
    vr["band"] = vr.apply(lambda r: f"{r.vol_lo:.1f}-{r.vol_hi:.1f}", axis=1)
    t8 = table(vr, ["etf", "regime", "band", "n", "corr_px_nav", "corr_dprem_nav", "gap_sd_bp", "prem_std"],
               ["ETF", "NAV 21d vol tercile", "NAV vol range (% ann.)", "Days", "Price vs NAV", "Δpremium vs NAV", "SD price minus NAV return (bp)", "SD premium %"],
               ["{}", "{}", "{}", "{:.0f}", "{:.2f}", "{:+.2f}", "{:.0f}", "{:.2f}"])
    ir = IR[IR.by == "H0A0 OAS tercile"].copy()
    ir["band"] = ir.apply(lambda r: f"{r.oas_lo:.2f}-{r.oas_hi:.2f}", axis=1)
    t9 = table(ir, ["etf", "regime", "band", "n", "corr_p_h", "corr_n_h", "corr_p_n", "te_px_bp", "te_nav_bp", "prem_mean", "prem_std"],
               ["ETF", "OAS tercile", "OAS range (%)", "Days", "Price vs H0A0", "NAV vs H0A0", "Price vs NAV", "SD price minus H0A0 (bp)", "SD NAV minus H0A0 (bp)", "Mean premium %", "SD premium %"],
               ["{}", "{}", "{}", "{:.0f}", "{:.2f}", "{:.3f}", "{:.2f}", "{:.1f}", "{:.1f}", "{:+.3f}", "{:.3f}"])
    t10 = table(DS, ["etf", "sample", "n", "mean_pct", "sd_pct", "cv", "min_pct", "max_pct", "ann_mean", "ann_sd", "mom_chg_sd"],
                ["ETF", "Sample", "N", "Mean % of spot", "SD (pp)", "CV", "Min %", "Max %", "Annualised mean %", "Annualised SD (pp)", "SD of month-on-month change in amount (%)"],
                ["{}", "{}", "{:.0f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.2f}", "{:.2f}", "{:.1f}"])
    t11 = table(yr_tbl, ["year"] + an.ETFS, ["Year"] + [f"{t} annualised %" for t in an.ETFS], ["{}"] + ["{:.2f}"] * 3)
    t12 = table(DYL, ["etf", "n", "corr_ann_vs_yld", "corr_chg", "ratio_mean", "ratio_sd"],
                ["ETF", "Distributions", "Corr of level with H0A0 yield", "Corr of change", "Mean ratio to H0A0 yield", "SD of ratio"],
                ["{}", "{:.0f}", "{:.2f}", "{:.2f}", "{:.3f}", "{:.3f}"])
    t13 = table(VO, ["etf", "vol_h", "vol_p", "vol_n", "wk_h", "wk_p", "wk_n", "ac1_h", "ac1_p", "ac1_n", "dd_h", "dd_p", "dd_n"],
                ["ETF", "H0A0 daily vol", "Price daily vol", "NAV daily vol", "H0A0 weekly vol", "Price weekly vol", "NAV weekly vol", "AC1 H0A0", "AC1 price", "AC1 NAV", "Max DD H0A0 %", "Max DD price %", "Max DD NAV %"],
                ["{}"] + ["{:.2f}"] * 6 + ["{:.2f}"] * 3 + ["{:.2f}"] * 3)
    t14 = table(NOW, ["etf", "date", "px", "nav", "prem", "px_1m", "nav_1m"], ["ETF", "As of", "Price", "NAV", "Premium %", "1-month price TR %", "1-month NAV TR %"],
                ["{}", "{}", "{:.2f}", "{:.3f}", "{:+.3f}", "{:+.2f}", "{:+.2f}"])

    L2 = lambda h: h.replace('<table class="">', '<table class="l2">')
    t1, t2, t4, t5, t7, t8, t9, t10, t14 = [L2(x) for x in (t1, t2, t4, t5, t7, t8, t9, t10, t14)]
    ir_ = R["idx_recent"]
    s4 = ir_[ir_.index == pd.Timestamp("2026-09-04")].iloc[0]
    o2 = ir_[ir_.index == pd.Timestamp("2026-10-02")].iloc[0]
    last = ir_.iloc[-1]
    wp = {t: q(PR, etf=t, sample="H0A0 window") for t in an.ETFS}
    fp = {t: q(PR, etf=t, sample="full history") for t in an.ETFS}
    tdw = {t: q(TD, etf=t, period="window") for t in an.ETFS}
    ea = lambda e: q(EP, etf="SPHY", episode=e)
    yrS = lambda y, k: q(YR, etf="SPHY", year=y)[k]
    vo = {t: q(VO, etf=t) for t in an.ETFS}
    vtab = {t: VR[VR.etf == t].set_index("regime") for t in an.ETFS}
    itab = {t: IR[(IR.etf == t) & (IR.by == "H0A0 OAS tercile")].set_index("regime") for t in an.ETFS}
    cu = {t: q(CU, etf=t) for t in an.ETFS}
    dl = {t: q(DYL, etf=t) for t in an.ETFS}
    nowt = {t: q(NOW, etf=t) for t in an.ETFS}

    body = f"""
<h1>SPHY, HYG and USHY against the ICE BofA US High Yield Index (H0A0)</h1>
<p class="sub">ETF prices to 30 Oct 2026; NAV and index to 29 Oct 2026. FRED serves H0A0 history only from {IS['start']:%d %b %Y}, so index-relative statistics use {wn} common trading days ({w0:%d %b %Y} to {w1:%d %b %Y}). Price-versus-NAV statistics use each fund's full history: SPHY from 2012, HYG from 2007, USHY from 2017.</p>

<h2>Summary</h2>
<ul>
<li>SPHY's NAV total return has a daily correlation of {c('SPHY','daily','NAV'):.3f} with H0A0, a beta of {c('SPHY','daily','NAV','beta'):.2f} and annualised tracking error of {c('SPHY','daily','NAV','te'):.2f}%. HYG's figures are {c('HYG','daily','NAV'):.3f}, {c('HYG','daily','NAV','beta'):.2f} and {c('HYG','daily','NAV','te'):.2f}%; USHY's are {c('USHY','daily','NAV'):.3f}, {c('USHY','daily','NAV','beta'):.2f} and {c('USHY','daily','NAV','te'):.2f}%.</li>
<li>Closing prices correlate with H0A0 at {pd_} on daily returns across the three funds, {pw_} weekly and {pm_} monthly.</li>
<li>SPHY traded at an average premium of {wp['SPHY']['mean']:+.2f}% to NAV in the window (standard deviation {wp['SPHY']['std']:.2f}%). The widest premium was {wp['SPHY']['max']:+.2f}% on {wp['SPHY']['max_date']:%d %b %Y} and the widest discount {wp['SPHY']['min']:+.2f}% on {wp['SPHY']['min_date']:%d %b %Y}. Premium deviations have a one-day autocorrelation of {wp['SPHY']['ac1']:.2f} and a five-day autocorrelation of {wp['SPHY']['ac5']:.2f}.</li>
<li>The daily correlation between SPHY's price return and NAV return was between {e_lo:.2f} and {e_hi:.2f} in each year from 2012 to 2017 and between {l_lo:.2f} and {l_hi:.2f} in each year from 2019 to 2026. It is {vtab['SPHY'].loc['high','corr_px_nav']:.2f} on the third of days with the highest NAV volatility, against {vtab['SPHY'].loc['low','corr_px_nav']:.2f} and {vtab['SPHY'].loc['mid','corr_px_nav']:.2f} on the other two thirds.</li>
<li>Over the last twelve distributions SPHY paid {ds('SPHY',L12)['mean_pct']:.3f}% of spot per month ({ds('SPHY',L12)['ann_mean']:.2f}% annualised) with a standard deviation of {ds('SPHY',L12)['sd_pct']:.3f} percentage points, the smallest dispersion of the three funds. Its annualised distribution yield averages {dl['SPHY']['ratio_mean']:.2f} times H0A0's effective yield (HYG {dl['HYG']['ratio_mean']:.2f}, USHY {dl['USHY']['ratio_mean']:.2f}).</li>
<li>Realised daily volatility is {vo['SPHY']['vol_n']:.2f}% for SPHY NAV, {vo['SPHY']['vol_h']:.2f}% for H0A0 and {vo['SPHY']['vol_p']:.2f}% for SPHY's closing price.</li>
</ul>

<h2>Funds and benchmarks</h2>
<p>SPHY (SPDR Portfolio High Yield Bond ETF) lists the ICE BofA US High Yield Index, that is H0A0, as its benchmark; SSGA's page states that its index history before 1 April 2019 reflects a different index, the ICE BofAML US Diversified Crossover Corporate Index. USHY tracks the ICE BofA US High Yield Constrained Index, a variant of the same index family that caps issuer weights; iShares states that USHY has followed the 4pm pricing variant since 1 December 2023 (previously 3pm). HYG tracks the iBoxx USD Liquid High Yield Index, which is a different index from H0A0. The tracking statistics below therefore compare SPHY with its own benchmark and the other two funds with a related index (USHY) and a different one (HYG).</p>
<p>Current readings: H0A0's effective yield was {last.yld:.2f}% and its option-adjusted spread {last.oas:.2f}% on {ir_.index[-1]:%d %b %Y}. On {s4.name:%d %b %Y} they were {s4.yld:.2f}% and {s4.oas:.2f}%, and on {o2.name:%d %b %Y} {o2.yld:.2f}% and {o2.oas:.2f}%, with the total return index falling {(1 - o2.tri / s4.tri) * 100:.1f}% over those four weeks.</p>
{t14}
<p class="note">All three funds close above NAV. The latest premiums sit above the window means ({wp['SPHY']['mean']:+.2f}% SPHY, {wp['HYG']['mean']:+.2f}% HYG, {wp['USHY']['mean']:+.2f}% USHY) by {(nowt['SPHY']['prem'] - wp['SPHY']['mean']) / wp['SPHY']['std']:.1f}, {(nowt['HYG']['prem'] - wp['HYG']['mean']) / wp['HYG']['std']:.1f} and {(nowt['USHY']['prem'] - wp['USHY']['mean']) / wp['USHY']['std']:.1f} standard deviations respectively.</p>

<h2>Correlation with H0A0</h2>
<p>Returns are total returns with each distribution added back on its ex-date, measured on dates when both the index and the ETF have an observation. The annualised tracking error (TE) is the standard deviation of the ETF return minus the H0A0 return.</p>
{t1}
<p>NAV correlations are {c('HYG','daily','NAV'):.3f} to {c('USHY','daily','NAV'):.3f} on daily returns. USHY's NAV is almost indistinguishable from the index, as expected for a Constrained variant of H0A0. SPHY's daily beta of {c('SPHY','daily','NAV','beta'):.2f} lies below one, and its weekly and monthly betas are {c('SPHY','weekly','NAV','beta'):.2f} and {c('SPHY','monthly','NAV','beta'):.2f}. HYG's beta is {c('HYG','daily','NAV','beta'):.2f} daily.</p>
<p>The daily price correlation is close to {c('SPHY','daily','price'):.2f} for all three funds even though the NAV correlations differ, and the daily tracking error of price against H0A0 is {c('USHY','daily','price','te'):.1f}% to {c('HYG','daily','price','te'):.1f}% for each. The gap between price and index therefore comes from the closing price of each fund, with the portfolio contributing little in comparison. Aggregating to weekly returns cuts the price tracking error to about {c('SPHY','weekly','price','te'):.1f}% and monthly returns to about {c('SPHY','monthly','price','te'):.1f}%.</p>

<h3>Timing between price, NAV and the index</h3>
<p>The table shows the correlation between the H0A0 return on day t and the ETF return on day t+k. H0A0's own daily returns have a first-order autocorrelation of {q(LL, etf='SPHY', series='H0A0 own')['k=+1']:.2f}.</p>
{t2}
<p>The price return one day before an index return has a correlation of {q(LL, etf='SPHY', series='price')['k=-1']:.2f} with it, above the {q(LL, etf='SPHY', series='H0A0 own')['k=-1']:.2f} implied by the index's own autocorrelation, while the price return one day after has a correlation of {q(LL, etf='SPHY', series='price')['k=+1']:.2f}. SPHY's NAV correlates {q(LL, etf='SPHY', series='NAV')['k=+1']:.2f} with the index return of the previous day, above the index's own {q(LL, etf='SPHY', series='H0A0 own')['k=+1']:.2f}. Regressions of NAV return on the same-day and prior-day price return give the following coefficients (Newey-West standard errors, 5 lags).</p>
{t3}
<p>For SPHY, about {cu['SPHY']['nav_on_p1']:.2f} of the previous day's price return appears in today's NAV return on top of {cu['SPHY']['nav_on_p0']:.2f} of today's, and the lagged term has a t-statistic of {cu['SPHY']['nav_p1_t']:.1f}. HYG and USHY give similar coefficients. In the regression on H0A0, SPHY's NAV carries a small positive weight of {cu['SPHY']['navh_on_h1']:.2f} on the prior-day index return (t = {cu['SPHY']['navh_h1_t']:.1f}), HYG a negative weight, and USHY none. These patterns are consistent with closing prices absorbing information a day before the bond valuations that feed NAV. I did not verify each provider's valuation time, so the mechanism is an inference from the lag structure.</p>

<h3>Tracking difference over time</h3>
{pic('track', CL, CD, 'Cumulative NAV total return minus H0A0 for SPHY, HYG and USHY')}
<p class="cap">Cumulative NAV total return minus H0A0 total return, percentage points, from {w0:%d %b %Y}.</p>
{t4}
<p>Over the window SPHY's NAV total return exceeded H0A0's by {tdw['SPHY']['nav_minus_h']:+.2f} percentage points ({tdw['SPHY']['nav']:.2f}% against {tdw['SPHY']['h']:.2f}%), USHY's by {tdw['USHY']['nav_minus_h']:+.2f} and HYG's trailed by {abs(tdw['HYG']['nav_minus_h']):.2f}. SPHY's annual differences were small and positive in each year shown. Total return here reinvests each distribution on its ex-date at the fund's own price or NAV, so part of any small difference reflects the index's reinvestment convention and not only fund-level effects. Pensions & Investments lists SPHY's expense ratio at 0.05% and iShares lists USHY's at 0.08%; I did not confirm either figure against a prospectus.</p>

<h2>Premium and discount to NAV</h2>
<p>Daily NAV is published by SSGA for SPHY and by iShares for HYG and USHY, so the premium (closing price over NAV, minus one) can be tracked for every trading day since each fund's launch.</p>
{pic('prem', CL, CD, 'Daily premium and discount to NAV for SPHY, HYG and USHY since November 2023')}
{t5}
<p>In the window the premium of all three funds is positive on average, between {wp['SPHY']['mean']:+.2f}% and {wp['HYG']['mean']:+.2f}%, and its standard deviation is {wp['HYG']['std']:.2f}% to {wp['USHY']['std']:.2f}%. SPHY's premium exceeded 50bp in absolute terms on {wp['SPHY']['abs_gt_50bp']:.1f}% of days, against {wp['HYG']['abs_gt_50bp']:.1f}% for HYG and {wp['USHY']['abs_gt_50bp']:.1f}% for USHY. All three reached their widest premium on 9 April 2025, the day of the tariff pause rally, with SPHY at {wp['SPHY']['max']:+.2f}%, HYG at {wp['HYG']['max']:+.2f}% and USHY at {wp['USHY']['max']:+.2f}%. The premium decays quickly: SPHY's autocorrelation is {wp['SPHY']['ac1']:.2f} at one day, {wp['SPHY']['ac5']:.2f} at five days and {wp['SPHY']['ac21']:.2f} at 21 days.</p>
<p>The full history shows larger excursions. SPHY's premium reached {fp['SPHY']['max']:+.2f}% on {fp['SPHY']['max_date']:%d %b %Y} and {fp['SPHY']['min']:+.2f}% on {fp['SPHY']['min_date']:%d %b %Y}, with a one-day autocorrelation of {fp['SPHY']['ac1']:.2f} and a 21-day autocorrelation of {fp['SPHY']['ac21']:.2f}. HYG's premium reached {fp['HYG']['max']:+.2f}% on {fp['HYG']['max_date']:%d %b %Y} and {fp['HYG']['min']:+.2f}% on {fp['HYG']['min_date']:%d %b %Y}. These extremes appear in both the published NAV and the closing price series, and I did not check them against a third source.</p>
<p>Within the window the premium widens with the level of H0A0's spread. SPHY's premium has a standard deviation of {itab['SPHY'].loc['tight','prem_std']:.2f}% in the tightest third of OAS days ({itab['SPHY'].loc['tight','oas_lo']:.2f}% to {itab['SPHY'].loc['tight','oas_hi']:.2f}%) and {itab['SPHY'].loc['wide','prem_std']:.2f}% in the widest third ({itab['SPHY'].loc['wide','oas_lo']:.2f}% to {itab['SPHY'].loc['wide','oas_hi']:.2f}%).</p>

<h2>Price changes against NAV changes by regime</h2>
<p>The correlation between a fund's daily price return and its daily NAV return varies a great deal across periods. A second statistic, the correlation between the daily change in premium and the NAV return, has a sign that shows which of the two moves further on a given day: a positive value means the price moves more than NAV, and a negative value means it moves less.</p>
{pic('rolling', CL, CD, 'Rolling 60-day correlations between price return, premium change and NAV return')}
{t6}
<p>For SPHY the price-to-NAV correlation was between {e_lo:.2f} and {e_hi:.2f} in each year from 2012 to 2017, rose to {yrS(2018,'corr_px_nav'):.2f} in 2018 and {yrS(2019,'corr_px_nav'):.2f} in 2019, and has stayed between {p_lo:.2f} and {p_hi:.2f} since 2020. The rolling 60-day correlation ranged from {sp_rc_min:.2f} ({sp_rc_min_d:%b %Y}) to {sp_rc_max:.2f} ({sp_rc_max_d:%b %Y}); it was below 0.5 on {sp_lt05:.0f}% of days and below zero on {sp_lt0:.1f}%. HYG's rolling correlation was below zero on {(hyg.rc60 < 0).mean() * 100:.1f}% of days (low {hyg.rc60.min():.2f} in {hyg.rc60.idxmin():%b %Y}) and USHY's never fell below {ushy.rc60.min():.2f}.</p>
<p>The premium-change correlation is negative for SPHY in every year from 2012 to 2018 (between {d_lo:.2f} and {d_hi:.2f}, with {yrS(2015,'corr_dprem_nav'):.2f} in 2015 the most negative) and positive from 2020 onward (between {dp_lo:.2f} and {dp_hi:.2f}). Its 60-day rolling version was negative on {sp_dp_neg:.0f}% of days over the full history. In the earlier years the closing price moved less than NAV on a typical day. Since 2020 it moves more, and the NAV catches up the following day, as the regressions above (2023 to 2026) show.</p>
{t7}
<p>By episode, SPHY's price-to-NAV correlation was lowest in the 2015 to 2016 energy sell-off ({ea('Energy sell-off (Jun 2015-Feb 2016)')['corr_px_nav']:.2f}) and the 2013 taper episode ({ea('Taper tantrum (May-Jul 2013)')['corr_px_nav']:.2f}), and highest during the SVB episode ({ea('SVB (Mar-Apr 2023)')['corr_px_nav']:.2f}), Covid ({ea('Covid (Mar-Apr 2020)')['corr_px_nav']:.2f}), the 2022 rate hikes ({ea('Rate hikes (Jan-Oct 2022)')['corr_px_nav']:.2f}) and the 2025 tariff shock ({ea('Tariff shock (25 Mar-15 May 2025)')['corr_px_nav']:.2f}). The Covid episode also produced the largest premium range in SPHY's history, from {ea('Covid (Mar-Apr 2020)')['prem_min']:+.2f}% to {ea('Covid (Mar-Apr 2020)')['prem_max']:+.2f}%, and the daily standard deviation of price minus NAV return in that episode was {ea('Covid (Mar-Apr 2020)')['gap_sd_bp']:.0f}bp against {ea('Q4 2018')['gap_sd_bp']:.0f}bp in the fourth quarter of 2018.</p>
{t8}
<p>Sorting days by trailing 21-day NAV volatility, SPHY's price-to-NAV correlation is {vtab['SPHY'].loc['low','corr_px_nav']:.2f}, {vtab['SPHY'].loc['mid','corr_px_nav']:.2f} and {vtab['SPHY'].loc['high','corr_px_nav']:.2f} from the calmest to the most volatile third. USHY's rises from {vtab['USHY'].loc['low','corr_px_nav']:.2f} to {vtab['USHY'].loc['high','corr_px_nav']:.2f}. HYG's is flat at about {vtab['HYG'].loc['mid','corr_px_nav']:.2f} across the three groups; its sample starts in 2007 and includes many years with low correlation. In all three funds the standard deviation of price minus NAV return grows with volatility, for SPHY from {vtab['SPHY'].loc['low','gap_sd_bp']:.0f}bp to {vtab['SPHY'].loc['high','gap_sd_bp']:.0f}bp. A plausible reading is that in stress the common move in credit dominates both returns, which lifts the correlation while the absolute gap between price and NAV also grows; I did not test this decomposition.</p>
{t9}
<p>The same pattern appears inside the H0A0 window when days are grouped by the index spread. SPHY's price-to-NAV correlation is {itab['SPHY'].loc['tight','corr_p_n']:.2f}, {itab['SPHY'].loc['mid','corr_p_n']:.2f} and {itab['SPHY'].loc['wide','corr_p_n']:.2f} from tight to wide OAS, its price correlation with H0A0 falls from {itab['SPHY'].loc['tight','corr_p_h']:.2f} to {itab['SPHY'].loc['wide','corr_p_h']:.2f}, and the daily tracking error of price against H0A0 rises from {itab['SPHY'].loc['tight','te_px_bp']:.1f}bp to {itab['SPHY'].loc['wide','te_px_bp']:.1f}bp. SPHY's NAV correlation with H0A0 rises from {itab['SPHY'].loc['tight','corr_n_h']:.3f} to {itab['SPHY'].loc['wide','corr_n_h']:.3f}, and its NAV tracking error widens only from {itab['SPHY'].loc['tight','te_nav_bp']:.1f}bp to {itab['SPHY'].loc['wide','te_nav_bp']:.1f}bp.</p>

<h2>Distributions as a share of spot</h2>
<p>Each fund pays twelve distributions a year; the late-December payment replaces January's. The yield per distribution is the amount divided by the last close before the ex-date, and the annualised figure multiplies it by twelve. SPHY's distribution amounts come from Yahoo Finance and are rounded to three decimals, which adds a sampling noise of roughly 0.001 percentage points to its percentage figures, small against the dispersion reported.</p>
{pic('div', CL, CD, 'Annualised distribution yield on spot for SPHY, HYG and USHY with the H0A0 effective yield')}
{t10}
<p>Since May 2019 SPHY's distribution has averaged {ds('SPHY',SINCE)['mean_pct']:.3f}% of spot ({ds('SPHY',SINCE)['ann_mean']:.2f}% annualised), with a standard deviation of {ds('SPHY',SINCE)['sd_pct']:.3f} percentage points; the comparable figures are {ds('HYG',SINCE)['mean_pct']:.3f}% and {ds('HYG',SINCE)['sd_pct']:.3f} for HYG and {ds('USHY',SINCE)['mean_pct']:.3f}% and {ds('USHY',SINCE)['sd_pct']:.3f} for USHY. Within the H0A0 window SPHY averaged {ds('SPHY',WIN)['mean_pct']:.3f}% ({ds('SPHY',WIN)['ann_mean']:.2f}% annualised) and over the last twelve distributions {ds('SPHY',L12)['mean_pct']:.3f}% ({ds('SPHY',L12)['ann_mean']:.2f}%). The coefficient of variation over the last twelve is {ds('SPHY',L12)['cv']:.3f} for SPHY, {ds('HYG',L12)['cv']:.3f} for HYG and {ds('USHY',L12)['cv']:.3f} for USHY, and the standard deviation of month-on-month change in the dollar amount is {ds('SPHY',L12)['mom_chg_sd']:.1f}%, {ds('HYG',L12)['mom_chg_sd']:.1f}% and {ds('USHY',L12)['mom_chg_sd']:.1f}%.</p>
{t11}
<p>SPHY's annualised yield on spot was {q(DY, etf='SPHY', year=2021)['ann_mean']:.2f}% in 2021, when the average price was {q(DY, etf='SPHY', year=2021)['spot_mean']:.2f} and distributions summed to {q(DY, etf='SPHY', year=2021)['div_ps']:.2f} per share. In 2023 it was {q(DY, etf='SPHY', year=2023)['ann_mean']:.2f}% on an average price of {q(DY, etf='SPHY', year=2023)['spot_mean']:.2f} and {q(DY, etf='SPHY', year=2023)['div_ps']:.2f} per share, and in 2026 to date {q(DY, etf='SPHY', year=2026)['ann_mean']:.2f}%. On the latest close, the twelve most recent distributions amount to {ttm['SPHY']:.2f}% of price for SPHY, {ttm['HYG']:.2f}% for HYG and {ttm['USHY']:.2f}% for USHY.</p>
{t12}
<p>Against H0A0's effective yield, sampled on each ex-date, SPHY's annualised distribution yield has a level correlation of {dl['SPHY']['corr_ann_vs_yld']:.2f} and a mean ratio of {dl['SPHY']['ratio_mean']:.2f} (standard deviation {dl['SPHY']['ratio_sd']:.3f}). HYG's correlation is {dl['HYG']['corr_ann_vs_yld']:.2f} with a ratio of {dl['HYG']['ratio_mean']:.2f}, and USHY's {dl['USHY']['corr_ann_vs_yld']:.2f} with {dl['USHY']['ratio_mean']:.2f}. The standard deviation of the annualised distribution yield over the window is {ds('SPHY',WIN)['ann_sd']:.2f} percentage points for SPHY against {ydsd['SPHY']:.2f} for H0A0's yield on the same dates; HYG is {ds('HYG',WIN)['ann_sd']:.2f} against {ydsd['HYG']:.2f} and USHY {ds('USHY',WIN)['ann_sd']:.2f} against {ydsd['USHY']:.2f}. Distributions reflect coupon income less fees, while the effective yield also includes the pull to redemption value, so the two measures are defined differently and the ratio is a descriptive figure.</p>

<h2>Volatility</h2>
<p>Volatility figures are annualised standard deviations of daily or weekly total returns over the {wn}-day window. Weekly returns use Friday-to-Friday levels.</p>
{t13}
<p>On daily returns the three NAVs have volatilities of {vo['SPHY']['vol_n']:.2f}% (SPHY), {vo['HYG']['vol_n']:.2f}% (HYG) and {vo['USHY']['vol_n']:.2f}% (USHY) against {vo['SPHY']['vol_h']:.2f}% for H0A0, and the closing prices {vo['SPHY']['vol_p']:.2f}%, {vo['HYG']['vol_p']:.2f}% and {vo['USHY']['vol_p']:.2f}%. H0A0 has a first-order daily autocorrelation of {vo['SPHY']['ac1_h']:.2f}, so its weekly volatility of {vo['SPHY']['wk_h']:.2f}% is above the daily figure. The closing price has a first-order autocorrelation of {vo['SPHY']['ac1_p']:.2f} for SPHY, so premium noise that reverses within days adds to daily variance and washes out at longer horizons; at weekly frequency SPHY's price volatility is {vo['SPHY']['wk_p']:.2f}% against {vo['SPHY']['wk_n']:.2f}% for NAV.</p>
{pic('vol', CL, CD, '21-day realised volatility for H0A0, SPHY NAV and SPHY price')}
<p>Over the window H0A0's option-adjusted spread averaged {IS['oas_mean']:.2f}% with a standard deviation of {IS['oas_sd']:.2f}%, ranging from {IS['oas_min']:.2f}% ({IS['oas_min_d']:%d %b %Y}) to {IS['oas_max']:.2f}% ({IS['oas_max_d']:%d %b %Y}). Its effective yield averaged {IS['yld_mean']:.2f}% (standard deviation {IS['yld_sd']:.2f}%, range {IS['yld_min']:.2f}% to {IS['yld_max']:.2f}%). Daily changes have a standard deviation of {IS['doas_sd_bp']:.1f}bp for the spread and {IS['dyld_sd_bp']:.1f}bp for the yield. The worst daily index returns were on 4 April 2025 ({R['idx'].tri.pct_change().loc['2025-04-04'] * 100:.2f}%), 3 April 2025 ({R['idx'].tri.pct_change().loc['2025-04-03'] * 100:.2f}%) and 7 April 2025 ({R['idx'].tri.pct_change().loc['2025-04-07'] * 100:.2f}%).</p>

<h2>Data and method</h2>
<ul>
<li>H0A0 total return index, option-adjusted spread and effective yield: FRED series <a href="https://fred.stlouisfed.org/series/BAMLHYH0A0HYM2TRIV">BAMLHYH0A0HYM2TRIV</a>, <a href="https://fred.stlouisfed.org/series/BAMLH0A0HYM2">BAMLH0A0HYM2</a> and <a href="https://fred.stlouisfed.org/series/BAMLH0A0HYM2EY">BAMLH0A0HYM2EY</a>, ICE data. FRED provides about three years of history, which limits every index-relative statistic. The pricing time of the ICE standard variant changed on 1 December 2023 according to iShares; I made no adjustment for it, which at most affects the first month.</li>
<li>SPHY NAV and shares outstanding: <a href="https://www.ssga.com/library-content/products/fund-data/etfs/us/navhist-us-en-sphy.xlsx">SSGA NAV history</a>; fund details on the <a href="https://www.ssga.com/us/en/individual/etfs/spdr-portfolio-high-yield-bond-etf-sphy">SSGA SPHY page</a>. HYG and USHY NAV and distributions: the iShares fund download for <a href="https://www.ishares.com/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf">HYG</a> and <a href="https://www.ishares.com/us/products/291299/ishares-broad-usd-high-yield-corporate-bond-etf">USHY</a>. Closing prices and SPHY distributions: Yahoo Finance via yfinance, unadjusted close.</li>
<li>iShares and Yahoo distribution amounts agree to within 0.0006 per share for HYG and USHY. A duplicate SPHY distribution listed on consecutive days in May 2016 was removed. The late-October 2026 NAV for SPHY is missing for 30 October, so SPHY statistics end on 29 October.</li>
<li>Correlations of returns across index and ETF use total-return levels on common dates, because the FRED file includes weekends and bond-market holidays on which the ETFs do not trade. Price-versus-NAV statistics use each ETF's own trading days and exclude non-trading NAV strikes such as Good Friday 2024.</li>
<li>Tercile regimes are formed by sorting days on the trailing 21-day NAV volatility (full history) or on the index OAS level (window). Standard errors in the NAV regressions are Newey-West with five lags. Episode dates are chosen by hand and are listed in the table.</li>
<li>Past statistics describe this sample. Three years of index history is short for regime comparisons: the window contains one stress episode (April 2025) and a smaller sell-off in September and October 2026. This report is analysis for internal discussion and is not investment advice.</li>
</ul>
"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPHY HYG USHY vs H0A0</title><style>{CSS}</style></head><body><main>{body}</main></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return R


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
