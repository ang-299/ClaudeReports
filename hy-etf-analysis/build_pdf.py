"""Build findings.pdf: a short, number-first summary of the SPHY / HYG / USHY vs H0A0 analysis.

Usage: python3 build_pdf.py <data_dir> <output_pdf>
"""
import base64
import io
import sys

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

import analysis as an
import build_report as br

INK, GREY, RULE, BLUE = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d3"), colors.HexColor("#2a78d6")
W, H = letter
M = 0.75 * inch
CW = W - 2 * M

ST = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=10, leading=13, textColor=GREY, spaceAfter=10),
    "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=INK, spaceBefore=12, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "bul": ParagraphStyle("bul", fontName="Helvetica", fontSize=10, leading=14, textColor=INK, leftIndent=12, bulletIndent=0, spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName="Helvetica", fontSize=8, leading=10.5, textColor=GREY, spaceAfter=6),
    "big": ParagraphStyle("big", fontName="Helvetica-Bold", fontSize=24, leading=27, textColor=BLUE),
    "tl": ParagraphStyle("tl", fontName="Helvetica", fontSize=8.5, leading=11, textColor=GREY),
    "th": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=GREY),
    "td": ParagraphStyle("td", fontName="Helvetica", fontSize=8.5, leading=11, textColor=INK),
}


def P(t, s="body"):
    return Paragraph(t, ST[s])


def bullets(items):
    return [Paragraph(t, ST["bul"], bulletText="•") for t in items]


def tiles(items):
    cells = [[Paragraph(v, ST["big"]), Paragraph(l, ST["tl"])] for v, l in items]
    t = Table([[ [c[0], c[1]] for c in cells ]], colWidths=[CW / len(items)] * len(items))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 1.2, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    return t


def grid(head, rows, widths, left=1):
    data = [[Paragraph(h, ST["th"]) for h in head]] + [[Paragraph(str(c), ST["td"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, GREY), ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def img(b64, width=CW, max_h=None):
    from PIL import Image as PI
    raw = base64.b64decode(b64)
    w, h = PI.open(io.BytesIO(raw)).size
    hh = width * h / w
    if max_h and hh > max_h:
        width, hh = width * max_h / hh, max_h
    return Image(io.BytesIO(raw), width=width, height=hh)


def q(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m].iloc[0]


def page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(M, 0.5 * inch, "Sources: FRED (ICE), SSGA, iShares, Yahoo Finance. Analysis, not investment advice.")
    canvas.drawRightString(W - M, 0.5 * inch, f"{doc.page}")
    canvas.restoreState()


def build(data_dir, out):
    R = an.compute(data_dir)
    CH = br.charts(R, br.THEMES["light"])
    C, PR, CU, TD, YR, EP, VR, IR, DS, DYL, VO, NOW = (R[k] for k in ["corr", "prem", "catchup", "trackdiff", "year", "episodes", "volreg_own", "idxreg", "divstats", "divvsyld", "vol", "now"])
    IS, w0, w1, wn = R["idxstats"], R["window"][0], R["window"][1], R["window"][2]
    c = lambda t, f, s, k="corr": q(C, etf=t, freq=f, series=s)[k]
    wp = {t: q(PR, etf=t, sample="H0A0 window") for t in an.ETFS}
    fp = {t: q(PR, etf=t, sample="full history") for t in an.ETFS}
    ds = lambda t, s: q(DS, etf=t, sample=s)
    L12 = "last 12 distributions"
    vo = {t: q(VO, etf=t) for t in an.ETFS}
    dl = {t: q(DYL, etf=t) for t in an.ETFS}
    tdw = {t: q(TD, etf=t, period="window") for t in an.ETFS}
    cu = {t: q(CU, etf=t) for t in an.ETFS}
    nowt = {t: q(NOW, etf=t) for t in an.ETFS}
    sp = R["own"]["SPHY"]
    vt = VR[VR.etf == "SPHY"].set_index("regime")
    it = IR[(IR.etf == "SPHY") & (IR.by == "H0A0 OAS tercile")].set_index("regime")
    yr = lambda y, k="corr_px_nav": q(YR, etf="SPHY", year=y)[k]
    rng = lambda y0, y1, k="corr_px_nav": (YR[(YR.etf == "SPHY") & (YR.year >= y0) & (YR.year <= y1)][k].min(), YR[(YR.etf == "SPHY") & (YR.year >= y0) & (YR.year <= y1)][k].max())
    pn = [c(t, "daily", "price") for t in an.ETFS]
    story = []

    # ---------------- page 1 ----------------
    story += [P("SPHY, HYG and USHY against H0A0", "title"),
              P(f"ICE BofA US High Yield Index comparison. Index statistics use {wn} trading days, {w0:%d %b %Y} to {w1:%d %b %Y}, the full span FRED provides. Price-versus-NAV statistics use each fund's full history (HYG from 2007, SPHY from 2012, USHY from 2017).", "sub"),
              tiles([(f"{c('SPHY','daily','NAV'):.3f}", "SPHY NAV, daily correlation with H0A0"),
                     (f"{min(pn):.2f}-{max(pn):.2f}", "Closing-price daily correlation with H0A0, three funds"),
                     (f"{wp['SPHY']['mean']:+.2f}%", f"SPHY average premium to NAV (standard deviation {wp['SPHY']['std']:.2f}%)"),
                     (f"{ds('SPHY',L12)['ann_mean']:.1f}%", "SPHY distributions, last 12 months, annualised on spot")]),
              P("Correlation with H0A0", "h"),
              P("SPHY lists H0A0 as its benchmark. USHY follows the Constrained variant of the same index family, and HYG follows the iBoxx USD Liquid High Yield Index."),
              grid(["ETF", "NAV correlation (daily)", "NAV beta", "NAV tracking error (% a year)", "Price correlation (daily / weekly / monthly)", "Price tracking error (daily, % a year)"],
                   [[t, f"{c(t,'daily','NAV'):.3f}", f"{c(t,'daily','NAV','beta'):.2f}", f"{c(t,'daily','NAV','te'):.2f}",
                     f"{c(t,'daily','price'):.2f} / {c(t,'weekly','price'):.2f} / {c(t,'monthly','price'):.2f}", f"{c(t,'daily','price','te'):.2f}"] for t in an.ETFS],
                   [0.55 * inch, 1.05 * inch, 0.7 * inch, 1.2 * inch, 1.9 * inch, 1.6 * inch]),
              Spacer(1, 6)]
    story += bullets([
        f"NAV tracks the index closely. SPHY's beta of {c('SPHY','daily','NAV','beta'):.2f} means a 1% index move comes with about a {c('SPHY','daily','NAV','beta'):.2f}% NAV move; HYG's is {c('HYG','daily','NAV','beta'):.2f}.",
        f"Closing prices track less tightly: about {np.mean(pn):.2f} on daily returns for all three funds, rising to {c('SPHY','weekly','price'):.2f} weekly and {c('SPHY','monthly','price'):.2f} monthly. The funds' NAVs differ in how well they follow the index but their prices do not, which places the gap in the closing price.",
        f"Prices appear to move about a day ahead of NAV. About a quarter ({cu['SPHY']['nav_on_p1']:.2f}) of yesterday's SPHY price return shows up in today's NAV return (t-statistic {cu['SPHY']['nav_p1_t']:.1f}). This is read from the lag pattern; valuation times were not checked.",
        f"Over the window SPHY's NAV total return was {tdw['SPHY']['nav']:.2f}% against {tdw['SPHY']['h']:.2f}% for H0A0, a difference of {tdw['SPHY']['nav_minus_h']:+.2f} points. USHY's difference was {tdw['USHY']['nav_minus_h']:+.2f} and HYG's {tdw['HYG']['nav_minus_h']:+.2f}.",
    ])
    story += [img(CH["track"], CW * 0.93, 2.9 * inch),
              P(f"Cumulative NAV total return minus H0A0 total return, percentage points, from {w0:%d %b %Y}.", "cap"), PageBreak()]

    # ---------------- page 2 ----------------
    story += [P("Premium and discount to NAV", "h"),
              P("The premium is the closing price divided by NAV, minus one. SSGA and iShares publish NAV daily, so it can be computed for every trading day since launch."),
              tiles([(f"{wp['SPHY']['max']:+.2f}%", f"SPHY widest premium, {wp['SPHY']['max_date']:%d %b %Y}"),
                     (f"{wp['SPHY']['min']:+.2f}%", f"SPHY widest discount, {wp['SPHY']['min_date']:%d %b %Y}"),
                     (f"{wp['SPHY']['abs_gt_50bp']:.1f}%", f"Days SPHY was more than 50bp from NAV (HYG {wp['HYG']['abs_gt_50bp']:.1f}%, USHY {wp['USHY']['abs_gt_50bp']:.1f}%)"),
                     (f"{wp['SPHY']['ac5']:.2f}", "Five-day autocorrelation of SPHY's premium; the gap closes within days")]),
              Spacer(1, 6), img(CH["prem"], CW * 0.86, 5.2 * inch),
              P("Daily premium (+) or discount (-) to NAV, percent.", "cap")]
    story += bullets([
        f"All three funds reached their widest premium on 9 April 2025, the tariff-pause rally: SPHY {wp['SPHY']['max']:+.2f}%, HYG {wp['HYG']['max']:+.2f}%, USHY {wp['USHY']['max']:+.2f}%.",
        f"Over the full history SPHY ranged from {fp['SPHY']['min']:+.2f}% ({fp['SPHY']['min_date']:%d %b %Y}) to {fp['SPHY']['max']:+.2f}% ({fp['SPHY']['max_date']:%d %b %Y}). HYG reached {fp['HYG']['max']:+.2f}% in {fp['HYG']['max_date']:%b %Y}; that figure appears in both the NAV and price data and was not checked against a third source.",
        f"The premium varies more when index spreads are wide. SPHY's premium standard deviation is {it.loc['tight','prem_std']:.2f}% on the tightest third of OAS days and {it.loc['wide','prem_std']:.2f}% on the widest third.",
        f"Latest readings: SPHY {nowt['SPHY']['prem']:+.2f}%, HYG {nowt['HYG']['prem']:+.2f}%, USHY {nowt['USHY']['prem']:+.2f}%, each between {min((nowt[t]['prem'] - wp[t]['mean']) / wp[t]['std'] for t in an.ETFS):.1f} and {max((nowt[t]['prem'] - wp[t]['mean']) / wp[t]['std'] for t in an.ETFS):.1f} standard deviations above its window average.",
    ])
    story.append(PageBreak())

    # ---------------- page 3 ----------------
    e_lo, e_hi = rng(2012, 2017)
    l_lo, l_hi = rng(2020, 2026)
    story += [P("Price against NAV in different conditions", "h"),
              P("Two measures are used. The first is the correlation between a fund's daily price return and its daily NAV return. The second is the correlation between the daily change in premium and the NAV return, whose sign shows which moves further: positive means the price moves more than NAV on the day, negative means it moves less."),
              tiles([(f"{e_lo:.2f}-{e_hi:.2f}", "SPHY price-to-NAV correlation in each year, 2012 to 2017"),
                     (f"{l_lo:.2f}-{l_hi:.2f}", "Same measure in each year, 2020 to 2026"),
                     (f"{vt.loc['high','corr_px_nav']:.2f}", f"On the most volatile third of days (calmer thirds: {vt.loc['low','corr_px_nav']:.2f} and {vt.loc['mid','corr_px_nav']:.2f})"),
                     (f"{(sp.rc60 < 0).mean() * 100:.1f}%", f"Share of days the 60-day rolling correlation was below zero (low {sp.rc60.min():.2f})")]),
              Spacer(1, 6), img(CH["rolling"], CW * 0.88, 3.8 * inch),
              P("Rolling 60-day correlations. Upper panel: price return against NAV return. Lower panel: change in premium against NAV return.", "cap")]
    eps = EP[EP.etf == "SPHY"]
    rows = [[r.episode, f"{r.corr_px_nav:.2f}", f"{r.corr_dprem_nav:+.2f}", f"{r.prem_min:+.2f} to {r.prem_max:+.2f}", f"{r.gap_sd_bp:.0f}"] for r in eps.itertuples()]
    story += [grid(["SPHY episode", "Price vs NAV", "Change in premium vs NAV", "Premium range, %", "Daily SD of price minus NAV return, bp"], rows,
                   [2.4 * inch, 0.9 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch]), Spacer(1, 6)]
    story += bullets([
        f"The change-in-premium correlation was negative in every year from 2012 to 2018 (between {rng(2012,2018,'corr_dprem_nav')[0]:.2f} and {rng(2012,2018,'corr_dprem_nav')[1]:.2f}) and positive from 2020 ({rng(2020,2026,'corr_dprem_nav')[0]:.2f} to {rng(2020,2026,'corr_dprem_nav')[1]:.2f}). Closing prices used to move less than NAV on a typical day and now move more.",
        f"Within the index window, price-to-NAV correlation falls from {it.loc['tight','corr_p_n']:.2f} to {it.loc['wide','corr_p_n']:.2f} between the tightest and widest thirds of OAS, while the daily gap between price and H0A0 grows from {it.loc['tight','te_px_bp']:.0f}bp to {it.loc['wide','te_px_bp']:.0f}bp.",
    ])
    story.append(PageBreak())

    # ---------------- page 4 ----------------
    rows = [[t, f"{ds(t,L12)['mean_pct']:.3f}%", f"{ds(t,L12)['ann_mean']:.2f}%", f"{ds(t,L12)['sd_pct']:.3f}", f"{dl[t]['ratio_mean']:.2f}", f"{dl[t]['corr_ann_vs_yld']:.2f}",
             f"{vo[t]['vol_n']:.2f}%", f"{vo[t]['vol_p']:.2f}%"] for t in an.ETFS]
    story += [P("Distributions and volatility", "h"),
              P("Each fund pays twelve distributions a year. The yield on spot is each distribution divided by the last close before its ex-date, and the annualised figure multiplies it by twelve."),
              grid(["ETF", "Last 12: mean per distribution, % of spot", "Annualised", "Standard deviation (pp)", "Ratio to H0A0 effective yield", "Correlation with H0A0 yield", "NAV daily volatility", "Price daily volatility"],
                   rows, [0.55 * inch, 1.25 * inch, 0.8 * inch, 0.85 * inch, 0.95 * inch, 0.9 * inch, 0.8 * inch, 0.8 * inch]),
              Spacer(1, 4),
              P(f"H0A0 daily volatility over the window was {vo['SPHY']['vol_h']:.2f}% a year. Distribution yield and effective yield are defined differently (income less fees against total yield including pull to par), so the ratio is descriptive.", "cap")]
    story += bullets([
        f"SPHY's distribution has the smallest variation of the three. Over the last 12 its coefficient of variation is {ds('SPHY',L12)['cv']:.3f}, against {ds('HYG',L12)['cv']:.3f} for HYG and {ds('USHY',L12)['cv']:.3f} for USHY, and month-on-month changes in the amount have a standard deviation of {ds('SPHY',L12)['mom_chg_sd']:.1f}% (HYG {ds('HYG',L12)['mom_chg_sd']:.1f}%, USHY {ds('USHY',L12)['mom_chg_sd']:.1f}%).",
        f"SPHY's annualised distribution yield averages {dl['SPHY']['ratio_mean']:.2f} times the index's effective yield and has a {dl['SPHY']['corr_ann_vs_yld']:.2f} correlation with it. HYG's ratio is {dl['HYG']['ratio_mean']:.2f} with a {dl['HYG']['corr_ann_vs_yld']:.2f} correlation, and USHY's {dl['USHY']['ratio_mean']:.2f} with {dl['USHY']['corr_ann_vs_yld']:.2f}.",
        f"SPHY's annualised distribution yield on spot was {q(R['divyear'], etf='SPHY', year=2021)['ann_mean']:.1f}% in 2021 (average price {q(R['divyear'], etf='SPHY', year=2021)['spot_mean']:.2f}, {q(R['divyear'], etf='SPHY', year=2021)['div_ps']:.2f} per share) and {q(R['divyear'], etf='SPHY', year=2023)['ann_mean']:.1f}% in 2023 (average price {q(R['divyear'], etf='SPHY', year=2023)['spot_mean']:.2f}, {q(R['divyear'], etf='SPHY', year=2023)['div_ps']:.2f} per share).",
        f"Daily volatility is {vo['SPHY']['vol_n']:.2f}% for SPHY NAV and {vo['SPHY']['vol_p']:.2f}% for its closing price, against {vo['SPHY']['vol_h']:.2f}% for H0A0. At weekly frequency the price figure falls to {vo['SPHY']['wk_p']:.2f}% against {vo['SPHY']['wk_n']:.2f}% for NAV, because daily premium noise largely reverses within days.",
    ])
    story += [img(CH["vol"], CW * 0.8, 2.2 * inch), P("21-day realised volatility, annualised, H0A0 against SPHY NAV and SPHY price.", "cap"),
              P("Limits of the data", "h")]
    story += bullets([
        "FRED provides about three years of H0A0 history, which holds one major stress episode (April 2025). Regime findings for SPHY lean on its longer price-versus-NAV history, which does not need the index.",
        "Volatility groupings and the episode dates are the analyst's choices. Other cut-offs would move the exact numbers.",
        "Each fund's NAV comes from a single sponsor file and was not cross-checked. SPHY distributions come from Yahoo Finance, rounded to three decimals.",
        "The lead of prices over NAV and the explanation for rising correlation in stress are readings of the data and were not tested.",
    ])

    doc = BaseDocTemplate(out, pagesize=letter, leftMargin=M, rightMargin=M, topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                          title="SPHY, HYG and USHY against H0A0", author="Claude Code")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(M, 0.8 * inch, CW, H - 1.5 * inch, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page)])
    doc.build(story)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
