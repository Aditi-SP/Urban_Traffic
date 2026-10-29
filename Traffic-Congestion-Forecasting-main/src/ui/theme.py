"""
Visual theme and small reusable components for the dashboard (dark navy, readable, green / amber / red indicators).

The base colours come from .streamlit/config.toml ([theme]); this module adds the extra styling Streamlit does not offer
(cards, status chips, empty states) using plain HTML/CSS through st.markdown. No JavaScript and no external resources.
All dynamic text is HTML-escaped.

Semantic colours (used for STATUS and, in later stages, for real congestion values only):
    green = ok / free-flowing      amber = caution / slow / historical      red = problem / congested / unavailable
"""
import html

import streamlit as st

NAVY = "#0B1426"
PANEL = "#111C33"
PANEL_2 = "#16233F"
BORDER = "#22324F"
TEXT = "#E6EDF7"
MUTED = "#93A4C0"
BLUE = "#38BDF8"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED = "#EF4444"

KINDS = {"ok": GREEN, "warn": AMBER, "bad": RED, "info": BLUE, "mute": MUTED}

_CSS = f"""
<style>
/* layout */
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: #0E1830; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{ border-radius: 10px; padding: .45rem .7rem; }}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {PANEL_2}; border: 1px solid {BORDER}; }}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{ background: {PANEL}; }}

/* native components restyled as cards */
[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: .8rem 1rem; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .82rem; letter-spacing: .02em; }}
[data-testid="stMetricValue"] {{ font-size: 1.7rem; font-weight: 650; }}
[data-testid="stExpander"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; }}
[data-testid="stExpander"] summary {{ padding: .55rem .9rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ padding: .5rem .9rem; border-radius: 8px 8px 0 0; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {TEXT}; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}
button[kind="primary"] {{ border-radius: 10px; font-weight: 600; }}
button[kind="secondary"] {{ border-radius: 10px; }}

/* components */
.tc-hero {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
            padding: 1.1rem 1.3rem; margin-bottom: 1rem; border: 1px solid {BORDER}; border-radius: 16px;
            background: linear-gradient(135deg, {PANEL_2} 0%, {PANEL} 60%, {NAVY} 100%); }}
.tc-hero h1 {{ margin: 0; font-size: 1.75rem; font-weight: 700; padding: 0; }}
.tc-hero p {{ margin: .25rem 0 0 0; color: {MUTED}; font-size: .98rem; }}
.tc-hero.tc-compact {{ padding: .65rem 1.1rem; margin-bottom: .6rem; }}
.tc-hero.tc-compact h1 {{ font-size: 1.4rem; }}
.tc-hero.tc-compact p {{ margin-top: .1rem; font-size: .9rem; }}
div[data-testid="stPlotlyChart"] {{ border: 1px solid {BORDER}; border-radius: 14px; overflow: hidden; }}
iframe {{ border: 1px solid {BORDER}; border-radius: 14px; }}
@media (max-width: 1000px) {{
  div[data-testid="stHorizontalBlock"]:has(iframe) {{ flex-wrap: wrap; }}
  div[data-testid="stHorizontalBlock"]:has(iframe) > div[data-testid="stColumn"], div[data-testid="stHorizontalBlock"]:has(iframe) > div[data-testid="column"] {{ min-width: 100% !important; flex: 1 1 100% !important; }}
}}
.tc-chips {{ display: flex; gap: .4rem; flex-wrap: wrap; }}
.tc-chip {{ display: inline-flex; align-items: center; gap: .4rem; padding: .22rem .65rem; border-radius: 999px;
            font-size: .78rem; font-weight: 600; border: 1px solid; white-space: nowrap; }}
.tc-chip::before {{ content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; }}
.tc-ok   {{ color: {GREEN}; background: rgba(34,197,94,.10);  border-color: rgba(34,197,94,.35); }}
.tc-warn {{ color: {AMBER}; background: rgba(245,158,11,.10); border-color: rgba(245,158,11,.35); }}
.tc-bad  {{ color: {RED};   background: rgba(239,68,68,.10);  border-color: rgba(239,68,68,.35); }}
.tc-info {{ color: {BLUE};  background: rgba(56,189,248,.10); border-color: rgba(56,189,248,.35); }}
.tc-mute {{ color: {MUTED}; background: rgba(147,164,192,.08); border-color: rgba(147,164,192,.30); }}
.tc-card {{ background: {PANEL}; border: 1px solid {BORDER}; border-left-width: 4px; border-radius: 14px; padding: .8rem 1rem; min-height: 8.2rem; }}
.tc-card.tc-tight {{ min-height: 0; padding: .6rem .9rem; }}
.tc-card .lbl {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
.tc-card .val {{ font-size: 1.3rem; font-weight: 700; margin: .15rem 0; line-height: 1.2; }}
.tc-card .note {{ color: {MUTED}; font-size: .84rem; line-height: 1.35; }}
.tc-card.tc-ok {{ border-left-color: {GREEN}; }} .tc-card.tc-warn {{ border-left-color: {AMBER}; }}
.tc-card.tc-bad {{ border-left-color: {RED}; }}  .tc-card.tc-info {{ border-left-color: {BLUE}; }} .tc-card.tc-mute {{ border-left-color: {MUTED}; }}
.tc-card.tc-ok, .tc-card.tc-warn, .tc-card.tc-bad, .tc-card.tc-info, .tc-card.tc-mute {{ color: {TEXT}; background: {PANEL}; }}
.tc-empty {{ border: 1.5px dashed {BORDER}; border-radius: 16px; padding: 2rem 1.6rem; text-align: center;
             background: repeating-linear-gradient(135deg, {PANEL} 0 14px, #0F1A31 14px 28px); }}
.tc-empty h3 {{ margin: .2rem 0 .4rem 0; font-size: 1.25rem; }}
.tc-empty p {{ color: {MUTED}; max-width: 46rem; margin: .25rem auto; line-height: 1.5; }}
.tc-empty ul {{ text-align: left; display: inline-block; color: {MUTED}; margin: .6rem 0 0 0; line-height: 1.55; }}
.tc-section {{ margin: 1.2rem 0 .5rem 0; }}
.tc-section h2 {{ font-size: 1.15rem; margin: 0; padding: 0; font-weight: 650; }}
.tc-section p {{ margin: .15rem 0 0 0; color: {MUTED}; font-size: .9rem; }}
.tc-legend {{ display: flex; gap: 1.1rem; flex-wrap: wrap; align-items: center; color: {MUTED}; font-size: .85rem; }}
.tc-sw {{ display: inline-block; width: .9rem; height: .9rem; border-radius: 4px; margin-right: .35rem; vertical-align: -.12rem; }}
.tc-callout {{ border: 1px solid {BORDER}; border-left: 4px solid {AMBER}; background: {PANEL}; border-radius: 12px; padding: .75rem 1rem; margin: .3rem 0 1rem 0; }}
.tc-callout b {{ color: {TEXT}; }}
</style>
"""


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def inject_css():
    """Call once per run from the router (app.py)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def chip_html(text, kind="info") -> str:
    return f'<span class="tc-chip tc-{kind if kind in KINDS else "info"}">{_e(text)}</span>'


def page_header(title, subtitle="", chips=(), compact=False):
    """Large section header with status chips on the right: chips = [(text, kind)]."""
    chips_html = "".join(chip_html(t, k) for t, k in chips)
    st.markdown(f'<div class="tc-hero{" tc-compact" if compact else ""}"><div><h1>{_e(title)}</h1><p>{_e(subtitle)}</p></div>'
                f'<div class="tc-chips">{chips_html}</div></div>', unsafe_allow_html=True)


def section(title, subtitle=""):
    sub = f"<p>{_e(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="tc-section"><h2>{_e(title)}</h2>{sub}</div>', unsafe_allow_html=True)


def card_html(label, value, note="", kind="info", compact=False) -> str:
    kind = kind if kind in KINDS else "info"
    return (f'<div class="tc-card tc-{kind}{" tc-tight" if compact else ""}"><div class="lbl">{_e(label)}</div><div class="val">{_e(value)}</div>'
            f'<div class="note">{_e(note)}</div></div>')


def stat_cards(items):
    """A row of summary cards: items = [{"label", "value", "note", "kind"}]."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(card_html(it["label"], it["value"], it.get("note", ""), it.get("kind", "info")), unsafe_allow_html=True)


def empty_state(title, body, bullets=(), min_height=0):
    """A clearly-labelled empty panel (used instead of any invented content)."""
    items = "".join(f"<li>{_e(b)}</li>" for b in bullets)
    ul = f"<ul>{items}</ul>" if items else ""
    style = f' style="min-height:{int(min_height)}px;display:flex;flex-direction:column;justify-content:center;"' if min_height else ""
    st.markdown(f'<div class="tc-empty"{style}><h3>{_e(title)}</h3><p>{_e(body)}</p>{ul}</div>', unsafe_allow_html=True)


def callout(text_html_safe: str):
    """Amber note box. The argument must already be safe HTML (build it with _e / chip_html)."""
    st.markdown(f'<div class="tc-callout">{text_html_safe}</div>', unsafe_allow_html=True)


def legend_preview():
    """Colour key for the future live map. It is a design preview: no thresholds are claimed and no data is shown."""
    sw = lambda c, t: f'<span><span class="tc-sw" style="background:{c}"></span>{_e(t)}</span>'
    st.markdown('<div class="tc-legend">' + sw(GREEN, "Free-flowing") + sw(AMBER, "Slow") + sw(RED, "Congested")
                + f'<span style="opacity:.8">Legend preview only: thresholds will be defined from the provider\'s real data before any colouring is applied.</span></div>',
                unsafe_allow_html=True)


def legend_bands(thresholds_note: str):
    """Colour key for provider segments. The bands are this app's display convention, and the note says so."""
    sw = lambda c, t: f'<span><span class="tc-sw" style="background:{c}"></span>{_e(t)}</span>'
    st.markdown('<div class="tc-legend">' + sw(GREEN, "Free-flowing") + sw(AMBER, "Slow") + sw(RED, "Congested or closed")
                + sw(MUTED, "Stale (older reading)") + f'<span style="opacity:.85">{_e(thresholds_note)}</span></div>', unsafe_allow_html=True)
