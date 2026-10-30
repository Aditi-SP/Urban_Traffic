"""
Section 3: Community Reports.

Short, anonymous, UNVERIFIED text reports from road users, stored on this computer (src/community/store.py). They are not traffic
measurements and are never used by the forecasting models.

Honesty rules
  * "Saved" is shown only after the store has committed AND read the report back from the database; every other outcome is an error
    with a reason the user can act on. Nothing is dropped silently.
  * The list always comes from the database. With no reports the page says so: no sample reports are ever shown.
  * User text is displayed as escaped plain text (a report cannot inject markup, links or images).
  * There is no edit or delete, no accounts and no photo upload on this page.
"""
import html
import time
import uuid

import streamlit as st

from src.community import store
from src.ui import theme

S = st.session_state
LIST_LIMIT = 50


def _session_id() -> str:
    return S.setdefault("cr_session", uuid.uuid4().hex)


def _on_submit():
    """Callback of the form button: runs before the page reruns, so it can also clear the fields after a confirmed save."""
    S["cr_msg"] = None
    token = S.setdefault("cr_token", uuid.uuid4().hex)  # kept until the report is saved, so a retry can never create two rows
    try:
        report, outcome = store.submit(S.get("cr_category"), S.get("cr_place", ""), S.get("cr_description", ""), session_id=_session_id(), token=token)
    except store.ReportError as e:
        S["cr_msg"] = ("error", e.message)
        return
    except Exception as e:  # noqa: BLE001  never crash the page; the text of the exception is not shown
        S["cr_msg"] = ("error", f"The report could not be saved because of an unexpected problem ({type(e).__name__}). Nothing was stored; please try again.")
        return
    when = time.strftime("%H:%M:%S", time.localtime(report.created_at))
    if outcome == "saved":
        S["cr_msg"] = ("success", f"Saved: your report (number {report.id}) was stored on this computer at {when} and is listed below. "
                                  "It is anonymous and unverified.")
    else:
        S["cr_msg"] = ("info", f"This report had already been saved (number {report.id}, {when}); nothing was added twice.")
    S["cr_place"], S["cr_description"], S["cr_token"] = "", "", uuid.uuid4().hex


def age_text(seconds: float, created_at: float) -> str:
    if seconds < 90:
        return f"{int(seconds)} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h ago"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(created_at))


def report_html(r: store.Report, now: float) -> str:
    """One report as a card. All user text is escaped and newlines become <br>, so it cannot form markup (or a blank line that ends the block)."""
    esc = lambda t: html.escape(t, quote=True).replace("\n", "<br>")
    old = r.outdated(now)
    return (f'<div class="tc-card tc-{"mute" if old else "info"} tc-tight"><div class="lbl">{esc(r.category)} | '
            f'{esc(age_text(r.age_s(now), r.created_at))} | unverified{" | possibly outdated" if old else ""}</div>'
            f'<div class="val" style="font-size:1.05rem">{esc(r.place)}</div><div class="note">{esc(r.description)}</div></div>')


def render():
    try:
        total, db_error = store.count(), None
    except store.ReportError as e:
        total, db_error = None, e.message
    theme.page_header("Community Reports", "Short anonymous reports from road users. They are unverified and are not traffic measurements.",
                      chips=[("Anonymous, unverified", "warn"), ("Stored on this computer", "mute")], compact=True)
    theme.stat_cards([
        {"label": "Reports saved", "value": "Unavailable" if total is None else str(total), "kind": "bad" if total is None else "info",
         "note": "Read from the local report database." if total is not None else "The report database could not be read."},
        {"label": "Shared with", "value": "This computer", "kind": "info", "note": "Everyone using this running dashboard sees the same reports."},
        {"label": "Identity", "value": "Anonymous", "kind": "info", "note": "No accounts and no personal details are stored."},
        {"label": "Photos", "value": "Not available", "kind": "mute", "note": "Only short text reports are supported."},
    ])
    if db_error:
        st.error(db_error)

    theme.section("Send a report")
    msg = S.get("cr_msg")
    if msg:
        getattr(st, msg[0])(msg[1])
    with st.form("cr_form", border=False):
        c1, c2 = st.columns([2, 3])
        c1.selectbox("What are you reporting?", store.CATEGORIES, key="cr_category")
        c2.text_input("Road or place", max_chars=store.MAX_PLACE, key="cr_place", placeholder="For example: Silk Board Junction, towards Hosur Road")
        st.text_area("What did you see?", max_chars=store.MAX_DESC, key="cr_description", height=100,
                     placeholder=f"At least {store.MIN_DESC} characters. Describe what you saw; please do not include names, phone numbers or number plates.")
        st.form_submit_button("Send report", type="primary", on_click=_on_submit, disabled=db_error is not None)
    st.caption(f"Limits: {store.SESSION_LIMIT[0]} reports per {store.SESSION_LIMIT[1] // 60} minutes per browser session; identical reports within "
               f"{store.DUPLICATE_WINDOW_S // 60} minutes are refused. A report cannot be edited or deleted from this page.")

    theme.section("Recent reports", "Newest first. Reports older than a day are marked as possibly outdated.")
    if db_error:
        st.info("Reports cannot be listed until the report database works again.")
        return
    try:
        reports = store.recent(LIST_LIMIT)
    except store.ReportError as e:
        st.error(e.message)
        return
    if not reports:
        theme.empty_state("No community reports have been saved yet",
                          "Nothing is shown here unless someone has saved a report on this computer. No sample reports exist.",
                          min_height=160)
        return
    now = time.time()
    st.markdown("<div style='display:flex;flex-direction:column;gap:.55rem'>" + "".join(report_html(r, now) for r in reports) + "</div>",
                unsafe_allow_html=True)
    if total and total > len(reports):
        st.caption(f"Showing the newest {len(reports)} of {total} reports.")
