"""
Section 3: Community Reports.

Short, anonymous, UNVERIFIED reports from road users, stored on this computer: text reports (src/community/store.py) and PHOTO reports taken
at the reporter's current location (src/community/photos.py). They are not traffic measurements and are never used by the forecasting models.
A photo is shown as current traffic only while it is fresh (10 minutes after upload by default), then it is labelled expired.

Honesty rules
  * "Saved" is shown only after the store has committed AND read the report back from the database; every other outcome is an error
    with a reason the user can act on. Nothing is dropped silently.
  * The list always comes from the database. With no reports the page says so: no sample reports are ever shown.
  * User text is displayed as escaped plain text (a report cannot inject markup, links or images).
  * There is no edit or delete and there are no accounts. A photo is only accepted with the GPS position the reporter's device gave; nobody
    can type a position in, and the position is always labelled as reported by the device, not verified.
"""
import html
import time
import uuid

import streamlit as st

from src.community import photos, store
from src.ui import photo_capture, theme
from src.ui.photo_cards import photo_card

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


# ------------------------------------------------------------------------------------------------ photo reports
_last_purge = [0.0]


def _purge_old_files():
    """Delete the photo files older than the retention time, at most once a minute per process. Never raises: a housekeeping problem must not break the page."""
    if time.time() - _last_purge[0] < 60:
        return
    _last_purge[0] = time.time()
    try:
        photos.purge()
    except Exception:  # noqa: BLE001
        pass


def _open_photo():
    S["cr_photo_open"], S["cr_photo_msg"] = True, None
    S["cr_photo_token"] = uuid.uuid4().hex   # kept until the photo is saved, so a double click can never create two reports


def _close_photo():
    S["cr_photo_open"] = False
    S["cr_capture_n"] = S.get("cr_capture_n", 0) + 1   # a new key gives a fresh component: the old photo and position are forgotten
    S["cr_photo_desc"] = ""


def _submit_photo():
    """Callback of the Submit button. It runs BEFORE the page is redrawn, so it may reset the form. The payload is the one shown in the previous run."""
    info = photos.explain_capture(S.get("cr_photo_payload"))
    if not info["ready"]:
        S["cr_photo_msg"] = ("error", "Add a photo and share your location first.")
        return
    try:
        report, outcome = photos.submit_photo(info["image"], info["lat"], info["lon"], S.get("cr_photo_desc", ""), session_id=_session_id(),
                                              accuracy_m=info["accuracy_m"], token=S.get("cr_photo_token"))
    except photos.PhotoError as e:
        S["cr_photo_msg"] = ("error", e.message)
        return
    except Exception as e:  # noqa: BLE001  never crash the page; the text of the exception is not shown
        S["cr_photo_msg"] = ("error", f"The photo report could not be saved because of an unexpected problem ({type(e).__name__}). Nothing was stored; please try again.")
        return
    when = time.strftime("%H:%M:%S", time.gmtime(report.created_at))
    mins = photos.fresh_seconds() / 60
    if outcome == "saved":
        S["cr_photo_msg"] = ("success", f"Saved: your photo report was stored at {when} UTC. It is shown as current traffic to people looking at this area "
                                        f"for {mins:g} minutes, then it is marked as expired. It is anonymous and unverified.")
    else:
        S["cr_photo_msg"] = ("info", "This photo report had already been saved; nothing was added twice.")
    _close_photo()


def _photo_flow():
    msg = S.get("cr_photo_msg")
    if msg:
        getattr(st, msg[0])(msg[1])
    if not S.get("cr_photo_open"):
        st.button("Report Traffic with Photo", key="cr_photo_open_btn", type="primary", icon=":material/photo_camera:", on_click=_open_photo,
                  use_container_width=True)
        st.caption(f"Take a photo of the traffic where you are. People who look at this area on the Live Traffic Map see it for {photos.fresh_seconds() / 60:g} minutes, "
                   "then it is marked as expired. Your position comes from your device's GPS, and no name or account is saved.")
        return
    with st.container(border=True):
        st.markdown("#### Report Traffic with Photo")
        st.markdown("**Why location is required:** a photo is only useful to others if they know where it was taken. The report is accepted only with the GPS position "
                    "your device gives, which is shown as *reported by the device, not verified*. You cannot type a position in.")
        payload = photo_capture.capture(key=f"cr_capture_{S.get('cr_capture_n', 0)}")
        S["cr_photo_payload"] = payload       # remembered for the Submit callback of the next run
        info = photos.explain_capture(payload)
        for text in info["messages"]:
            st.warning(text)
        st.text_input("Short description (optional)", max_chars=photos.DESC_MAX, key="cr_photo_desc", placeholder="Heavy traffic near junction")
        if not info["ready"]:
            st.caption("The Submit button is enabled when both a photo and your location have been received.")
        c1, c2, _ = st.columns([2, 1, 3])
        c1.button("Submit photo report", key="cr_photo_submit", type="primary", disabled=not info["ready"], on_click=_submit_photo, use_container_width=True)
        c2.button("Cancel", key="cr_photo_cancel", on_click=_close_photo, use_container_width=True)


@st.fragment(run_every=photos.refresh_seconds())
def _recent_photos():
    """Fresh community photos (refreshed on a timer without reloading the page), then the history. The times are recomputed on every refresh."""
    now = time.time()
    try:
        every = photos.history(60)
    except (photos.PhotoError, store.ReportError) as e:
        st.error(getattr(e, "message", "The photo reports could not be read."))
        return
    fresh = [r for r in every if r.is_fresh(now)]
    theme.section("Recent community photos", f"Fresh for {photos.fresh_seconds() / 60:g} minutes after upload, then expired. Times are the server's UTC clock.")
    if not fresh:
        st.info("No recent community traffic reports.")
    for r in fresh[:12]:
        with st.container(border=True):
            photo_card(r, now=now)
    old = [r for r in every if not r.is_fresh(now)]
    with st.expander(f"Photo history ({len(old)} expired)"):
        if not old:
            st.caption("No expired photo reports yet.")
        for r in old[:20]:
            with st.container(border=True):
                photo_card(r, now=now)
        st.caption("Expired photos are history, never current traffic. Their image files are deleted after "
                   f"{photos.retention_seconds() / 3600:g} hours; the entry stays without the image.")


def render():
    _purge_old_files()
    try:
        total, db_error = store.count(), None
    except store.ReportError as e:
        total, db_error = None, e.message
    try:
        n_fresh = photos.count_fresh()
    except (photos.PhotoError, store.ReportError):
        n_fresh = None
    theme.page_header("Community Reports", "Anonymous reports and photos from road users. They are unverified and are not traffic measurements.",
                      chips=[("Anonymous, unverified", "warn"), ("Stored on this computer", "mute"), (f"Photos expire after {photos.fresh_seconds() / 60:g} min", "info")],
                      compact=True)
    theme.stat_cards([
        {"label": "Reports saved", "value": "Unavailable" if total is None else str(total), "kind": "bad" if total is None else "info",
         "note": "Read from the local report database." if total is not None else "The report database could not be read."},
        {"label": "Shared with", "value": "This computer", "kind": "info", "note": "Everyone using this running dashboard sees the same reports."},
        {"label": "Identity", "value": "Anonymous", "kind": "info", "note": "No accounts and no personal details are stored."},
        {"label": "Fresh photos", "value": "Unavailable" if n_fresh is None else str(n_fresh), "kind": "mute" if not n_fresh else "info",
         "note": f"Photo reports from the last {photos.fresh_seconds() / 60:g} minutes."},
    ])
    if db_error:
        st.error(db_error)

    theme.section("Report traffic with a photo", "Taken where you are, with your device's GPS position.")
    _photo_flow()
    _recent_photos()

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
