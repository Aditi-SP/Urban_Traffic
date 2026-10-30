"""
Section 4: Road & Incident Details.

What this page implements: the HISTORICAL profile of each road in the 2022-2024 dataset (computed from the real file), always labelled as
historical. It reads no live data. Live speed, travel time and a closure flag for a SEARCHED place are on the Live Traffic Map page (TomTom
Flow Segment Data); they are not linked to the dataset's roads, which have no coordinates. Incidents are NOT implemented anywhere: no
incident source is connected (the provider's incidents API is not called) and community reports are separate and unverified.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dataProcessing import forecast_data as fd
from src.ui import theme
from src.ui.common import dataset_unavailable, get_data, plotly_template

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def render():
    theme.page_header(
        "Road & Incident Details",
        "The 2022-2024 historical profile of each road in the dataset; live conditions for a place are on the Live Traffic Map",
        chips=[("Live details: not on this page", "warn"), ("Incidents: not implemented", "warn"), ("History: 2022-2024 dataset", "info")])

    theme.callout("<b>What this page shows:</b> the historical profile of the roads in the dataset (real records, not live conditions). "
                  "<b>It reads no live data.</b> Live speed, travel time and closure for a place you search for are on the "
                  "<b>Live Traffic Map</b> page (they need a TomTom key). <b>Incidents are not implemented:</b> no incident source is connected, "
                  "and community reports are separate and unverified.")

    theme.section("Live status", "Not shown on this page; the incident feed is not implemented")
    theme.stat_cards([
        {"label": "Speed vs free-flow", "value": "Not shown here", "kind": "mute", "note": "Shown for a searched place on the Live Traffic Map; not linked to these roads."},
        {"label": "Travel time", "value": "Not shown here", "kind": "mute", "note": "Shown for a searched place on the Live Traffic Map; not linked to these roads."},
        {"label": "Road closure", "value": "Not shown here", "kind": "mute", "note": "Shown as a flag for a searched place on the Live Traffic Map; not linked to these roads."},
        {"label": "Incidents", "value": "Not implemented", "kind": "mute", "note": "No incident data source is connected. Community reports are unverified and listed on their own page."},
    ])

    df = get_data()
    theme.section("Historical profile (dataset: 2022-01-01 to 2024-08-09)", "Real records from the dataset; not live conditions")
    if df is None:
        dataset_unavailable()
        return
    roads = df.groupby(fd.ROAD_COL)["Area Name"].first().sort_index()
    road = st.selectbox("Road / intersection (historical dataset)", list(roads.index), key="details_road")
    d = df[df[fd.ROAD_COL] == road]
    span = (df[fd.DATE_COL].max() - df[fd.DATE_COL].min()).days + 1

    theme.stat_cards([
        {"label": "Area", "value": roads[road], "note": "From the dataset", "kind": "info"},
        {"label": "Days with a record", "value": f"{len(d):,}", "note": f"{len(d) / span * 100:.0f}% of the {span} days in the dataset", "kind": "info"},
        {"label": "Mean Congestion Level", "value": f"{d[fd.TARGET_COL].mean():.1f}", "note": "Scale 0-100; historical average", "kind": "info"},
        {"label": "Days at the cap of 100", "value": f"{(d[fd.TARGET_COL] >= 100).mean() * 100:.0f}%", "note": "Share of this road's records; historical", "kind": "info"},
    ])

    by_day = d.assign(weekday=d[fd.DATE_COL].dt.day_name()).groupby("weekday")[fd.TARGET_COL].mean().reindex(WEEKDAYS).reset_index()
    fig = px.bar(by_day, x="weekday", y=fd.TARGET_COL, template=plotly_template(), labels={fd.TARGET_COL: "Mean Congestion Level"},
                 title=f"{road}: mean historical congestion by weekday (2022-2024)")
    fig.update_layout(height=340, margin=dict(t=50, b=10), yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The dataset has no coordinates, so roads are not shown on a map. It is daily data, so there is no hour-of-day profile.")
