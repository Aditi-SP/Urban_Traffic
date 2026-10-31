"""
Section 4: Road History (it was "Road & Incident Details" while it still showed placeholders for live status and incidents; those are removed).

What this page implements: the HISTORICAL profile of each road in the 2022-2024 dataset (computed from the real file), always labelled as
historical. It reads no live data. Live speed and travel time for a SEARCHED place are on the Live Traffic Map (TomTom Flow Segment Data);
they are not linked to the dataset's roads, which have no coordinates. Incidents are not implemented anywhere, so none are shown.
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
        "Road History",
        "The 2022-2024 profile of each road in the dataset: real historical records, not live conditions and not incidents",
        chips=[("History: 2022-2024 dataset", "info")])

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
