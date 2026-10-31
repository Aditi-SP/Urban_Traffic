"""
The five sections of the dashboard. One definition used by the router (app.py) and by the tests.

Each page is a small script in src/ui/pages/ that calls the section's render() function, so a page can also be run
and tested on its own (AppTest.from_file on the page script).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PageDef:
    key: str
    title: str
    icon: str
    path: str       # relative to the main script (app.py)
    url_path: str
    default: bool = False
    full_title: str = ""   # the page header, when it is longer than the short navigation label

    @property
    def heading(self) -> str:
        return self.full_title or self.title


PAGES = [
    PageDef("live_map", "Live Traffic Map", ":material/map:", "src/ui/pages/live_map.py", "live-map", default=True),
    PageDef("forecasts", "Traffic Forecasts", ":material/trending_up:", "src/ui/pages/forecasts.py", "forecasts"),
    PageDef("community", "Community Reports", ":material/forum:", "src/ui/pages/community.py", "community"),
    PageDef("details", "Road History", ":material/traffic:", "src/ui/pages/details.py", "details"),
    PageDef("performance", "Model Insights", ":material/analytics:", "src/ui/pages/performance.py", "performance",
            full_title="Model Performance & Data Sources"),
]


def build_pages(st):
    """The st.Page objects for st.navigation (the first one is the landing page)."""
    return [st.Page(p.path, title=p.title, icon=p.icon, url_path=p.url_path, default=p.default) for p in PAGES]
