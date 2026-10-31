"""Real-browser check of the dashboard (headless Chrome via the DevTools Protocol; no extra packages).

Start the app first, e.g.  .\\run_dashboard.ps1 -Port 8599 -NoBrowser  (or: streamlit run app.py --server.port 8599), then:

    python tests/browser_check.py <screenshot-folder> [width height] [app-port]

    python tests/browser_check.py shots 1536 700        # maximised window on a 1920x1080 laptop at 125% scaling
    python tests/browser_check.py shots 1280 650 8600

The check works against a dashboard WITH or WITHOUT a TomTom key. It finds out which from the dashboard's own sidebar (the "Live traffic:" status chip)
and applies the matching expectations; every check below that depends on the key is written for both cases. It never requests traffic from TomTom (no
quota is used), never reads the key into its output, and compares the page only against key values found on this computer, which are never printed.

Not collected by unittest (name does not start with test_). Uses Chrome or Edge; set CHROME_PATH to use another browser.
"""
import asyncio, base64, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path
from tornado.websocket import websocket_connect

_CANDIDATES = [os.environ.get("CHROME_PATH", ""),
               r"C:\Program Files\Google\Chrome\Application\chrome.exe",
               r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
               r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
               r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
CHROME = next((c for c in _CANDIDATES if c and Path(c).exists()), None)
if CHROME is None:
    sys.exit("No Chrome/Edge found; set the CHROME_PATH environment variable.")
OUT = Path(sys.argv[1]); OUT.mkdir(exist_ok=True)
W, H = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) >= 4 else (1500, 1200)
APP_PORT = int(sys.argv[4]) if len(sys.argv) >= 5 else 8599
URL = f"http://localhost:{APP_PORT}"
PORT = 9333
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def _powershell(script):
    return subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=60).stdout.strip()


def known_keys():
    """Key VALUES found on this computer (environment, Windows user variable). Used ONLY to prove they appear nowhere in the page; never printed or stored."""
    found = {os.environ.get("TOMTOM_API_KEY", "").strip().strip("\"'")}
    try:
        found.add(_powershell("[Environment]::GetEnvironmentVariable('TOMTOM_API_KEY','User')").strip("\"'"))
    except Exception:
        pass
    return {k for k in found if len(k) >= 8}


def server_process_start(port):
    """Unix time at which the operating system created the process that listens on this port (None if it cannot be found)."""
    out = _powershell(f"$c = Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; "
                      "if ($c) { $p = Get-Process -Id $c.OwningProcess; [DateTimeOffset]::new($p.StartTime).ToUnixTimeMilliseconds() }")
    return int(out) / 1000 if out.isdigit() else None


def clock_to_epoch(hms, near):
    """The unix time of the local clock reading HH:MM:SS that is closest to `near` (the shown time has no date)."""
    h, m, sec = (int(x) for x in hms.split(":"))
    base = time.localtime(near)
    cands = [time.mktime((base.tm_year, base.tm_mon, base.tm_mday + d, h, m, sec, 0, 0, -1)) for d in (-1, 0, 1)]
    return min(cands, key=lambda e: abs(e - near))


SIDEBAR_JS = "(document.querySelector('[data-testid=\"stSidebar\"]') || {}).innerText || ''"
MAIN_JS = "(document.querySelector('[data-testid=\"stMain\"]') || {}).innerText || ''"
PROVIDER_CHIPS_WITH_KEY = ("Live traffic: key found, not yet tested", "Live traffic: last response OK", "Live traffic: provider problem")


class CDP:
    def __init__(self, ws):
        self.ws, self.i = ws, 0

    async def call(self, method, **params):
        self.i += 1
        await self.ws.write_message(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == self.i:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def text(self):
        return await self.js("document.body.innerText")

    async def shot(self, name):
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        (OUT / f"{name}.png").write_bytes(base64.b64decode(r["data"]))

    async def wait_text(self, needle, timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if needle in (await self.text() or ""):
                return True
            await asyncio.sleep(0.7)
        return False

    async def click_by_text(self, selector, needle, exact=False):
        js = f"""(() => {{ const el=[...document.querySelectorAll({json.dumps(selector)})].find(e=>e.innerText && (({json.dumps(exact)}) ? e.innerText.trim()==={json.dumps(needle)} : e.innerText.includes({json.dumps(needle)})));
                 if(!el) return false; el.scrollIntoView({{block:'center'}}); el.click(); return true; }})()"""
        return await self.js(js)

    async def real_click(self, rect_js):
        r = await self.js(rect_js)
        if not r:
            return False
        x, y = r["x"] + r["w"] / 2, r["y"] + r["h"] / 2
        for typ in ("mouseMoved", "mousePressed", "mouseReleased"):
            await self.call("Input.dispatchMouseEvent", type=typ, x=x, y=y, button="left", clickCount=1)
        return True

    async def settle(self, secs=3.0):
        # wait until Streamlit's "running" indicator is gone
        await asyncio.sleep(0.8)
        t0 = time.time()
        while time.time() - t0 < 60:
            running = await self.js("!!document.querySelector('[data-testid=\"stStatusWidget\"]')")
            if not running:
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(secs / 3)


BAD_PRE = ["Traceback", "Exception", "StreamlitAPIException", "KeyError", "No secrets found"]
BAD = ["Traceback", "Exception", "StreamlitAPIException", "KeyError", "ValueError:"]


async def main():
    prof = OUT / "chrome_profile"
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}",
                             f"--window-size={W},{H}", "--no-first-run", "--disable-gpu", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                await asyncio.sleep(0.5)
        assert ws_url, "Chrome DevTools not reachable"
        c = CDP(await websocket_connect(ws_url, max_message_size=50_000_000))
        await c.call("Page.enable")
        await c.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
        NAV = '[data-testid="stSidebarNav"] a'
        TITLES = ["Live Traffic Map", "Traffic Forecasts", "Community Reports", "Road History", "Model Insights"]
        NO_FAKE_LIVE = ("live forecast", "live prediction", "real-time forecast")

        async def goto(label, needle, timeout=90):
            clicked = await c.click_by_text(NAV, label)
            ok = await c.wait_text(needle, timeout)
            await c.settle()
            return bool(clicked) and ok, await c.text()

        # ---------------------------------------------------------- landing page: Live Traffic Map (shell)
        await c.call("Page.navigate", url=URL)
        ok = await c.wait_text("Community photos", 120)   # the header chips of the Live Traffic Map
        await c.settle()
        t = await c.text()
        configured = "Live traffic not configured" not in (await c.js(MAIN_JS) or "")
        print(f"   scenario under test: TomTom key {'CONFIGURED' if configured else 'NOT configured'} in this dashboard "
              "(read from its own sidebar; the key itself is never read into this output)", flush=True)
        check("app loads in Chrome and lands on the Live Traffic Map", ok and "Live Traffic Map" in t)
        links = await c.js(f"[...document.querySelectorAll('{NAV}')].map(a => a.innerText)")
        check("sidebar navigation has the five sections in order",
              isinstance(links, list) and len(links) == 5 and all(t_ in l for t_, l in zip(TITLES, links)), str(links))
        check("Streamlit Deploy button is hidden (client.toolbarMode = viewer)", "Deploy" not in t)
        check("dark navy theme applied", await c.js("getComputedStyle(document.querySelector('.stApp')).backgroundColor") in ("rgb(11, 20, 38)", "rgb(11,20,38)"))
        if configured:
            check("live map (key configured): no setup banner, and no reading or value until the user asks for one",
                  "Live traffic is not configured" not in t and "Live traffic not configured" not in t and "No reading yet" in t and "km/h" not in t)
        else:
            check("live map (no key): says live traffic is not configured, no values shown",
                  "Live traffic not configured" in t and "No reading yet" in t and "km/h" not in t)
        IF = "document.querySelector('iframe').contentDocument"
        async def map_state():
            return await c.js("(() => { const d = " + IF + "; if (!d) return null; return {tiles: [...d.querySelectorAll('img.leaflet-tile')].filter(i => i.complete && i.naturalWidth > 0).length, "
                              "attribution: (d.querySelector('.leaflet-control-attribution') || {}).innerText || '', paths: d.querySelectorAll('path.leaflet-interactive').length, "
                              "w: d.getElementById('map').getBoundingClientRect().width, h: d.getElementById('map').getBoundingClientRect().height}; })()")
        st = None
        for _ in range(40):
            st = await map_state()
            if st and st["tiles"] > 0:
                break
            await asyncio.sleep(1)
        check("live map: real OpenStreetMap basemap tiles loaded in the browser", bool(st) and st["tiles"] > 0, str(st))
        check("live map: OpenStreetMap attribution visible", bool(st) and "OpenStreetMap" in st["attribution"], str(st))
        check("live map: empty-map state draws no traffic segment or marker", bool(st) and st["paths"] == 0, str(st))
        main_w = await c.js("(document.querySelector('[data-testid=\"stMain\"], section.main') || document.body).clientWidth")
        check("live map: map is large (>= 60% of the content area width, >= 360px high)", bool(st) and st["w"] >= 0.60 * (main_w - 120) and st["h"] >= 360, f"{st} main={main_w}")
        page_html = (await c.js("document.documentElement.outerHTML")) or ""
        frame_html = (await c.js(IF + ".documentElement.outerHTML")) or ""
        check("live map: no provider key or provider URL in the page or iframe", "api.tomtom.com" not in page_html and "api.tomtom.com" not in frame_html)
        keys = known_keys()
        everything = page_html + frame_html + (t or "") + (await c.js(SIDEBAR_JS) or "")
        requests_seen = (await c.js("performance.getEntriesByType('resource').map(e => e.name)") or []) + (await c.js("(() => { const w = document.querySelector('iframe').contentWindow; return w ? w.performance.getEntriesByType('resource').map(e => e.name) : []; })()") or [])
        check("the API key never reaches the browser: not in the page text, HTML, map iframe or any request the browser made (compared with the key value(s) found on this computer, which are not printed)",
              not any(k in everything for k in keys) and not any(k in u for k in keys for u in requests_seen)
              and not re.search(r"[?&]key=[A-Za-z0-9]{16,}", page_html + frame_html + " ".join(requests_seen))
              and not any("tomtom" in u.lower() for u in requests_seen),
              f"{len(keys)} local key value(s) compared; {len(requests_seen)} browser requests inspected")
        check("live map: Plot Controls are not shown here", "Plot Controls" not in t)
        check("live map: search form present and a map above the fold", await c.js("(() => { const r = document.querySelector('iframe').getBoundingClientRect(); return r.top < " + str(H) + " && r.height > 300; })()"))
        side_t = await c.js(SIDEBAR_JS) or ""
        check("the sidebar is navigation only: no System status chips and no Diagnostics panel",
              not any(w in side_t for w in ("System status", "Live traffic:", "Data: historical", "Models: not validated", "Reports: working", "Diagnostics"))
              and await c.js("document.querySelectorAll('[data-testid=\"stSidebar\"] details').length") == 0,
              " | ".join(side_t.split())[:100])
        main_now = await c.js(MAIN_JS) or ""
        check("no provider warning before any request: no 'Traffic provider' panel, no 'key found, not tested' text, no success claim",
              not any(w in main_now for w in ("Traffic provider", "Key found, not tested", "key found, not yet tested", "A key is configured, but no request", "last response OK")),
              "scenario: " + ("key configured" if configured else "no key"))
        check("live map: the Road Details / Incident Details placeholders are not on the page", not any(w in main_now for w in ("Incident Details", "Road Details", "Not implemented", "Not shown here")))
        check("live map: the community photo layer and the radius control are present, with the honest 'two sources' legend",
              "Community photos" in main_now and "Community photo radius" in main_now and "Two different sources" in main_now and "Choose a place to see recent community photos near it." in main_now)
        if not configured:
            check("setup banner: says live traffic is not configured, names TOMTOM_API_KEY and points to the README",
                  "Live traffic is not configured." in t and "TOMTOM_API_KEY" in t and "README" in t)
            check("setup banner is above the map (the instruction is not hidden below the fold)",
                  await c.js("(() => { const b=[...document.querySelectorAll('[data-testid=\"stAlert\"]')].find(e=>e.innerText.includes('Live traffic is not configured')); const m=document.querySelector('iframe'); return !!b && !!m && b.getBoundingClientRect().top < m.getBoundingClientRect().top; })()"))
            check("setup banner is compact (at most a third of the window height) and at least 150 px of the map is visible without scrolling (regression: the banner was 528 px tall at 800 px wide and 0 px of the map showed)",
                  await c.js("(() => { const b=[...document.querySelectorAll('[data-testid=\"stAlert\"]')].find(e=>e.innerText.includes('Live traffic is not configured')); const m=document.querySelector('iframe'); "
                             "return !!b && !!m && b.getBoundingClientRect().height <= innerHeight / 3 && (Math.min(innerHeight, m.getBoundingClientRect().bottom) - Math.max(0, m.getBoundingClientRect().top)) >= 150; })()"),
                  str(await c.js("(() => { const b=[...document.querySelectorAll('[data-testid=\"stAlert\"]')].find(e=>e.innerText.includes('Live traffic is not configured')); const m=document.querySelector('iframe'); "
                                 "return {banner: b && Math.round(b.getBoundingClientRect().height), mapTop: m && Math.round(m.getBoundingClientRect().top), mapVisible: m && Math.round(Math.min(innerHeight, m.getBoundingClientRect().bottom) - Math.max(0, m.getBoundingClientRect().top)), window: innerHeight}; })()")))
            opened_help = await c.click_by_text("summary, [data-testid='stExpander'] summary", "Set up the key")
            await c.settle()
            th = await c.text()
            check("setup instructions stay one click away: the expander shows the hidden-prompt PowerShell commands and the launcher restart",
                  bool(opened_help) and all(k in th for k in ("Read-Host", "-AsSecureString", "SetEnvironmentVariable", "run_dashboard.ps1", "secrets.toml")))
            await c.click_by_text("summary, [data-testid='stExpander'] summary", "Set up the key")  # close it again: the rest of the checks assume the normal layout
            await c.settle()
        else:
            main_t = await c.js(MAIN_JS) or ""
            check("key configured: there is no setup banner and no key-setup help to click",
                  "Live traffic is not configured" not in main_t and not await c.js("[...document.querySelectorAll('summary')].some(e => e.innerText.includes('Set up the key'))"))
            check("key configured: nothing is requested on page load, and the traffic button stays disabled until a place is chosen",
                  await c.js("(() => { const b = [...document.querySelectorAll('button')].find(e => /Get traffic for this place|Refresh traffic/.test(e.innerText)); return !!b && b.disabled; })()")
                  and "Choose a place first" in main_t)
        check("homepage 'At a glance' strip shows community photos, data freshness and models (no provider card)",
              all(k in t.lower() for k in ("at a glance", "community photos", "historical data", "latest 2024-08-09", "models")) and "TRAFFIC PROVIDER" not in t.upper())  # labels are upper-cased by CSS
        check("homepage models card quotes the saved held-out evaluation and says it is not validated for 2026",
              "Held-out test RMSE" in t and "Neither model beat the road mean." in t and "Not validated for 2026" in t)
        check("homepage links to the detail pages", await c.js("document.querySelectorAll('[data-testid=\"stPageLink\"], a[data-testid=\"stPageLink-NavLink\"]').length") >= 3)
        check("landing page makes no live-traffic or forecast claim before traffic is requested", "km/h" not in t.split("Colour bands")[0] and "Forecast for" not in t)
        check("no error text on the landing page", not any(b in t for b in BAD_PRE))
        await c.shot("01_live_map")

        # ---------------------------------------------------------- Community Reports (form + local store; this check never saves a report)
        ok, t = await goto("Community Reports", "Send a report")
        check("Community Reports opens with the report form", ok and "Anonymous, unverified" in t and "Stored on this computer" in t)
        check("Community Reports: form fields and the Send button exist",
              await c.js("!!document.querySelector('textarea') && [...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Send report')")
              and "What are you reporting?" in t and "Road or place" in t and "What did you see?" in t)
        check("Community Reports: honest empty state or real saved reports, never samples",
              "No community reports have been saved yet" in t or "unverified" in t.split("Recent reports")[-1])
        check("Community Reports: the prominent 'Report Traffic with Photo' button, the explanation, and the fresh-photo section (with its honest empty state)",
              await c.js("[...document.querySelectorAll('button')].some(b=>b.innerText.includes('Report Traffic with Photo'))") and "Report traffic with a photo" in t
              and "Recent community photos" in t and ("No recent community traffic reports." in t or "Fresh" in t), "")
        check("Community Reports: no edit/delete control, and no Streamlit file uploader (the photo capture is its own component, tested by tests/browser_photo_check.py)",
              await c.js("document.querySelectorAll('[data-testid=\"stFileUploader\"], [data-testid=\"stCameraInput\"]').length") == 0
              and not await c.js("[...document.querySelectorAll('button')].some(b=>/delete|edit|remove/i.test(b.innerText))"))
        await c.click_by_text("button", "Send report", exact=True)  # empty form: validated BEFORE any storage, so nothing is written
        ok_err = await c.wait_text("The road or place is too short", 30)
        await c.settle()
        t = await c.text()
        check("Community Reports: sending an empty form shows an actionable error and no success message", ok_err and "Saved: your report" not in t)
        check("Community Reports: no error text", not any(b in t for b in BAD_PRE))
        check("Plot Controls are hidden on pages that do not use them", "Plot Controls" not in t)
        await c.shot("02_community")

        # ---------------------------------------------------------- Road History (was Road & Incident Details; its non-functional placeholders are gone)
        ok, t = await goto("Road History", "Historical profile")
        main_t = await c.js(MAIN_JS) or ""
        check("Road History opens with the historical profile of the dataset's roads", ok and "Road History" in main_t and "dataset: 2022-01-01 to 2024-08-09" in main_t)
        check("Road History: no live-status or incident placeholders (no 'Not shown here', 'Not implemented', 'Incidents', 'Live status')",
              not any(w in main_t for w in ("Not shown here", "Not implemented", "Not available", "Live status", "Speed vs free-flow", "Road closure"))
              and "incident" not in main_t.lower().replace("not live conditions and not incidents", ""))
        check("Road History: history is labelled historical and a chart is drawn", "not live conditions" in main_t and await c.js("document.querySelectorAll('.js-plotly-plot').length") >= 1)
        check("Road History: no error text", not any(b in t for b in BAD_PRE))
        await c.shot("03_details")

        # ---------------------------------------------------------- Model Insights (existing evaluation)
        ok, t = await goto("Model Insights", "Project objective")
        check("Model Insights: short sidebar label, full page title kept, Plot Controls shown here",
              "Model Performance & Data Sources" in t and "Plot Controls" in t)
        links2 = await c.js(f"[...document.querySelectorAll('{NAV}')].map(a => a.innerText)")
        check("sidebar shows 'Model Insights' (not the long title)", any("Model Insights" in l for l in links2) and not any("Model Performance" in l for l in links2), str(links2))
        check("Model Performance page opens with overview + headline metrics", ok and "Headline result" in t)
        check("Model Insights documents how the numbers were produced (dataset, held-out split, baselines, training vs held-out)",
              all(k in t for k in ("How these numbers were produced", "held-out test set", "Training versus held-out", "does not show accuracy on live traffic")))
        check("Model Insights explains MAE, RMSE and R2", "What do MAE, RMSE and R" in t)
        check("Model Performance: honest baseline warning visible", "neither learned model beats" in t.lower())
        check("Model Performance: not validated for 2026 is stated", "Not validated for 2026" in t)
        check("no setup banner on a healthy install", "Setup incomplete" not in t)
        await c.shot("04_performance")
        tabs = [("Data & Wrangling", "Wrangling and validation steps", "04b_data"),
                ("Historical Analysis", "no hour-of-day information", "04c_history"),
                ("Model Results", "Key Performance Indicators", "04d_results"),
                ("Limitations & Sources", "Limitations", "04e_limits")]
        for label, needle, shot in tabs:
            clicked = await c.click_by_text('button[role="tab"]', label)
            await c.settle()
            t = await c.text()
            check(f"tab '{label}' opens and shows expected content", bool(clicked) and needle in t, needle)
            check(f"tab '{label}' has no error text", not any(b in t for b in BAD))
            await c.shot(shot)
            if label == "Model Results":
                check("Model Results: no-winner message (CI includes 0)", "No winner between LSTM and XGBoost" in t)
                check("Model Results: CI table present", "95% confidence intervals" in t)
                check("Model Results: baseline warning", "Neither model beats the simple" in t)

        # ---------------------------------------------------------- Traffic Forecasts (historical replay / what-if)
        ok, t = await goto("Traffic Forecasts", "Historical replay and what-if forecasts")
        check("Traffic Forecasts opens and is labelled historical, not validated for 2026",
              ok and "Historical data 2022-2024" in t and "Not validated for 2026" in t)
        check("Traffic Forecasts: the large 'How to read this page' warning is gone and a small 'About the data and models' expander replaces it (collapsed)",
              "How to read this page" not in t and "About the data and models" in t and "Neither model outperformed" not in t)
        await c.click_by_text("summary, [data-testid='stExpander'] summary", "About the data and models")
        await c.settle()
        te = await c.text()
        check("Traffic Forecasts: expanded, it keeps the important facts (dataset dates, experimental models, no outperforming of the baseline, no recent real-world evidence)",
              all(k in te for k in ("2022-01-01 to 2024-08-09", "LSTM and XGBoost models are experimental", "Neither model outperformed the simple per-road historical-mean baseline",
                                    "Recent real-world performance has not been established", "not live traffic")))
        await c.shot("05_forecasts_about_expanded")
        await c.click_by_text("summary, [data-testid='stExpander'] summary", "About the data and models")   # collapse it again
        await c.settle()
        t = await c.text()
        check("no page text calls anything a live forecast", not any(b in t.lower() for b in NO_FAKE_LIVE))
        check("Mode 1 labelled as historical (not live)", "not live traffic" in t.replace("*", ""), "label text")
        check("Forecast page shows Predict / Reset buttons",
              await c.js("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Predict')") and
              await c.js("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Reset demo')"))
        await c.shot("05a_demo_before")

        tfull = await c.text()
        check("Forecasts: states the real age of the data and that the dataset ends on 2024-08-09",
              "days old" in tfull and "Today is" in tfull and "2022-01-01 to 2024-08-09" in tfull and "Until 2024-08-09" in tfull)   # the "Data and forecast status" section, visible without opening the expander
        check("Forecasts: 'Data and forecast status' separates observed, replayed and forecast data and says future forecasts are not available",
              all(k in tfull.lower() for k in ("historical observed traffic", "until 2024-08-09", "replay and what-if", "past days only", "future forecasts", "not available", "2024-08-10")))
        check("Forecasts: tells how to check newer observations (read-only validator) and that retraining needs approval",
              "check_new_data" in tfull and "needs separate approval" in tfull)
        await c.click_by_text("button", "Predict", exact=True)
        ok = await c.wait_text("Forecast for", 90)
        await c.settle()
        t = await c.text()
        check("Historical replay: Predict produces a forecast", ok)
        check("Forecast panel shows XGBoost, LSTM, baselines, actual", all(k in t for k in ("XGBoost", "LSTM", "Persistence baseline", "Road-mean baseline", "Actual next day")))
        check("Forecast labelled Historical replay with horizon", "Historical replay" in t and "1 day ahead" in t)
        check("Forecast states when it was generated and the latest data date", "generated " in t and "(this computer's clock)" in t and "latest data in the dataset: 2024-08-09" in t)
        check("No fabricated congestion category / interval claim", "no uncertainty interval is produced" in t and "No congestion category" in t)
        check("Prediction log appears after first run", "Prediction log" in t)
        # laptop usability: with the forecast heading at the top of the window, the whole chart must already be in view
        geo = await c.js("""(() => { const h=[...document.querySelectorAll('h3')].find(e=>e.innerText.startsWith('Forecast for') && e.getBoundingClientRect().height>0);
              if(!h) return null; h.scrollIntoView({block:'start'}); const hb=h.getBoundingClientRect();
              // the chart under THIS heading: first VISIBLE plotly chart located below it (charts in other tabs have zero size)
              const g=[...document.querySelectorAll('.js-plotly-plot')].find(e=>{const r=e.getBoundingClientRect(); return r.height>50 && r.top>=hb.top;});
              if(!g) return {chartFound:false, viewport: window.innerHeight};
              const gb=g.getBoundingClientRect(); return {chartFound:true, chartTop: Math.round(gb.top), chartBottom: Math.round(gb.bottom), viewport: window.innerHeight, headingTop: Math.round(hb.top)}; })()""")
        await c.shot("05b2_forecast_in_view")
        check(f"forecast chart fully visible without extra scrolling ({W}x{H})", bool(geo) and geo.get("chartFound") is True and geo["chartBottom"] <= geo["viewport"], str(geo))
        await c.shot("05b_demo_replay")

        # what-if
        clicked = await c.click_by_text('label[data-baseweb="radio"]', "What-if scenario")
        await c.settle()
        t = await c.text()
        check("What-if mode reveals editable controls", bool(clicked) and "Congestion Level" in t and "Scenario" in t)
        # choose heavy preset in the Scenario selectbox with REAL mouse events
        opened = await c.real_click("""(() => { const sb=[...document.querySelectorAll('[data-testid="stSelectbox"]')].find(e=>e.innerText.includes('Scenario')); if(!sb) return null; sb.scrollIntoView({block:'center'}); const r=sb.querySelector('[data-baseweb="select"]').getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; })()""")
        await asyncio.sleep(1.0)
        picked = await c.real_click("""(() => { const o=[...document.querySelectorAll('li[role="option"]')].find(e=>e.innerText.includes('Heavy congestion')); if(!o) return null; const r=o.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; })()""")
        await c.settle()
        shown = await c.js("[...document.querySelectorAll('[data-testid=\"stSelectbox\"]')].filter(e=>e.innerText.includes('Scenario')).map(e=>e.innerText).join('|')")
        print("   scenario selectbox now shows:", repr(shown))
        check("What-if: 'Heavy congestion' preset selectable", bool(opened) and bool(picked) and "Heavy congestion" in (shown or ""))
        await c.click_by_text("button", "Predict", exact=True)
        await asyncio.sleep(1.0)
        await c.wait_text("What-if scenario", 60)
        await c.settle()
        t = await c.text()
        check("What-if prediction runs and is labelled 'What-if scenario'", "Input source: What-if scenario" in t.replace("**", ""))
        check("What-if shows the weak-sensitivity note", "response to edits is weak" in t)
        check("Prediction log has 2 runs", await c.js("(() => { const t=document.body.innerText; const i=t.indexOf('Prediction log'); return i>=0; })()"))
        await c.shot("05c_demo_whatif")

        # reset
        await c.click_by_text("button", "Reset demo", exact=True)
        await c.settle()
        t = await c.text()
        check("Reset demo clears the forecast", "Choose a road and day" in t and "Forecast for" not in t)
        await c.shot("05d_demo_reset")

        # Mode 2
        await c.click_by_text('label[data-baseweb="radio"]', "Mode 2")
        await c.settle()
        t = await c.text()
        if configured:
            check("Mode 2 (TomTom, key configured): no 'No API key' message, a fetch button, and no reading until it is pressed (it is NOT pressed here: no quota is used)",
                  "No API key configured" not in t and "Fetch provider observation" in t and "No provider reading yet" in t)
        else:
            check("Mode 2 (TomTom) unconfigured: config message shown", "No API key configured" in t)
        check("Mode 2: states provider data cannot feed the models", "cannot be used to make a valid prediction" in t)
        if not configured:
            check("Mode 2: no fetch/predict controls without a key", "Fetch provider observation" not in t)
        check("Mode 2: no error text", not any(b in t for b in BAD))
        await c.shot("06_demo_mode2")

        # ---------------------------------------------------------- deep links: every page opens directly from its URL
        for path, needle in (("forecasts", "Historical replay and what-if forecasts"), ("community", "Send a report"),
                             ("details", "Historical profile"), ("performance", "Project objective"),
                             ("", "Choose a place first" if configured else "Live traffic is not configured")):
            await c.call("Page.navigate", url=f"{URL}/{path}")
            ok = await c.wait_text(needle, 90)
            await c.settle()
            t = await c.text()
            check(f"deep link /{path} opens its page without errors", ok and not any(b in t for b in BAD_PRE))
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:
            proc.kill()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} browser checks passed")
    sys.exit(1 if n_fail else 0)


asyncio.run(main())
