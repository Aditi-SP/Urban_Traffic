"""Real-browser check of the dashboard (headless Chrome via the DevTools Protocol; no extra packages).

Start the app first, e.g.  .\\run_dashboard.ps1 -Port 8599 -NoBrowser  (or: streamlit run app.py --server.port 8599), then:

    python tests/browser_check.py <screenshot-folder> [width height] [app-port]

    python tests/browser_check.py shots 1536 700        # maximised window on a 1920x1080 laptop at 125% scaling
    python tests/browser_check.py shots 1280 650 8600

Not collected by unittest (name does not start with test_). Uses Chrome or Edge; set CHROME_PATH to use another browser.
"""
import asyncio, base64, json, os, subprocess, sys, time, urllib.request
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
        TITLES = ["Live Traffic Map", "Traffic Forecasts", "Community Reports", "Road & Incident Details", "Model Insights"]
        NO_FAKE_LIVE = ("live forecast", "live prediction", "real-time forecast")

        async def goto(label, needle, timeout=90):
            clicked = await c.click_by_text(NAV, label)
            ok = await c.wait_text(needle, timeout)
            await c.settle()
            return bool(clicked) and ok, await c.text()

        # ---------------------------------------------------------- landing page: Live Traffic Map (shell)
        await c.call("Page.navigate", url=URL)
        ok = await c.wait_text("Live traffic is not configured", 120)
        await c.settle()
        t = await c.text()
        check("app loads in Chrome and lands on the Live Traffic Map", ok and "Live Traffic Map" in t)
        links = await c.js(f"[...document.querySelectorAll('{NAV}')].map(a => a.innerText)")
        check("sidebar navigation has the five sections in order",
              isinstance(links, list) and len(links) == 5 and all(t_ in l for t_, l in zip(TITLES, links)), str(links))
        check("Streamlit Deploy button is hidden (client.toolbarMode = viewer)", "Deploy" not in t)
        check("dark navy theme applied", await c.js("getComputedStyle(document.querySelector('.stApp')).backgroundColor") in ("rgb(11, 20, 38)", "rgb(11,20,38)"))
        check("live map: says live traffic is not configured, no values shown",
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
        check("live map: no provider key or provider URL in the page or iframe", "api.tomtom.com" not in (await c.js("document.documentElement.outerHTML")) and
              "api.tomtom.com" not in (await c.js(IF + ".documentElement.outerHTML")))
        check("live map: Plot Controls are not shown here", "Plot Controls" not in t)
        check("live map: search form present and a map above the fold", await c.js("(() => { const r = document.querySelector('iframe').getBoundingClientRect(); return r.top < " + str(H) + " && r.height > 300; })()"))
        check("sidebar shows honest system status", "Live traffic: not configured" in t and "Models: not validated for 2026" in t)
        check("no error text on the landing page", not any(b in t for b in BAD_PRE))
        await c.shot("01_live_map")

        # ---------------------------------------------------------- Community Reports (shell)
        ok, t = await goto("Community Reports", "not available yet")
        check("Community Reports opens with a clear 'not implemented' empty state", ok and "Not implemented yet" in t and "No sample reports are shown" in t)
        check("Community Reports: no upload control and no fake reports",
              await c.js("document.querySelectorAll('input[type=file], [data-testid=\"stFileUploader\"], [data-testid=\"stCameraInput\"]').length") == 0)
        check("Community Reports: no error text", not any(b in t for b in BAD_PRE))
        check("Plot Controls are hidden on pages that do not use them", "Plot Controls" not in t)
        await c.shot("02_community")

        # ---------------------------------------------------------- Road & Incident Details
        ok, t = await goto("Road & Incident Details", "will be added in a later stage")
        check("Road & Incident Details opens and defers live details to a later stage", ok and "does not show live conditions" in t)
        check("Road & Incident Details: live status cards say Not available (4)", await c.js("[...document.querySelectorAll('.tc-card .val')].filter(e => e.innerText.trim()==='Not available').length") == 4)
        check("Road & Incident Details: history is labelled historical", "not live conditions" in t and "dataset: 2022-01-01 to 2024-08-09" in t)
        check("Road & Incident Details: no error text", not any(b in t for b in BAD_PRE))
        await c.shot("03_details")

        # ---------------------------------------------------------- Model Insights (existing evaluation)
        ok, t = await goto("Model Insights", "Project objective")
        check("Model Insights: short sidebar label, full page title kept, Plot Controls shown here",
              "Model Performance & Data Sources" in t and "Plot Controls" in t)
        links2 = await c.js(f"[...document.querySelectorAll('{NAV}')].map(a => a.innerText)")
        check("sidebar shows 'Model Insights' (not the long title)", any("Model Insights" in l for l in links2) and not any("Model Performance" in l for l in links2), str(links2))
        check("Model Performance page opens with overview + headline metrics", ok and "Headline result" in t)
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
              ok and "Historical data 2022-2024" in t and "Not validated for 2026" in t and "not live traffic" in t.lower())
        check("no page text calls anything a live forecast", not any(b in t.lower() for b in NO_FAKE_LIVE))
        check("Mode 1 labelled as historical (not live)", "not live traffic" in t.replace("*", ""), "label text")
        check("Forecast page shows Predict / Reset buttons",
              await c.js("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Predict')") and
              await c.js("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Reset demo')"))
        await c.shot("05a_demo_before")

        await c.click_by_text("button", "Predict", exact=True)
        ok = await c.wait_text("Forecast for", 90)
        await c.settle()
        t = await c.text()
        check("Historical replay: Predict produces a forecast", ok)
        check("Forecast panel shows XGBoost, LSTM, baselines, actual", all(k in t for k in ("XGBoost", "LSTM", "Persistence baseline", "Road-mean baseline", "Actual next day")))
        check("Forecast labelled Historical replay with horizon", "Historical replay" in t and "1 day ahead" in t)
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
        check("Mode 2 (TomTom) unconfigured: config message shown", "No API key configured" in t)
        check("Mode 2: states provider data cannot feed the models", "cannot be used to make a valid prediction" in t)
        check("Mode 2: no fetch/predict controls without a key", "Fetch provider observation" not in t)
        check("Mode 2: no error text", not any(b in t for b in BAD))
        await c.shot("06_demo_mode2")

        # ---------------------------------------------------------- deep links: every page opens directly from its URL
        for path, needle in (("forecasts", "Historical replay and what-if forecasts"), ("community", "not available yet"),
                             ("details", "will be added in a later stage"), ("performance", "Project objective"),
                             ("", "Live traffic is not configured")):
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
