"""Real-browser end-to-end check of the community PHOTO reports (headless Chrome via the DevTools Protocol; no extra packages).

    python tests/browser_photo_check.py <screenshot-folder> [port] [width height]

It starts its OWN throw-away dashboard on the given port (default 8650) with a temporary community database, a freshness window of 30 seconds and a refresh
every 5 seconds, and no TomTom key, so it never touches your data, your dashboards or your quota. Then it plays two people in two separate browser profiles:

  User A  (location granted):  opens Community Reports, shares a real browser location (set by the DevTools geolocation override), adds a photo (from a camera
          stream and from a file), writes a description and submits.   Also: location denied, and camera denied.
  User B  (another browser profile = another session): opens the Live Traffic Map, chooses a place 0.8 km from where A stood, and sees A's photo on the map and
          under it; clicks the marker; sees the popup; then WAITS until the report expires and watches it disappear WITHOUT reloading the page.

What it cannot prove: that a phone's GPS is truthful (the server cannot either), or the real camera hardware (a fake video stream is used).
"""
import asyncio, base64, io, json, os, shutil, sqlite3, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
from tornado.websocket import websocket_connect

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="photo_check_"))
OUT.mkdir(parents=True, exist_ok=True)
APP_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8650
W, H = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1400, 1000)
URL = f"http://localhost:{APP_PORT}"
FRESH_S, REFRESH_S = 30, 5
AREA_G = (12.9159, 77.6240)
NORTH_08_KM = (12.9159 + 0.8 / 111.32, 77.6240)
_C = [os.environ.get("CHROME_PATH", ""), r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
      r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
CHROME = next((c for c in _C if c and Path(c).exists()), None) or sys.exit("No Chrome/Edge found; set CHROME_PATH.")
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
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
        return (await self.js("document.body.innerText")) or ""

    async def wait_text(self, needle, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if needle in await self.text():
                return True
            await asyncio.sleep(0.4)
        return False

    async def wait_js(self, expr, timeout=40):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if await self.js(expr):
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.4)
        return False

    async def settle(self):
        await asyncio.sleep(0.8)
        t0 = time.time()
        while time.time() - t0 < 60 and await self.js("!!document.querySelector('[data-testid=\"stStatusWidget\"]')"):
            await asyncio.sleep(0.4)
        await asyncio.sleep(0.6)

    async def click_text(self, selector, needle, exact=False):
        return await self.js(f"""(() => {{ const e=[...document.querySelectorAll({json.dumps(selector)})].find(x=>x.innerText && ({json.dumps(exact)} ? x.innerText.trim()==={json.dumps(needle)} : x.innerText.includes({json.dumps(needle)})));
            if(!e) return false; e.scrollIntoView({{block:'center'}}); e.click(); return true; }})()""")

    async def shot(self, name):
        r = await self.call("Page.captureScreenshot", format="png")
        (OUT / f"{name}.png").write_bytes(base64.b64decode(r["data"]))


IFRAME = "document.querySelector('iframe[src*=\"tc_photo_capture\"]')"
FRAME_DOC = IFRAME + ".contentDocument"


async def launch(port, profile, extra_flags=()):
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}", f"--window-size={W},{H}", "--no-first-run",
                             "--disable-gpu", *extra_flags, "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ws = None
    for _ in range(60):
        try:
            ws = next(t["webSocketDebuggerUrl"] for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list")) if t["type"] == "page")
            break
        except Exception:
            await asyncio.sleep(0.5)
    assert ws, "Chrome DevTools not reachable"
    c = CDP(await websocket_connect(ws, max_message_size=80_000_000))
    await c.call("Page.enable")
    await c.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
    return proc, c


async def grant_location(c, lat, lon, allow=True):
    if allow:
        await c.call("Browser.grantPermissions", origin=URL, permissions=["geolocation"])
        await c.call("Emulation.setGeolocationOverride", latitude=lat, longitude=lon, accuracy=20)
    else:
        await c.call("Browser.setPermission", permission={"name": "geolocation"}, setting="denied", origin=URL)


async def open_photo_flow(c):
    await c.call("Page.navigate", url=f"{URL}/community")
    ok = await c.wait_text("Report Traffic with Photo", 120)
    await c.settle()
    await c.click_text("button", "Report Traffic with Photo")
    await c.settle()
    got = await c.wait_js(f"!!{IFRAME} && !!{FRAME_DOC} && !!{FRAME_DOC}.getElementById('btn-loc')", 40)
    return ok and got


async def put_file_photo(c):
    """Give the component a photo through its own file input (what 'Choose an existing photo' does), generated in the page so nothing is read from disk."""
    code = """(async () => { const c = document.createElement('canvas'); c.width = 1600; c.height = 1000; const x = c.getContext('2d');
        x.fillStyle = '#b03030'; x.fillRect(0, 0, 1600, 1000); x.fillStyle = '#fff'; x.font = '120px sans-serif'; x.fillText('HEAVY TRAFFIC', 120, 520);
        const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92)); const f = new File([blob], 'photo.jpg', {type: 'image/jpeg'});
        const dt = new DataTransfer(); dt.items.add(f); const i = document.getElementById('file-choose'); i.files = dt.files;
        i.dispatchEvent(new Event('change', {bubbles: true})); return blob.size; })()"""
    return await c.js(f"{FRAME_DOC}.defaultView.eval({json.dumps(code)})")


def db_rows(db):
    if not Path(db).exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute("select report_id, latitude, longitude, accuracy_m, description, image_file, created_at, expires_at from photo_reports").fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()


async def main():
    work = Path(tempfile.mkdtemp(prefix="photo_check_work_"))
    db = work / "community.db"
    env = {k: v for k, v in os.environ.items() if k not in ("TOMTOM_API_KEY", "LIVE_CACHE_TTL_S", "LIVE_MAX_REQUESTS")}
    env.update({"COMMUNITY_DB_PATH": str(db), "COMMUNITY_PHOTO_FRESH_MINUTES": str(FRESH_S / 60), "COMMUNITY_PHOTO_REFRESH_S": str(REFRESH_S),
                "PYTHONUTF8": "1", "TF_CPP_MIN_LOG_LEVEL": "3"})
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(APP_PORT), "--server.headless", "true",
                            "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs = []
    try:
        for _ in range(90):
            try:
                if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=3).read() == b"ok":
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            sys.exit("the throw-away dashboard did not start")
        print(f"throw-away dashboard on {URL}: temporary database, freshness {FRESH_S} s, refresh {REFRESH_S} s, no TomTom key", flush=True)

        # ===================================================================================== USER A: location granted, file photo
        pa, a = await launch(9440, work / "profile_a")
        procs.append(pa)
        # Warm the server up with the default page first. (Streamlit 1.40 shows a one-off "Page not found" notice when the VERY FIRST request to a freshly started
        # server is a deep link such as /community; the previous release does the same. It is cosmetic, and this keeps the screenshots clean.)
        await a.call("Page.navigate", url=URL)
        await a.wait_text("Live Traffic Map", 180)
        await a.settle()
        await grant_location(a, *AREA_G)
        check("User A: the Community Reports page opens the photo flow from 'Report Traffic with Photo'", await open_photo_flow(a))
        t = await a.text()
        check("User A: the page explains why location is required and that it cannot be typed in", "Why location is required" in t and "cannot type a position in" in t)
        check("User A: Submit is disabled before a photo and a position exist", await a.js("(() => { const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Submit photo report'); return !!b && b.disabled; })()"))
        await a.shot("a1_flow_open")
        await a.js(f"{FRAME_DOC}.getElementById('btn-loc').click()")
        got_loc = await a.wait_js(f"{FRAME_DOC}.getElementById('loc-status').innerText.includes('Location received')", 30)
        check("User A: the browser's geolocation is requested and received (accuracy shown)", got_loc,
              await a.js(f"{FRAME_DOC}.getElementById('loc-status').innerText") or "")
        size = await put_file_photo(a)
        got_img = await a.wait_js(f"{FRAME_DOC}.getElementById('photo-status').innerText.includes('Photo ready')", 30)
        check("User A: a chosen photo is accepted, shrunk in the browser and previewed", got_img and not await a.js(f"{FRAME_DOC}.getElementById('preview').hidden"), f"input {size} bytes")
        await a.settle()
        check("User A: Submit is enabled once both exist", await a.wait_js("(() => { const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Submit photo report'); return !!b && !b.disabled; })()", 30))
        await a.js("""(() => { const i=document.querySelector('input[aria-label="Short description (optional)"]'); i.focus(); i.select(); })()""")
        await a.call("Input.insertText", text="Heavy traffic near junction")
        await a.js("document.activeElement.blur()")
        await asyncio.sleep(1.0)
        await a.shot("a2_ready")
        await a.click_text("button", "Submit photo report", exact=True)
        saved = await a.wait_text("Saved: your photo report", 40)
        await a.settle()
        check("User A: the server confirms the save", saved)
        rows = db_rows(db)
        check("User A: exactly one report is stored with the browser's position, accuracy and description",
              len(rows) == 1 and abs(rows[0][1] - AREA_G[0]) < 1e-6 and abs(rows[0][2] - AREA_G[1]) < 1e-6 and 0 < rows[0][3] <= 1000 and rows[0][4] == "Heavy traffic near junction", str(rows[0][1:5]) if rows else "no row")
        uploaded_at, expires_at = (rows[0][6], rows[0][7]) if rows else (0, 0)
        check("the expiry is the SERVER's upload time plus the configured window", rows and abs((expires_at - uploaded_at) - FRESH_S) < 0.01 and abs(uploaded_at - time.time()) < 120)
        photo_file = work / "photos" / rows[0][5] if rows else None
        check("the photo file exists, is a JPEG and has no metadata", bool(photo_file) and photo_file.is_file() and photo_file.read_bytes()[:2] == b"\xff\xd8" and b"Exif" not in photo_file.read_bytes()[:64])
        t = await a.text()
        check("User A: the new photo is listed as Fresh with its upload time and 'Unverified'", "Heavy traffic near junction" in t and "Fresh" in t and "Uploaded" in t and "Unverified" in t.title() or "UNVERIFIED" in t)
        await a.shot("a3_saved")

        # ===================================================================================== USER B: another browser profile, another session
        pb, b = await launch(9441, work / "profile_b")
        procs.append(pb)
        await b.call("Page.navigate", url=URL)
        await b.wait_text("Live Traffic Map", 120)
        await b.settle()
        t0_b = time.time()
        await b.js("window.__same_page_marker = 'still-here'")     # survives only while the page is NOT reloaded
        await b.click_text("summary, [data-testid='stExpander'] summary", "Enter coordinates instead")
        await b.settle()
        for label, value in (("Latitude", f"{NORTH_08_KM[0]:.6f}"), ("Longitude", f"{NORTH_08_KM[1]:.6f}")):
            await b.js(f"""(() => {{ const i=[...document.querySelectorAll('input[aria-label="{label}"]')].pop(); i.scrollIntoView({{block:'center'}}); i.focus(); i.select(); }})()""")
            await b.call("Input.insertText", text=value)
            await b.js("document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',code:'Enter',keyCode:13,bubbles:true})); document.activeElement.blur()")
            await b.settle()
        await b.click_text("button", "Use these coordinates", exact=True)
        await b.settle()
        MAP = "document.querySelector('iframe[srcdoc]')"
        found_marker = await b.wait_js(f"!!{MAP} && !!{MAP}.contentDocument && {MAP}.contentDocument.querySelectorAll('.tc-photo-marker').length === 1", 40)
        check("User B (another session): the place is chosen and A's photo appears on the map as a community marker", found_marker)
        t = await b.text()
        check("User B: below the map: the photograph's description, upload time, distance and Fresh status",
              "Heavy traffic near junction" in t and "Uploaded" in t and "km from your searched location" in t and "Fresh" in t, "")
        check("User B: the photo is a real image on the page", await b.js("[...document.querySelectorAll('img')].some(i => i.naturalWidth > 100 && i.src.startsWith('http'))"))
        check("User B: the two sources are explained and separate (TomTom line vs purple camera markers)", "Two different sources" in t and "TomTom traffic (provider)" in t)
        await b.shot("b1_map_with_photo")
        await b.js(f"{MAP}.contentDocument.querySelector('.tc-photo-marker').click()")
        popup = await b.wait_js(f"!!{MAP}.contentDocument.querySelector('.leaflet-popup-content .tc-popup')", 15)
        check("User B: clicking the marker opens a popup", popup)
        pop = await b.js(f"""(() => {{ const p={MAP}.contentDocument.querySelector('.leaflet-popup-content .tc-popup'); if(!p) return null;
            const img=p.querySelector('img'); return {{text: p.innerText, img: !!img && img.src.startsWith('data:image/jpeg;base64,'), loaded: !!img && img.naturalWidth > 50}}; }})()""")
        check("User B: the popup shows the photo, description, upload time ('N minutes ago'), distance and status",
              bool(pop) and pop["img"] and pop["loaded"] and "Heavy traffic near junction" in pop["text"] and "Uploaded" in pop["text"] and "km from your searched location" in pop["text"]
              and "Fresh" in pop["text"] and "COMMUNITY PHOTO" in pop["text"].upper(), (pop or {}).get("text", "")[:120].replace("\n", " | "))
        await b.shot("b2_popup")
        # ---- expiry on the timer, without reloading the page
        remaining = (expires_at + REFRESH_S + 6) - time.time()
        print(f"   waiting {max(0, remaining):.0f} s for the report to expire and the page to refresh itself (no reload)...", flush=True)
        await asyncio.sleep(max(0, remaining))
        gone = await b.wait_js(f"{MAP}.contentDocument.querySelectorAll('.tc-photo-marker').length === 0", 30)
        t = await b.text()
        check("User B: after the freshness window the marker and the photo disappear from the live view, without reloading the page",
              gone and "Heavy traffic near junction" not in t.split("Community photos")[-1] and "No recent community traffic reports for this area." in t
              and await b.js("window.__same_page_marker") == "still-here", "")
        await b.shot("b3_expired_gone")
        await a.call("Page.navigate", url=f"{URL}/community")
        await a.wait_text("Photo history", 60)
        await a.settle()
        await a.click_text("summary, [data-testid='stExpander'] summary", "Photo history")
        await a.settle()
        t = await a.text()
        check("the expired report is kept in the history, labelled Expired, and is not listed as recent", "Expired" in t and "Photo history (1 expired)" in t and "No recent community traffic reports." in t)
        await a.shot("a4_history_expired")

        # ===================================================================================== permission denial
        pc, c = await launch(9442, work / "profile_c")
        procs.append(pc)
        await grant_location(c, *AREA_G, allow=False)
        check("Denied location: the photo flow opens", await open_photo_flow(c))
        await c.js(f"{FRAME_DOC}.getElementById('btn-loc').click()")
        denied = await c.wait_js(f"{FRAME_DOC}.getElementById('loc-status').innerText.includes('denied')", 20)
        await c.settle()
        t = await c.text()
        check("Denied location: the component says so, the page explains why location is required, and nothing can be submitted",
              denied and "Location permission was denied" in t and "GPS position" in t
              and await c.js("(() => { const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Submit photo report'); return !!b && b.disabled; })()"))
        await put_file_photo(c)
        await c.wait_js(f"{FRAME_DOC}.getElementById('photo-status').innerText.includes('Photo ready')", 20)
        await c.settle()
        check("Denied location: even WITH a photo the Submit button stays disabled", await c.js("(() => { const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Submit photo report'); return !!b && b.disabled; })()"))
        await c.shot("c1_location_denied")
        await c.js(f"{FRAME_DOC}.defaultView.eval(\"navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('denied', 'NotAllowedError'))\")")
        await c.js(f"{FRAME_DOC}.getElementById('btn-camera').click()")
        cam_denied = await c.wait_js(f"{FRAME_DOC}.getElementById('photo-status').innerText.includes('Camera permission was denied')", 20)
        await c.settle()
        t = await c.text()
        check("Denied camera: the component and the page explain it and offer choosing an existing photo", cam_denied and "Camera permission was denied" in t and "choose an existing photo" in t)
        await c.shot("c2_camera_denied")
        await c.call("Browser.close") if False else None

        # ===================================================================================== a live camera stream (fake device) for a granted camera
        pd_, d = await launch(9443, work / "profile_d", ("--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"))
        procs.append(pd_)
        await grant_location(d, *AREA_G)
        await d.call("Browser.grantPermissions", origin=URL, permissions=["geolocation", "videoCapture"])
        check("Camera granted: the photo flow opens", await open_photo_flow(d))
        await d.js(f"{FRAME_DOC}.getElementById('btn-loc').click()")
        await d.wait_js(f"{FRAME_DOC}.getElementById('loc-status').innerText.includes('Location received')", 30)
        await d.js(f"{FRAME_DOC}.getElementById('btn-camera').click()")
        streaming = await d.wait_js(f"(() => {{ const v={FRAME_DOC}.getElementById('video'); return !v.hidden && v.videoWidth > 0; }})()", 30)
        check("Camera granted: 'Open camera' shows a live stream (a fake device stands in for the hardware)", streaming)
        await d.js(f"{FRAME_DOC}.getElementById('btn-snap').click()")
        snapped = await d.wait_js(f"{FRAME_DOC}.getElementById('photo-status').innerText.includes('Photo ready')", 20)
        await d.settle()
        check("Camera granted: 'Take photo' captures a frame and enables Submit", snapped and await d.wait_js("(() => { const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()==='Submit photo report'); return !!b && !b.disabled; })()", 20))
        await d.shot("d1_camera_snapshot")
    finally:
        for pr in procs:
            try:
                pr.terminate()
            except Exception:
                pass
        srv.terminate()
        try:
            srv.wait(15)
        except Exception:
            srv.kill()
        shutil.rmtree(work, ignore_errors=True)
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} photo-report browser checks passed")
    sys.exit(1 if n_fail else 0)


asyncio.run(main())
