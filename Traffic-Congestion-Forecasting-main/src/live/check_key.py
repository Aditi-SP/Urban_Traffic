"""
Safe self-test of the TomTom setup. It never prints the key (not even part of it or its length).

    python -m src.live.check_key              what a dashboard started from THIS terminal would see (no network request at all)
    python -m src.live.check_key --request    additionally make exactly ONE real request to the Traffic Flow API for a fixed point in
                                              Bengaluru and say what came back

Without --request nothing leaves your computer. With --request one request is sent to TomTom with your key (it counts against your
TomTom quota, nothing is stored). The request uses the same client as the dashboard (src/live/tomtom.py), so a success here means the key,
the Traffic Flow API access and the response format all work for that point.

Result of --request, and exit code
  0  OK                    a valid traffic reading came back
  1  NOT CONFIGURED        no usable key was found (the reasons are printed)
  2  REQUEST FAILED        TomTom or the network said no; one of: key not accepted, key may not be enabled for the Traffic Flow API, quota may be used up,
                           rate limit, timeout, no connection, TomTom problem, no traffic data near the point, unreadable answer
Remember: this checks THIS terminal's environment. The running dashboard has its own: see "Diagnostics" in its sidebar.
"""
import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.live import credentials, geo, tomtom  # noqa: E402

TEST_POINT = (12.9159, 77.6240)  # Silk Board Junction area: a main road inside the Bengaluru area
LABELS = {"auth": "KEY NOT ACCEPTED", "forbidden": "KEY MAY NOT BE ENABLED FOR THE TRAFFIC FLOW API", "quota": "QUOTA MAY BE USED UP", "rate_limited": "RATE LIMITED",
          "timeout": "TIMEOUT", "network": "NO CONNECTION", "server": "TOMTOM PROBLEM", "no_segment": "NO TRAFFIC DATA NEAR THE POINT",
          "invalid_response": "UNREADABLE ANSWER", "unexpected_status": "UNEXPECTED ANSWER", "no_key": "NOT CONFIGURED"}


def main(argv=None, *, http_get=None, out=None) -> int:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description="Safe self-test of the TomTom setup (never prints the key).")
    ap.add_argument("--request", action="store_true", help="also make ONE real request to the Traffic Flow API (counts against your quota)")
    args = ap.parse_args(argv)
    say = lambda text="": print(text, file=out)

    cred = credentials.load()
    for line in cred.report_lines():
        say(line)
    if not cred.configured:
        say("\nRESULT: NOT CONFIGURED. See README > 'Setting up TOMTOM_API_KEY'.")
        return 1
    if not args.request:
        say("\nRESULT: a key is configured in this terminal. That does not prove it works: run again with --request to test it once.")
        return 0

    say(f"\nSending ONE request to the TomTom Traffic Flow API for {TEST_POINT[0]}, {TEST_POINT[1]} ...")
    try:
        obs = tomtom.fetch_flow(*TEST_POINT, cred.key, http_get=http_get)
    except tomtom.ProviderError as e:
        say(f"RESULT: REQUEST FAILED ({LABELS.get(e.kind, e.kind)}). {e.message}")
        return 2
    except geo.LocationError as e:
        say(f"RESULT: REQUEST NOT SENT: {e}")
        return 2
    except Exception as e:  # noqa: BLE001  only the type is printed: the text of an unexpected exception could contain the request URL
        say(f"RESULT: REQUEST FAILED (unexpected {type(e).__name__}).")
        return 2
    say(f"RESULT: OK. TomTom returned a valid traffic reading: current speed {obs.current_speed_kmh:.0f} km/h, free-flow speed "
        f"{obs.free_flow_speed_kmh:.0f} km/h, confidence {obs.confidence:.0%}.")
    say("This key can use the Traffic Flow API for this point. It says nothing about other roads, the quota that is left, or TomTom's terms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
