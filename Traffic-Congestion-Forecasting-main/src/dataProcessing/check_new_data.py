"""
Read-only check of a file of NEWER observations before anyone decides to use it. It changes nothing.

Why this exists: the project's dataset ends on 2024-08-09, so the dashboard cannot show or forecast later dates. There is no source of newer
observations in this project and none are ever invented. If you obtain real newer observations (with the right to use them), this tool tells
you, before any use, whether the file is valid and what it would and would not cover.

What it does
  * validates the file with the project's own loader rules (required columns, dates, numbers, known weather categories, no missing values,
    no duplicate road-and-date rows, congestion within 0-100);
  * compares it with the current dataset: date range, overlap, whether it continues after the last observation, gaps, roads, and whether it
    reaches a target date (default: yesterday);
  * says what would still be needed. It never writes, merges, retrains or touches model files.

Run from the project root:   python -m src.dataProcessing.check_new_data path\\to\\new_observations.csv [--target 2026-10-28]
Exit code 0 = the file is valid for these rules; 1 = it is not, or it could not be read.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.dataProcessing import forecast_data as fd  # noqa: E402


def check(path, target=None, existing_path=None) -> dict:
    """Returns a report dict. Raises forecast_data.DatasetError if the file itself is invalid."""
    new = fd.load_raw(str(path))           # the same validation the training and the dashboard use
    old = fd.load_raw(str(existing_path or fd.DATA_PATH))  # read at call time, so the location of the current dataset can be redirected
    target = pd.Timestamp(target) if target else pd.Timestamp(dt.date.today() - dt.timedelta(days=1))
    old_last, new_first, new_last = old[fd.DATE_COL].max(), new[fd.DATE_COL].min(), new[fd.DATE_COL].max()
    key = [fd.DATE_COL, fd.ROAD_COL]
    overlap = int(new.merge(old[key], on=key).shape[0])
    unknown_roads = sorted(set(new[fd.ROAD_COL]) - set(old[fd.ROAD_COL]))
    after = new[new[fd.DATE_COL] > old_last]
    days_after = after[fd.DATE_COL].nunique()
    span_after = (new_last - old_last).days if new_last > old_last else 0
    notes = []
    if overlap:
        notes.append(f"{overlap} rows repeat a date and road that the current dataset already has; they would have to be reconciled, not appended blindly.")
    if unknown_roads:
        notes.append(f"Roads not known to the trained models: {unknown_roads[:5]}. The models cannot use them.")
    if new_last <= old_last:
        notes.append("The file contains nothing after the current last observation, so it cannot extend the dashboard's dates.")
    elif new_first > old_last + pd.Timedelta(days=1):
        notes.append(f"There is a gap of {(new_first - old_last).days - 1} day(s) between the current last observation and this file's first row; "
                     "next-day samples need consecutive days.")
    if new_last < target:
        notes.append(f"The file ends {(target - new_last).days} day(s) before the target date {target:%Y-%m-%d}; later dates would stay unavailable.")
    return {"rows": int(len(new)), "roads": int(new[fd.ROAD_COL].nunique()), "first": new_first, "last": new_last,
            "current_last": old_last, "overlap_rows": overlap, "unknown_roads": unknown_roads, "days_after_current_last": int(days_after),
            "reaches_target": bool(new_last >= target and new_last > old_last), "target": target, "calendar_days_after_current_last": int(span_after),
            "notes": notes}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only validation of a file of newer observations.")
    ap.add_argument("csv")
    ap.add_argument("--target", help="date the dashboard should reach (default: yesterday)")
    a = ap.parse_args(argv)
    try:
        r = check(a.csv, a.target)
    except fd.DatasetError as e:
        print(f"NOT VALID: {e}")
        return 1
    print(f"Valid file: {r['rows']} rows, {r['roads']} roads, {r['first']:%Y-%m-%d} to {r['last']:%Y-%m-%d}.")
    print(f"Current dataset ends {r['current_last']:%Y-%m-%d}; this file adds {r['days_after_current_last']} later day(s); "
          f"reaches {r['target']:%Y-%m-%d}: {'yes' if r['reaches_target'] else 'NO'}.")
    for n in r["notes"]:
        print("  note:", n)
    print("Nothing was changed. Using this data for forecasts would still require merging it deliberately, retraining and re-evaluating the "
          "models (which needs approval), and a check that you may use the data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
