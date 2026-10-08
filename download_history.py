# download_history.py
#
# What it does: downloads every F1 race result since 1950 and every qualifying
#   result the service has (1994 onwards) from the Jolpica F1 API, the free
#   successor to the Ergast API.
# What it reads: https://api.jolpi.ca/ergast/f1/ (internet needed).
# What it produces: the answers exactly as the service sent them, 100 rows per
#   file, in data/jolpica/:
#     results_000000.json, results_000100.json, ...        race results
#     qualifying_000000.json, qualifying_000100.json, ...  qualifying results
#   Nothing is changed or thrown away here. Cleaning is a separate step, so a
#   cleaning mistake never means downloading again.
# Which files use it: clean_history.py reads data/jolpica/.
#
# Run:  .venv\Scripts\python.exe download_history.py
# Safe to stop and run again: pages already saved are skipped.
# Expect about 5 minutes the first time (about 380 pages), seconds after that.

import json
import time
from pathlib import Path

import requests

API = "https://api.jolpi.ca/ergast/f1"
FOLDER = Path("data/jolpica")
PAGE_SIZE = 100  # the most rows the service gives per request

# The service allows 4 requests a second and 500 an hour. One request every
# half second stays far inside the first limit, and everything this script
# needs is under 500 requests. It is a free service run by volunteers.
SECONDS_BETWEEN_REQUESTS = 0.5
SECONDS_TO_WAIT_WHEN_LIMITED = 120

# Each kind of page keeps its rows under a different name inside a race.
ROWS_KEY = {"results": "Results", "qualifying": "QualifyingResults"}


def page_file(kind: str, offset: int) -> Path:
    """Where one page is saved, for example data/jolpica/results_000100.json."""
    return FOLDER / f"{kind}_{offset:06d}.json"


def count_rows(page: dict, kind: str) -> int:
    """How many driver rows a page holds.

    A page is a list of races, each with a list of drivers. The 100 row limit
    counts drivers, so one race is often split across two pages.
    """
    races = page["MRData"]["RaceTable"]["Races"]
    return sum(len(race[ROWS_KEY[kind]]) for race in races)


def fetch_page(kind: str, offset: int) -> dict:
    """Ask the service for one page, waiting and retrying if it says "slow down"."""
    while True:
        response = requests.get(
            f"{API}/{kind}/", params={"limit": PAGE_SIZE, "offset": offset}, timeout=60
        )
        if response.status_code == 429:  # 429 means "too many requests"
            print(f"  the service asked us to slow down, waiting {SECONDS_TO_WAIT_WHEN_LIMITED}s...")
            time.sleep(SECONDS_TO_WAIT_WHEN_LIMITED)
            continue
        response.raise_for_status()
        time.sleep(SECONDS_BETWEEN_REQUESTS)
        return response.json()


def saved_full_page(kind: str, offset: int) -> dict | None:
    """A page saved earlier, if it is full. Otherwise None.

    A full page (100 rows) can never change: results are in date order and new
    races are added at the end. A page with fewer rows was the last page when
    it was saved, so it is downloaded again in case new races have been added.
    """
    file = page_file(kind, offset)
    if not file.exists():
        return None
    page = json.loads(file.read_text(encoding="utf-8"))
    return page if count_rows(page, kind) == PAGE_SIZE else None


def download_all(kind: str) -> int:
    """Download every page of one kind. Returns how many rows there are."""
    total_rows = 0
    offset = 0
    while True:
        page = saved_full_page(kind, offset)
        if page is None:
            page = fetch_page(kind, offset)
            page_file(kind, offset).write_text(json.dumps(page), encoding="utf-8")
            print(f"{kind}  rows {offset} to {offset + PAGE_SIZE - 1} downloaded")
        rows = count_rows(page, kind)
        total_rows += rows
        if rows < PAGE_SIZE:  # a page that is not full is the last one
            return total_rows
        offset += PAGE_SIZE


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    for kind in ROWS_KEY:
        rows = download_all(kind)
        print(f"{kind}: {rows:,} rows saved in {FOLDER}")


if __name__ == "__main__":
    main()
