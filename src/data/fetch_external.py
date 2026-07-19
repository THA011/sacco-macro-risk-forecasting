"""
Fetch real macro series to replace `simulate_macro.py` output once you're
running this outside a network-restricted sandbox.

NOT executed as part of this repo's test suite or CI — it depends on
external government sites whose exact page structure changes over time.
Treat these as documented starting points, not finished scrapers: run each,
inspect what actually comes back, and adjust the parsing before trusting it.

Sources:
    EPRA monthly petroleum pricing statements — https://www.epra.go.ke
        (published as PDF press releases; the pricing template lists pump
        prices per litre by product and by region)
    KNBS Consumer Price Index / inflation releases — https://www.knbs.or.ke
        (published as PDF/Excel monthly CPI reports)
    CBK indicative foreign exchange rates — https://www.centralbank.go.ke
        (published as a daily/monthly rates page and downloadable CSV)

Usage:
    python -m src.data.fetch_external --all
    python -m src.data.fetch_external --source epra
"""

import argparse

import requests

from src.config import DATA_EXTERNAL

EPRA_URL = "https://www.epra.go.ke"
KNBS_URL = "https://www.knbs.or.ke"
CBK_URL = "https://www.centralbank.go.ke"

REQUEST_TIMEOUT_SECONDS = 30


def fetch_epra_pump_prices() -> None:
    """
    Download the latest EPRA monthly pricing statement.

    EPRA publishes these as PDF press releases without a stable predictable
    filename — locate the current month's release URL from the site's press
    releases index page first, then fetch it here. Once downloaded, pump
    prices need extracting from the PDF table (the `pdf` skill's table
    extraction approach applies directly if you're doing this from Claude).
    """
    resp = requests.get(EPRA_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    out_path = DATA_EXTERNAL / "epra_index.html"
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"Saved EPRA index page -> {out_path}. Find the current pricing "
          f"statement link on this page and fetch it directly.")


def fetch_knbs_cpi() -> None:
    """Download the KNBS CPI / inflation landing page for the current release."""
    resp = requests.get(KNBS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    out_path = DATA_EXTERNAL / "knbs_index.html"
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"Saved KNBS index page -> {out_path}. Locate the current CPI "
          f"release link on this page and fetch it directly.")


def fetch_cbk_rates() -> None:
    """Download the CBK indicative exchange rates landing page."""
    resp = requests.get(CBK_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    out_path = DATA_EXTERNAL / "cbk_index.html"
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"Saved CBK index page -> {out_path}. Locate the indicative rates "
          f"CSV/table link on this page and fetch it directly.")


SOURCES = {
    "epra": fetch_epra_pump_prices,
    "knbs": fetch_knbs_cpi,
    "cbk": fetch_cbk_rates,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(SOURCES.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.source:
        parser.error("pass --all or --source {epra,knbs,cbk}")

    targets = SOURCES.keys() if args.all else [args.source]
    for name in targets:
        print(f"Fetching {name}...")
        SOURCES[name]()


if __name__ == "__main__":
    main()
