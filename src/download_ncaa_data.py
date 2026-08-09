from __future__ import annotations

import argparse
from pathlib import Path

import requests


DATA_URLS = {
    "MTeams.csv": "https://huggingface.co/Jensen-holm/Nigl/resolve/main/data/MTeams.csv",
    "MNCAATourneyCompactResults.csv": "https://huggingface.co/Jensen-holm/Nigl/resolve/main/data/MNCAATourneyCompactResults.csv",
    "MNCAATourneySeeds.csv": "https://huggingface.co/Jensen-holm/Nigl/resolve/main/data/MNCAATourneySeeds.csv",
    "MRegularSeasonDetailedResults.csv": (
        "https://huggingface.co/Jensen-holm/Nigl/resolve/main/data/MRegularSeasonDetailedResults.csv"
    ),
}


def download_file(url: str, path: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Replace files that already exist.")
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in DATA_URLS.items():
        target = raw_dir / filename
        if target.exists() and not args.refresh:
            print(f"Using existing {target}")
            continue
        print(f"Downloading {filename}")
        download_file(url, target)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
