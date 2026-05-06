from __future__ import annotations

from pathlib import Path

import requests


DATA_URLS = {
    "MTeams.csv": "https://huggingface.co/Jensen-holm/Nigl/resolve/main/data/MTeams.csv",
    "MNCAATourneyCompactResults.csv": "https://huggingface.co/Jensen-holm/Nigl/resolve/main/data/MNCAATourneyCompactResults.csv",
}


def download_file(url: str, path: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in DATA_URLS.items():
        target = raw_dir / filename
        print(f"Downloading {filename}")
        download_file(url, target)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
