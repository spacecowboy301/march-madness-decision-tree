from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
import pandas as pd

from .team_names import normalize_team_name


BASE_URL = "https://kenpom.com"
RAW_DIR = Path("data/raw/kenpom")
PROCESSED_PATH = Path("data/processed/kenpom_team_features.csv")


def flatten_columns(columns) -> list[str]:
    flat = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(part) for part in col if str(part) != "nan" and not str(part).startswith("Unnamed")]
            name = "_".join(parts)
        else:
            name = str(col)
        name = re.sub(r"\s+", "_", name.strip().lower())
        name = re.sub(r"[^a-z0-9_]+", "", name)
        flat.append(name.strip("_"))
    counts: dict[str, int] = {}
    unique = []
    for name in flat:
        base = name or "col"
        counts[base] = counts.get(base, 0) + 1
        unique.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return unique


def clean_numeric(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    text = re.sub(r"\s*\d+$", "", text) if re.search(r"[+-]?\d+\.\d+\s+\d+$", text) else text
    text = text.replace("%", "")
    if text in {"", "-", "--"}:
        return pd.NA
    try:
        return float(text)
    except ValueError:
        return value


class KenPomClient:
    def __init__(self, email: str | None, password: str | None, delay: float = 1.0):
        self.session = requests.Session()
        self.delay = delay
        self.email = email
        self.password = password
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; local research script; +https://kenpom.com)",
                "Referer": BASE_URL,
            }
        )

    def login(self) -> None:
        if not self.email or not self.password:
            return
        login_url = f"{BASE_URL}/handlers/login_handler.php"
        payload = {"email": self.email, "password": self.password, "remember": "1", "submit": "Login!"}
        response = self.session.post(login_url, data=payload, timeout=30)
        response.raise_for_status()

    def get(self, path: str, **params) -> str:
        query = ""
        if params:
            query = "?" + "&".join(f"{key}={quote_plus(str(value))}" for key, value in params.items())
        response = self.session.get(f"{BASE_URL}/{path}{query}", timeout=30)
        response.raise_for_status()
        time.sleep(self.delay)
        return response.text


EFFICIENCY_COLUMNS = [
    "rk",
    "team",
    "conf",
    "wl",
    "netrtg",
    "ortg",
    "ortg_rank",
    "drtg",
    "drtg_rank",
    "adjt",
    "adjt_rank",
    "luck",
    "luck_rank",
    "sos_netrtg",
    "sos_netrtg_rank",
    "sos_ortg",
    "sos_ortg_rank",
    "sos_drtg",
    "sos_drtg_rank",
    "ncsos_netrtg",
    "ncsos_netrtg_rank",
]


def expand_header_names(table, width: int, source: str) -> list[str]:
    header_rows = table.select("thead tr")
    if source == "efficiency" and width == len(EFFICIENCY_COLUMNS):
        return EFFICIENCY_COLUMNS
    if not header_rows:
        return [f"col_{idx}" for idx in range(width)]

    names = []
    for th in header_rows[-1].find_all("th"):
        label = th.get("title") or th.get_text(" ", strip=True) or "col"
        label = re.sub(r"\s+", "_", label.lower())
        label = re.sub(r"[^a-z0-9_]+", "", label).strip("_") or "col"
        colspan = int(th.get("colspan", "1"))
        if colspan == 1:
            names.append(label)
        else:
            names.append(label)
            for idx in range(2, colspan + 1):
                suffix = "rank" if idx == 2 else str(idx)
                names.append(f"{label}_{suffix}")
    if len(names) < width:
        names.extend(f"col_{idx}" for idx in range(len(names), width))
    return flatten_columns(names[:width])


def parse_kenpom_table(html: str, source: str, year: int) -> pd.DataFrame:
    if "available to subscribers only" in html.lower():
        raise PermissionError(f"{source} requires a KenPom subscriber login for {year}.")

    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table#ratings-table") or soup.find("table")
    if table is None:
        raise ValueError(f"No table found in {source} for {year}.")

    rows = []
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue
        rank_text = cells[0].get_text(" ", strip=True)
        if not rank_text.isdigit():
            continue
        row = []
        for idx, cell in enumerate(cells):
            if idx == 1:
                team_link = cell.find("a", href=re.compile(r"team\.php"))
                row.append(team_link.get_text(" ", strip=True) if team_link else cell.get_text(" ", strip=True))
            else:
                row.append(cell.get_text(" ", strip=True))
        rows.append(row)

    if not rows:
        raise ValueError(f"No team rows found in {source} for {year}.")

    width = max(len(row) for row in rows)
    names = expand_header_names(table, width, source)
    records = [row + [pd.NA] * (width - len(row)) for row in rows]
    parsed = pd.DataFrame(records, columns=names)
    if "team" not in parsed.columns and width > 1:
        parsed = parsed.rename(columns={parsed.columns[1]: "team"})
    parsed["season"] = year
    parsed["source"] = source
    return parsed


def clean_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    if "team" not in df.columns:
        raise ValueError(f"Missing team column for {prefix}")
    df["team"] = df["team"].astype(str).str.replace(r"\s+\d+$", "", regex=True).str.strip()
    df["team_norm"] = df["team"].map(normalize_team_name)
    keep = ["season", "team", "team_norm"]
    for col in df.columns:
        if col in keep or col in {"source"}:
            continue
        new_col = f"{prefix}_{col}"
        df[new_col] = df[col].map(clean_numeric)
        keep.append(new_col)
    return df[keep]


def scrape_efficiency(client: KenPomClient, year: int) -> pd.DataFrame:
    html = client.get("index.php", y=year)
    table = parse_kenpom_table(html, "efficiency", year)
    return clean_table(table, "eff")


def scrape_four_factors(client: KenPomClient, year: int) -> pd.DataFrame:
    html = client.get("stats.php", y=year)
    table = parse_kenpom_table(html, "four_factors", year)
    return clean_table(table, "ff")


def scrape_misc(client: KenPomClient, year: int) -> pd.DataFrame:
    html = client.get("teamstats.php", y=year)
    table = parse_kenpom_table(html, "misc", year)
    return clean_table(table, "misc")


def scrape_team_page_details(client: KenPomClient, year: int, teams: pd.Series) -> pd.DataFrame:
    rows = []
    for team in teams.dropna().unique():
        html = client.get("team.php", team=team, y=year)
        if "available to subscribers only" in html.lower():
            raise PermissionError(f"team.php requires a KenPom subscriber login for {year}.")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        row = {"season": year, "team": team, "team_norm": normalize_team_name(team)}
        for label in ["Experience", "Continuity", "Average Height", "Bench Minutes"]:
            match = re.search(rf"{re.escape(label)}\s+([+-]?\d+(?:\.\d+)?)", text, flags=re.I)
            if match:
                key = "team_page_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
                row[key] = float(match.group(1))
        rows.append(row)
    return pd.DataFrame(rows)


def merge_year_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    merged = tables[0]
    for table in tables[1:]:
        suffix_cols = [col for col in table.columns if col not in {"season", "team_norm"}]
        merged = merged.merge(table[["season", "team_norm", *suffix_cols]], on=["season", "team_norm"], how="outer")
    team_cols = [col for col in merged.columns if col == "team" or col.endswith("_team")]
    if team_cols:
        merged["team"] = merged[team_cols].bfill(axis=1).iloc[:, 0]
        merged = merged.drop(columns=[col for col in team_cols if col != "team"])
    return merged


def combine_cached_years(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        path = RAW_DIR / f"kenpom_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"WARNING: missing cached KenPom {year}: {path}")
    if not frames:
        raise FileNotFoundError(f"No cached KenPom files found in {RAW_DIR}")
    combined = pd.concat(frames, ignore_index=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PROCESSED_PATH, index=False)
    print(f"Wrote {PROCESSED_PATH} with {len(combined):,} rows")
    return combined


def scrape_year(
    client: KenPomClient,
    year: int,
    include_team_pages: bool = False,
    public_efficiency_only: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    cached = RAW_DIR / f"kenpom_{year}.csv"
    if use_cache and cached.exists():
        print(f"Using cached KenPom {year}")
        return pd.read_csv(cached)

    tables = [scrape_efficiency(client, year)]
    if not public_efficiency_only:
        try:
            tables.append(scrape_four_factors(client, year))
        except PermissionError as exc:
            print(f"WARNING: {exc}")
        try:
            tables.append(scrape_misc(client, year))
        except PermissionError as exc:
            print(f"WARNING: {exc}")
    if include_team_pages and not public_efficiency_only:
        tables.append(scrape_team_page_details(client, year, tables[0]["team"]))
    merged = merge_year_tables(tables)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(RAW_DIR / f"kenpom_{year}.csv", index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2002)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--include-team-pages", action="store_true")
    parser.add_argument("--public-efficiency-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    if args.combine_only:
        combine_cached_years(args.start_year, args.end_year)
        return

    client = KenPomClient(os.getenv("KENPOM_EMAIL"), os.getenv("KENPOM_PASSWORD"), delay=args.delay)
    client.login()

    all_years = []
    for year in range(args.start_year, args.end_year + 1):
        print(f"Scraping KenPom {year}")
        all_years.append(
            scrape_year(
                client,
                year,
                include_team_pages=args.include_team_pages,
                public_efficiency_only=args.public_efficiency_only,
                use_cache=not args.refresh,
            )
        )

    combined = pd.concat(all_years, ignore_index=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PROCESSED_PATH, index=False)
    print(f"Wrote {PROCESSED_PATH} with {len(combined):,} rows")


if __name__ == "__main__":
    main()
