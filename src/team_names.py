import re
import unicodedata


ALIASES = {
    "brigham young": "byu",
    "cal st bakersfield": "cal st. bakersfield",
    "cal st fullerton": "cal st. fullerton",
    "central connecticut state": "central connecticut",
    "college of charleston": "charleston",
    "connecticut": "connecticut",
    "east tennessee state": "east tennessee st.",
    "florida international": "fiu",
    "georgia state": "georgia st.",
    "illinois chicago": "uic",
    "iowa state": "iowa st.",
    "kennesaw state": "kennesaw st.",
    "louisiana lafayette": "louisiana",
    "loyola chicago": "loyola il",
    "loyola md": "loyola maryland",
    "miami fl": "miami fl",
    "miami florida": "miami fl",
    "mississippi": "ole miss",
    "mount st marys": "mount st. mary's",
    "michigan state": "michigan st.",
    "nc state": "n.c. state",
    "north carolina state": "n.c. state",
    "north dakota state": "north dakota st.",
    "ohio state": "ohio st.",
    "pittsburgh": "pitt",
    "saint marys": "saint mary's",
    "southern california": "usc",
    "st johns": "st. john's",
    "texas a m": "texas a&m",
    "texas am corpus christi": "texas a&m corpus chris",
    "tennessee state": "tennessee st.",
    "uc santa barbara": "ucsb",
    "uconn": "connecticut",
    "utah state": "utah st.",
    "virginia commonwealth": "vcu",
    "wright state": "wright st.",
}


def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = text.lower()
    text = text.replace("&amp;", "&")
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9&. ]+", " ", text)
    text = re.sub(r"\bst\b", "st.", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = ALIASES.get(text, text)
    return text
