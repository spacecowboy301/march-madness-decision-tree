"""Fail when tracked files could expose an owner-funded account."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ENV_FILES = {".env.example", ".env.sample"}
SECRET_PATTERNS = {
    "OpenAI API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT, text=True
    )
    return [ROOT / name for name in output.split("\0") if name]


def main() -> None:
    files = tracked_files()
    findings: list[str] = []

    for path in files:
        name = path.name
        relative = path.relative_to(ROOT)
        if (
            (name.startswith(".env") and name not in ALLOWED_ENV_FILES)
            or path.suffix.lower() in {".key", ".pem", ".p12", ".pfx"}
            or str(relative).startswith(".github/workflows/")
        ):
            findings.append(f"tracked sensitive file: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{relative}: possible {label}")

    if findings:
        raise SystemExit("Cost-safety audit failed:\n" + "\n".join(findings))

    print(f"Cost-safety audit passed ({len(files)} tracked files checked).")


if __name__ == "__main__":
    main()
