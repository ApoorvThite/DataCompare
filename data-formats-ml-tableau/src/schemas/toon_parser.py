# src/schemas/toon_parser.py

from pathlib import Path
import json


def parse_toon_literal(text: str):
    """
    Parse a TOON literal string into a Python value.

    Handles:
    - null -> None
    - [ ... ] -> list using JSON parsing
    - "string" -> string with quotes stripped
    - bare numbers -> int or float
    - fallback: raw string
    """
    s = text.strip()

    if s == "null":
        return None

    # List or JSON style literal
    if s.startswith("[") and s.endswith("]"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s

    # Quoted string
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        inner = s[1:-1]
        inner = inner.replace('\\"', '"')
        return inner

    # Try int then float
    try:
        i = int(s)
        return i
    except ValueError:
        pass

    try:
        f = float(s)
        return f
    except ValueError:
        pass

    return s


def parse_toon_records(path: Path) -> list[dict]:
    """
    Parse the TOON file and return a list of record dicts.

    Expects blocks like:

    record Job {
      job_id = "JOB000001"
      title = "Data Scientist"
      ...
    }
    """
    if not path.exists():
        raise FileNotFoundError(f"TOON file not found at {path}")

    records: list[dict] = []
    current: dict | None = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # Start of a record
            if line.startswith("record Job"):
                current = {}
                continue

            # End of a record
            if line == "}" and current is not None:
                records.append(current)
                current = None
                continue

            # Inside a record, parse key = value
            if current is not None and "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                current[key] = parse_toon_literal(val)

    return records
