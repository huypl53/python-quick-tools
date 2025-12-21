#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Escape a CSV file's raw text so it can be pasted into a JSON payload field.",
    )
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write the .txt file. Defaults to the CSV name with a .txt extension in the same directory.",
    )
    parser.add_argument(
        "--ensure-ascii",
        action="store_true",
        help="Escape non-ASCII characters in the JSON output.",
    )
    parser.add_argument(
        "--no-quotes",
        action="store_true",
        help="Omit the surrounding JSON quotes; useful when pasting into an existing JSON string field.",
    )
    return parser.parse_args()


def read_raw(csv_path: Path) -> str:
    # Read the file as raw text so we preserve the exact content
    return csv_path.read_text(encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_path = (
        Path(args.output).expanduser().resolve() if args.output else csv_path.with_suffix(".txt")
    )

    raw_content = read_raw(csv_path)

    # json.dumps escapes quotes, backslashes, and newlines so the result can be dropped into a JSON string value.
    escaped = json.dumps(raw_content, ensure_ascii=args.ensure_ascii)

    if args.no_quotes and len(escaped) >= 2 and escaped[0] == '"' and escaped[-1] == '"':
        escaped = escaped[1:-1]

    output_path.write_text(escaped, encoding="utf-8")

    print(f"Wrote escaped content ({len(raw_content)} chars) to {output_path}")


if __name__ == "__main__":
    main()
