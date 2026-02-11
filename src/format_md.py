from __future__ import annotations

import argparse
import re
from pathlib import Path


TABLE_SPLIT_RE = re.compile(r"\s{2,}")
LINK_RE = re.compile(r"\(([^)]+)\)")


def _is_table_line(line: str) -> bool:
    if "|" in line:
        return False
    parts = [part.strip() for part in TABLE_SPLIT_RE.split(line.strip())]
    return len(parts) >= 2 and all(parts)


def _format_table_block(lines: list[str]) -> list[str]:
    rows = [
        [part.strip() for part in TABLE_SPLIT_RE.split(line.strip())]
        for line in lines
    ]
    if not rows:
        return lines

    header = rows[0]
    col_count = len(header)
    normalized = [
        row + [""] * (col_count - len(row))
        if len(row) < col_count
        else row[:col_count]
        for row in rows
    ]

    sep = ["---"] * col_count
    table_lines = [
        "| " + " | ".join(normalized[0]) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in normalized[1:]:
        table_lines.append("| " + " | ".join(row) + " |")
    return table_lines


def _normalize_links(line: str) -> str:
    def replace(match: re.Match[str]) -> str:
        target = match.group(1).replace("\\", "/")
        return f"({target})"

    return LINK_RE.sub(replace, line)


def format_markdown(text: str) -> str:
    lines = text.splitlines()
    formatted: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "":
            formatted.append(line)
            idx += 1
            continue

        if _is_table_line(line):
            block = [line]
            idx += 1
            while idx < len(lines) and _is_table_line(lines[idx]):
                block.append(lines[idx])
                idx += 1
            formatted.extend(_format_table_block(block))
            continue

        formatted.append(_normalize_links(line))
        idx += 1

    return "\n".join(formatted) + "\n"


def format_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    formatted = format_markdown(content)
    path.write_text(formatted, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format markdown tables and links in a file."
    )
    parser.add_argument("path", type=Path, help="Markdown file to format")
    args = parser.parse_args()
    format_file(args.path)


if __name__ == "__main__":
    main()
