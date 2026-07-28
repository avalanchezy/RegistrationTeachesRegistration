from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HTML gallery for coarse-to-fine registration outputs."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="STS Task 2 registration gallery")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def _number(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _read_summary(run_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    summary = run_dir / "summary.csv"
    if not summary.exists():
        return {}
    with summary.open(newline="", encoding="utf-8-sig") as handle:
        return {
            (row.get("case_id", ""), row.get("jaw", "")): row
            for row in csv.DictReader(handle)
        }


def _case_metadata(case_dir: Path, summary: dict[tuple[str, str], dict[str, str]]) -> dict[str, object]:
    payload = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    record = payload.get("record", {})
    key = (str(record.get("case_id", case_dir.name.split("_")[0])), str(record.get("jaw", "")))
    row = summary.get(key, {})
    metrics = payload.get("metrics", {})
    registration = payload.get("registration", {})
    return {
        "case_id": key[0],
        "jaw": key[1],
        "source": payload.get("source_variant", row.get("source_variant", "")),
        "target": payload.get("target", {}).get("name", row.get("target", "")),
        "method": registration.get("method", row.get("method", "")),
        "tre": metrics.get("mean_tre_mm", row.get("mean_tre_mm")),
        "rotation": metrics.get("rotation_error_deg", row.get("rotation_error_deg")),
        "translation": metrics.get("translation_error_mm", row.get("translation_error_mm")),
        "fit": registration.get("score", row.get("fit_score_mm")),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "gallery.html").resolve()
    summary = _read_summary(run_dir)
    case_dirs = sorted(
        path for path in run_dir.iterdir() if path.is_dir() and (path / "result.json").exists()
    )
    if args.max_cases > 0:
        case_dirs = case_dirs[: args.max_cases]

    cards: list[str] = []
    for case_dir in case_dirs:
        metadata = _case_metadata(case_dir, summary)
        image_path = (case_dir / "stages.png").resolve()
        try:
            image_href = image_path.relative_to(output.parent).as_posix()
        except ValueError:
            image_href = image_path.as_uri()
        caption = (
            f"TRE {_number(metadata['tre'])} mm | rotation {_number(metadata['rotation'], 2)} deg | "
            f"translation {_number(metadata['translation'])} mm | fit {_number(metadata['fit'])} mm"
        )
        cards.append(
            "\n".join(
                [
                    '<article class="case">',
                    f"<h2>{html.escape(str(metadata['case_id']))} {html.escape(str(metadata['jaw']))}</h2>",
                    f'<a href="{html.escape(image_href)}"><img loading="lazy" src="{html.escape(image_href)}" '
                    f'alt="Registration stages for {html.escape(str(metadata["case_id"]))} {html.escape(str(metadata["jaw"]))}"></a>',
                    f"<p class=\"metrics\">{html.escape(caption)}</p>",
                    f"<p>{html.escape(str(metadata['source']))} to {html.escape(str(metadata['target']))}<br>"
                    f"{html.escape(str(metadata['method']))}</p>",
                    "</article>",
                ]
            )
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(args.title)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Segoe UI, sans-serif; color: #202124; }}
body {{ margin: 0; background: #f4f6f8; }}
header {{ padding: 24px max(24px, 4vw) 16px; background: white; border-bottom: 1px solid #dfe3e8; }}
h1 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
header p {{ margin: 8px 0 0; color: #5f6368; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(560px, 100%), 1fr)); gap: 16px; padding: 20px max(20px, 4vw) 40px; }}
.case {{ min-width: 0; background: white; border: 1px solid #dfe3e8; border-radius: 6px; overflow: hidden; }}
.case h2 {{ margin: 14px 16px 10px; font-size: 18px; letter-spacing: 0; }}
.case img {{ display: block; width: 100%; height: auto; border-block: 1px solid #e5e7eb; }}
.case p {{ margin: 10px 16px 14px; color: #5f6368; font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; }}
.case .metrics {{ color: #202124; font-weight: 600; }}
</style>
</head>
<body>
<header><h1>{html.escape(args.title)}</h1><p>{len(cards)} jaw registrations. Click a panel for the full-resolution coarse-to-fine view.</p></header>
<main>{''.join(cards)}</main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output} with {len(cards)} cases")


if __name__ == "__main__":
    main()
