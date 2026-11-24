"""Compare downstream responses to supplied keys on the same target images."""

import argparse
import json
from pathlib import Path

from PIL import Image
from run_probe import Reader, correct_value


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("data/models/qwen2.5-vl-7b"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path("data/mmlongbench-full.json").read_text())
    questions = {
        q["id"]: q["question"].strip() for doc in manifest["documents"] for q in doc["questions"]
    }
    cases = {c["id"]: c for c in json.loads(Path(__file__).with_name("cases.json").read_text())}
    source = json.loads((args.focused / "results.json").read_text())
    crops = [row for row in source["rows"] if row["view"] == "crop"]
    assert set(cases) == {row["id"] for row in crops}

    reader = Reader(args.model, "cuda")
    outputs = []
    for row in crops:
        case = cases[row["id"]]
        conditions = []
        with Image.open(row["target"]["image"]) as image:
            for name in ("gold", "alternate"):
                key = case[f"{name}_key"]
                expected = case[f"{name}_value"]
                result = reader.read(
                    image,
                    f"Original question: {questions[row['id']]}\n"
                    f"Use the following supplied intermediate key to answer the downstream measurement: {key}.\n"
                    f"Measurement guide: {case['value_prompt']}\n"
                    "Use the label matching the supplied key. If the supplied key has no matching value, "
                    'return null. Return only JSON {"key": "label used", "value": number or null}.',
                    320,
                )
                result_payload = result["parsed"]
                value = result_payload.get("value") if isinstance(result_payload, dict) else None
                if expected is None:
                    valid = (
                        isinstance(result_payload, dict)
                        and "value" in result_payload
                        and value is None
                    )
                else:
                    valid = correct_value(value, expected, case["decimals"])
                bound = row["oracle"] if name == "gold" else row["alternate"]
                conditions.append(
                    {
                        "condition": name,
                        "key": key,
                        "expected": expected,
                        "free_value": value,
                        "free_correct": valid,
                        "bound_value": bound["value"]
                        if name == "gold"
                        else (bound["selected"] or {}).get("value"),
                        "bound_correct": bound["answer_correct"]
                        if name == "gold"
                        else bound["correct"],
                        "free_read": result,
                    }
                )
        outputs.append(
            {"id": row["id"], "target": row["target"]["image"], "conditions": conditions}
        )
        print(
            row["id"],
            [(x["condition"], x["free_value"], x["bound_value"]) for x in conditions],
            flush=True,
        )

    args.output.write_text(
        json.dumps({"metadata": source["metadata"], "rows": outputs}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    run()
