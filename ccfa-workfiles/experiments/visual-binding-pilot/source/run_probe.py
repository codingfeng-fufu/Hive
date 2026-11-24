"""Test exact evidence binding with frozen visual reads on known document pages."""

import argparse
import csv
import json
import re
import time
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path

import pymupdf
from PIL import Image


def render(pdf: pymupdf.Document, region: list, view: str, path: Path) -> Image.Image:
    number, bbox = region
    page = pdf[number - 1]
    x0, y0, x1, y1 = bbox
    width, height = page.rect.width, page.rect.height
    clip = (
        pymupdf.Rect(x0 * width, y0 * height, x1 * width, y1 * height)
        if view == "crop"
        else page.rect
    )
    pixmap = page.get_pixmap(clip=clip, dpi=144, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return image


def read_json(raw: str) -> dict | list:
    opening = re.search(r"[\[{]", raw)
    if opening is None:
        return {}
    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw[opening.start() :])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, (dict, list)) else {}


def normalized(label: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).casefold()) if label is not None else ""


def read_number(value: object) -> Decimal | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip().replace(",", "").removesuffix("%"))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def correct_value(value: object, expected: float, decimals: int) -> bool:
    actual = read_number(value)
    if actual is None:
        return False
    precision = Decimal(1).scaleb(-decimals)
    return actual.quantize(precision, rounding=ROUND_HALF_UP) == Decimal(str(expected)).quantize(
        precision, rounding=ROUND_HALF_UP
    )


def evaluate(case: dict, source: dict, target: dict, *, year_keys: bool = False) -> dict:
    aliases = {normalized(k): normalized(v) for k, v in case.get("aliases", {}).items()}

    def key(label: object) -> str:
        if year_keys:
            years = re.findall(r"\b(?:19|20)\d{2}\b", str(label))
            return years[0] if len(years) == 1 else ""
        name = normalized(label)
        return aliases.get(name, name)

    def lookup(label: object) -> dict | None:
        name = key(label)
        matches = [item for item in candidates if name and key(item.get("key")) == name]
        if len(matches) == 1 and read_number(matches[0].get("value")) is not None:
            return matches[0]
        return None

    entries = target.get("candidates")
    candidates = (
        [item for item in entries if isinstance(item, dict)] if isinstance(entries, list) else []
    )
    selected = target.get("selected")
    free = selected if isinstance(selected, dict) else {}
    predicted_key = source.get("key")
    bound = lookup(predicted_key)
    oracle = lookup(case["gold_key"])
    alternate = lookup(case["alternate_key"])

    def grade(record: dict | None) -> dict:
        record = record or {}
        return {
            "key": record.get("key"),
            "value": record.get("value"),
            "key_correct": key(record.get("key")) == key(case["gold_key"]),
            "answer_correct": correct_value(
                record.get("value"), case["gold_value"], case["decimals"]
            ),
        }

    expected_alt = case["alternate_value"]
    alternate_correct = (
        bool(candidates) and alternate is None
        if expected_alt is None
        else (
            alternate is not None
            and correct_value(alternate.get("value"), expected_alt, case["decimals"])
        )
    )
    return {
        "source_key": predicted_key,
        "source_key_correct": key(predicted_key) == key(case["gold_key"]),
        "free": grade(free),
        "bound": grade(bound),
        "oracle": grade(oracle),
        "alternate": {
            "input_key": case["alternate_key"],
            "selected": alternate,
            "expected": expected_alt,
            "correct": alternate_correct,
        },
    }


def paired_reads(
    reader: "Reader", case: dict, question: str, source_path: Path, target_path: Path, protocol: str
) -> tuple[dict, dict, dict]:
    with Image.open(source_path) as image:
        source_prompt = (
            f"Question: {question}\nRead the evidence in the image. {case['key_prompt']}\n"
            if protocol == "initial"
            else f"{case['key_prompt']}\n"
        )
        first = reader.read(
            image,
            source_prompt + 'Return only JSON with "key" (the literal label) and "note" (one short '
            "sentence describing what you identified and its evidence). Use null for an unreadable key.",
            112,
        )
    first_payload = first["parsed"] if isinstance(first["parsed"], dict) else {}
    note = first_payload.get("note") or "No note available."
    with Image.open(target_path) as image:
        if protocol == "initial":
            second = reader.read(
                image,
                f"Question: {question}\nFirst image note: {note}\n{case['value_prompt']}\n"
                'Return only JSON with "candidates" (a list of all readable key/value pairs in '
                'this target evidence, regardless of the note) and "selected" (the key/value pair '
                "you choose as the answer using the question and note). Each pair is "
                '{"key": "literal label", "value": number}. Use null for the selection if unsupported.',
                320,
            )
            parsed = second["parsed"]
            # The initial protocol accepted only a top-level object.
            result = evaluate(case, first_payload, parsed if isinstance(parsed, dict) else {})
            return first, {"shared": second}, result

        free = reader.read(
            image,
            f"Question: {question}\nFirst image note: {note}\n"
            f"Use the note and the new image to answer the question. Measurement guide: {case['value_prompt']}\n"
            'Return only the best answer as a JSON object {"key": "label used", "value": number}. '
            "If the evidence is insufficient, use null for the corresponding field.",
            320,
        )
        target = reader.read(
            image,
            case["value_prompt"] + "\nExtract records from this image. Return only a JSON array of "
            '{"key": "label", "value": number} objects, one per category. Include all readable '
            "values for the requested measurement and keep labels associated with their values.",
            320,
        )
    parsed = target["parsed"]
    if isinstance(parsed, dict):
        parsed = parsed.get("candidates", parsed)
    if isinstance(parsed, dict):
        parsed = [{"key": label, "value": value} for label, value in parsed.items()]
    result = evaluate(
        case,
        first_payload,
        {"candidates": parsed, "selected": free["parsed"]},
        year_keys=case.get("key_type") == "year",
    )
    return first, {"free": free, "bound": target}, result


class Reader:
    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, min_pixels=56 * 56, max_pixels=448 * 896, local_files_only=True
        )
        options = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
        if device == "cuda":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, device_map={"": device}, local_files_only=True, **options
        ).eval()
        self.vision_token = self.model.config.image_token_id

    def read(self, image: Image.Image, prompt: str, limit: int) -> dict:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[rendered], images=[image], return_tensors="pt").to(
            self.model.device
        )
        # Greedy decoding holds the same visual reads fixed for both decision rules.
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        seconds = time.perf_counter() - start
        prompt_tokens = inputs.input_ids.shape[-1]
        completion = output[0, prompt_tokens:]
        raw = self.processor.decode(completion, skip_special_tokens=True)
        return {
            "prompt": prompt,
            "raw": raw,
            "parsed": read_json(raw),
            "seconds": seconds,
            "prompt_tokens": prompt_tokens,
            "vision_tokens": int(inputs.input_ids.eq(self.vision_token).sum()),
            "completion_tokens": len(completion),
        }


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/mmlongbench-full.json"))
    parser.add_argument("--model", type=Path, default=Path("data/models/qwen2.5-vl-7b"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--view", nargs="+", choices=("crop", "page"), default=["crop", "page"])
    parser.add_argument("--protocol", choices=("initial", "focused"), default="focused")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    manifest = json.loads(args.manifest.read_text())
    requested = {case["id"] for case in cases}
    questions = {
        q["id"]: (args.manifest.parent / doc["pdf"], q)
        for doc in manifest["documents"]
        for q in doc["questions"]
        if q["id"] in requested
    }
    prepared = []
    for case in cases:
        pdf_path, question = questions[case["id"]]
        assert {case[role][0] for role in ("source", "target")} == set(
            question["evidence_pages_raw"]
        )
        with pymupdf.open(pdf_path) as pdf:
            for view in args.view:
                paths = [
                    args.output / "images" / view / f"{case['id']}-{role}.png"
                    for role in ("source", "target")
                ]
                for role, path in zip(("source", "target"), paths, strict=True):
                    render(pdf, case[role], view, path)
                prepared.append((case, question["question"].strip(), view, paths))
    if args.prepare_only:
        return

    print(f"Loading {args.model} on {args.device}", flush=True)
    reader = Reader(args.model, args.device)
    print("Visual reader ready", flush=True)
    output_rows = []
    with (args.output / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for case, question, view, (source_path, target_path) in prepared:
            first, followup, result = paired_reads(
                reader, case, question, source_path, target_path, args.protocol
            )
            costs = {}
            for method in ("free", "bound"):
                read = followup.get(method, followup.get("shared"))
                costs[method] = {
                    name: first[name] + read[name]
                    for name in ("prompt_tokens", "vision_tokens", "completion_tokens", "seconds")
                }
            row = {
                "id": case["id"],
                "view": view,
                "gold_key": case["gold_key"],
                "gold_value": case["gold_value"],
                "model_calls": 2,
                "source": {"image": str(source_path), **first},
                "target": {"image": str(target_path), **followup},
                "costs": costs,
                **result,
            }
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")
            trace.flush()
            output_rows.append(row)
            print(case["id"], view, result, flush=True)

    metadata = {
        "model": str(args.model),
        "device": args.device,
        "protocol": args.protocol,
        "weights": "NF4 with bfloat16 compute" if args.device == "cuda" else "bfloat16",
        "torch": version("torch"),
        "transformers": version("transformers"),
        "bitsandbytes": version("bitsandbytes") if args.device == "cuda" else None,
    }
    with (args.output / "results.json").open("w", encoding="utf-8") as stream:
        json.dump({"metadata": metadata, "rows": output_rows}, stream, ensure_ascii=False, indent=2)
    with (args.output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "view",
                "gold_key",
                "gold_value",
                "source_key",
                "source_key_correct",
                "free_key_correct",
                "free_value",
                "free_answer_correct",
                "bound_key_correct",
                "bound_value",
                "bound_answer_correct",
                "oracle_answer_correct",
                "alternate_correct",
                "free_prompt_tokens",
                "bound_prompt_tokens",
                "vision_tokens",
                "free_model_seconds",
                "bound_model_seconds",
            ]
        )
        for row in output_rows:
            writer.writerow(
                [
                    row["id"],
                    row["view"],
                    row["gold_key"],
                    row["gold_value"],
                    row["source_key"],
                    row["source_key_correct"],
                    row["free"]["key_correct"],
                    row["free"]["value"],
                    row["free"]["answer_correct"],
                    row["bound"]["key_correct"],
                    row["bound"]["value"],
                    row["bound"]["answer_correct"],
                    row["oracle"]["answer_correct"],
                    row["alternate"]["correct"],
                    row["costs"]["free"]["prompt_tokens"],
                    row["costs"]["bound"]["prompt_tokens"],
                    row["costs"]["bound"]["vision_tokens"],
                    row["costs"]["free"]["seconds"],
                    row["costs"]["bound"]["seconds"],
                ]
            )


if __name__ == "__main__":
    run()
