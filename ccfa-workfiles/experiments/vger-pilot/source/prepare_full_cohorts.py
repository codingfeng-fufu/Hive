"""Prepare the two full-document QA development cohorts from local releases."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path("data")

MMLONGBENCH_MACRO = {
    289,
    293,
    294,
    295,
    296,
    702,
    703,
    798,
    368,
    371,
    378,
    379,
    380,
    210,
    212,
    214,
    216,
    515,
    516,
    517,
}
DOCBENCH_SUBSETS = {
    "0": {
        "0": "macro",
        "1": "micro",
        "2": "micro",
        "3": "micro",
        "4": "micro",
        "5": "micro",
        "6": "macro",
    },
    "103": {"0": "micro", "1": "micro", "2": "micro"},
    "141": {"0": "macro", "1": "macro"},
    "179": {
        "0": "macro",
        "1": "micro",
        "3": "micro",
        "4": "micro",
        "5": "micro",
        "6": "micro",
        "7": "micro",
    },
}


def prepare(source: str, cohort: str, output: str, layouts: str) -> None:
    manifest = json.loads((DATA / source).read_text())
    selected = (ROOT / cohort).read_text().splitlines()
    documents = {doc["id"]: doc for doc in manifest["documents"]}
    manifest["documents"] = [documents[doc_id] for doc_id in selected]
    manifest["dataset"] += "-full-document-pilot"
    for doc in manifest["documents"]:
        pdf = DATA / doc["pdf"]
        stem = pdf.stem
        layout = Path(layouts) / stem / "vlm" / f"{stem}_content_list.json"
        if not (DATA / layout).is_file():
            raise FileNotFoundError(DATA / layout)
        doc["layout"] = str(layout)
        for question in doc["questions"]:
            if source.startswith("docbench"):
                subset = DOCBENCH_SUBSETS[doc["id"]].get(question["id"], "unlabeled")
            elif not question["evidence_pages_raw"]:
                subset = "unlabeled"
            else:
                subset = "macro" if int(question["id"]) in MMLONGBENCH_MACRO else "micro"
            question["subset"] = subset
    (DATA / output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        manifest["dataset"], len(selected), sum(len(d["questions"]) for d in manifest["documents"])
    )


if __name__ == "__main__":
    prepare(
        "mmlongbench-full.json",
        "full-document-cohort.txt",
        "mmlongbench-full-document-pilot.json",
        "mineru-full",
    )
    prepare(
        "docbench-full.json",
        "docbench-cohort.txt",
        "docbench-full-document-pilot.json",
        "mineru-docbench",
    )
