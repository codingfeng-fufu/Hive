"""Inspect or restore the external files required by the frozen VGER batch."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDFS = Path("data/upstream/mmlongbench-doc/documents")
MODELS = Path("data/models")


def load_manifest():
    return json.loads(Path(__file__).with_name("reference-assets.json").read_text())


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def model_file_matches(path, size):
    if not path.is_file():
        return False
    actual_size = path.stat().st_size
    return actual_size == size if path.suffix == ".safetensors" else actual_size > 0


def problems(manifest):
    for name, info in manifest["dataset"]["pdfs"].items():
        path = ROOT / PDFS / name
        if not path.is_file():
            yield f"Missing {PDFS / name}"
        elif path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            yield f"PDF does not match the frozen copy: {PDFS / name}"
    for folder, model in manifest["models"].items():
        for name, size in model["files"].items():
            relative = MODELS / folder / name
            path = ROOT / relative
            if not path.is_file():
                yield f"Missing {relative}"
            elif not model_file_matches(path, size):
                yield f"Unexpected file size: {relative}"


def download(manifest):
    from huggingface_hub import hf_hub_download, snapshot_download

    dataset = manifest["dataset"]
    for name, info in dataset["pdfs"].items():
        path = ROOT / PDFS / name
        if path.is_file() and digest(path) == info["sha256"]:
            continue
        hf_hub_download(
            repo_id=dataset["repository"],
            repo_type="dataset",
            revision=dataset["revision"],
            filename=f"documents/{name}",
            local_dir=ROOT / PDFS.parent,
            force_download=path.is_file(),
        )
    for folder, model in manifest["models"].items():
        target = ROOT / MODELS / folder
        files = model["files"]
        missing = [name for name in files if not (target / name).is_file()]
        if missing:
            snapshot_download(
                repo_id=model["repository"],
                local_dir=target,
                allow_patterns=missing,
            )
        for name, size in files.items():
            path = target / name
            if path.is_file() and not model_file_matches(path, size):
                hf_hub_download(
                    repo_id=model["repository"],
                    filename=name,
                    local_dir=target,
                    force_download=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "files", "download"))
    args = parser.parse_args()
    manifest = load_manifest()
    if args.action == "files":
        for name in manifest["dataset"]["pdfs"]:
            print(PDFS / name)
        for folder, model in manifest["models"].items():
            for name in model["files"]:
                print(MODELS / folder / name)
        return
    if args.action == "download":
        download(manifest)
    issues = list(problems(manifest))
    if issues:
        parser.exit(1, "\n".join(issues) + "\n")
    print("Reference PDFs and model files are available.")


if __name__ == "__main__":
    main()
