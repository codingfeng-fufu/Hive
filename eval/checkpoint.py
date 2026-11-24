"""Persist completed documents so a detached benchmark can resume after interruption."""

import json
from contextlib import contextmanager
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def lock_output(path: Path):
    import fcntl

    with (path / ".benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another benchmark is writing to {path}") from error
        yield


class DocumentCheckpoints:
    def __init__(self, output: Path, metadata: dict, *, resume: bool) -> None:
        record = output / "run.json"
        if resume and record.exists():
            if json.loads(record.read_text()) != metadata:
                raise ValueError(
                    f"Run configuration changed for {output}. Choose a new output path."
                )
        else:
            write_json(record, metadata)
        self.directory = output / "checkpoints"
        self.directory.mkdir(exist_ok=True)

    def load(self, key: str, document: str, questions: list[str]) -> dict | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        checkpoint = json.loads(path.read_text())
        expected = [(document, question) for question in questions]
        actual = [(row["doc_id"], row["question_id"]) for row in checkpoint["rows"]]
        if checkpoint["index"]["doc_id"] != document or actual != expected:
            raise ValueError(f"Checkpoint case IDs changed at {path}")
        return checkpoint

    def save(self, key: str, index: dict, rows: list[dict]) -> None:
        write_json(self.directory / f"{key}.json", {"index": index, "rows": rows})
