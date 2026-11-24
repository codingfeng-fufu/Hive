import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.document import Document, Region
from src.retriever import Evidence, Retrieval


class CommandBaseline:
    def __init__(
        self,
        command: list[str],
        working_dir: Path,
        timeout: float = 1800,
        *,
        persistent: bool = False,
        server: dict | None = None,
    ) -> None:
        if not command:
            raise ValueError("Baseline command is empty")
        self.command = command
        self.working_dir = working_dir.resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.server = server or {}
        self.persistent = persistent
        self.process = None
        self.executor = None
        self.log = None

    def _call(self, request: dict) -> dict:
        request["server"] = self.server
        if self.persistent:
            reply = self._call_worker(request)
        else:
            reply = self._call_once(request)
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return reply

    def _call_once(self, request: dict) -> dict:
        result = subprocess.run(
            self.command,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout,
        )
        if result.returncode:
            raise RuntimeError(
                f"Baseline command exited {result.returncode}: {result.stderr[-2000:]}"
            )
        return json.loads(result.stdout)

    def _call_worker(self, request: dict) -> dict:
        if self.process is None:
            self.log = (self.working_dir / "worker.log").open("a")
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
            )
            self.executor = ThreadPoolExecutor(max_workers=1)
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            response = self.executor.submit(self.process.stdout.readline).result(self.timeout)
            if not response:
                raise RuntimeError("Baseline worker exited without a response")
            return json.loads(response)
        except (TimeoutError, BrokenPipeError, RuntimeError, json.JSONDecodeError):
            self.close()
            raise

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.executor.shutdown()
        self.log.close()
        self.process = self.executor = self.log = None

    def index(self, document: Document) -> object:
        return self._call(
            {
                "operation": "index",
                "workdir": str(self.working_dir),
                "document": {
                    "id": document.id,
                    "pdf": str(document.source_pdf.resolve()) if document.source_pdf else None,
                    "pages": {
                        str(page): str(path.resolve()) for page, path in document.pages.items()
                    },
                },
            }
        )["index"]

    def retrieve(
        self, index: object, question: str, *, query_image: Path | None = None
    ) -> Retrieval:
        reply = self._call(
            {
                "operation": "retrieve",
                "index": index,
                "question": question,
                "query_image": str(query_image.resolve()) if query_image else None,
                "workdir": str(self.working_dir),
            }
        )
        evidence = []
        for original in reply["evidence"]:
            item = dict(original)
            if raw := item.get("region"):
                item["region"] = Region(int(raw["page"]), tuple(raw["bbox"]))
            if raw := item.get("image_path"):
                path = (self.working_dir / raw).resolve()
                if not path.is_relative_to(self.working_dir):
                    raise ValueError("Baseline image_path must be inside its workdir")
                item["image_path"] = path
            evidence.append(Evidence(**item))
        return Retrieval(evidence, reply.get("trace", {}))
