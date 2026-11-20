"""Upload a local file as an Orchestrator attachment and write input.json.

    uv run python upload.py reviews.csv "Classify each review"

Uses the folder of the current ``uipath auth`` session (``UIPATH_FOLDER_PATH``
when set), the same one the agent reads attachments from.
"""

import json
import mimetypes
import sys
from pathlib import Path

from dotenv import load_dotenv
from uipath.platform import UiPath


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <file> <task>")
    load_dotenv()
    path = Path(sys.argv[1])
    key = UiPath().attachments.upload(name=path.name, source_path=str(path))
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    document = {"ID": str(key), "FullName": path.name, "MimeType": mime}
    Path("input.json").write_text(
        json.dumps({"reviews": document, "task": sys.argv[2]}, indent=2) + "\n"
    )
    print(f"Uploaded {path.name} as attachment {key}; wrote input.json")


if __name__ == "__main__":
    main()
