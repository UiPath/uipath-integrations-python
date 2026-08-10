#!/usr/bin/env bash
#
# Pack a sample against the local uipath-claude-sdk instead of a published release.
#
#   ./scripts/pack_sample.sh samples/quickstart-agent
#
# The samples resolve uipath-claude-sdk from this checkout with an editable path
# dependency, which is right for `uipath run` and wrong for `uipath pack`: the
# path points outside the project directory, so the packer never sees it and the
# executor cannot install it. This builds a wheel into the sample's gitignored
# wheels/ directory, points the sample at that wheel for the duration of the
# pack, and restores every file it touched.
#
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <sample-directory>" >&2
    exit 2
fi

pkg_root=$(cd "$(dirname "$0")/.." && pwd)
sample=$(cd "$1" && pwd)

for required in pyproject.toml uipath.json entry-points.json; do
    if [ ! -f "$sample/$required" ]; then
        echo "$sample has no $required. Run 'uipath init' there first." >&2
        exit 1
    fi
done

uv build --wheel --project "$pkg_root" --out-dir "$sample/wheels"
wheel=$(cd "$sample" && ls -t wheels/uipath_claude_sdk-*.whl | head -1)

backup=$(mktemp -d)
restore() {
    for f in pyproject.toml uipath.json uv.lock; do
        [ -f "$backup/$f" ] && cp "$backup/$f" "$sample/$f"
    done
    rm -rf "$backup"
    # Packing installs the wheel over the editable checkout, so without this the
    # sample would keep running a frozen copy of the source until the next sync.
    (cd "$sample" && uv sync --quiet) || true
}
trap restore EXIT

for f in pyproject.toml uipath.json uv.lock; do
    [ -f "$sample/$f" ] && cp "$sample/$f" "$backup/$f"
done

cd "$sample"

python3 - "$wheel" <<'PY'
import json
import re
import sys

wheel = sys.argv[1]

source = open("pyproject.toml").read()
pattern = re.compile(r'^uipath-claude-sdk\s*=\s*\{[^}]*\}$', re.MULTILINE)
if not pattern.search(source):
    sys.exit(
        "pyproject.toml has no [tool.uv.sources] entry for uipath-claude-sdk, "
        "so there is nothing to redirect at the local wheel."
    )
open("pyproject.toml", "w").write(
    pattern.sub(f'uipath-claude-sdk = {{ path = "{wheel}" }}', source)
)

config = json.load(open("uipath.json"))
options = config.setdefault("packOptions", {})
extensions = options.setdefault("fileExtensionsIncluded", [])
if ".whl" not in extensions:
    extensions.append(".whl")
json.dump(config, open("uipath.json", "w"), indent=2)
PY

uv lock
uv run uipath pack

echo
echo "Packed with $wheel vendored into the package."
echo "The sample's own files are back as they were."
