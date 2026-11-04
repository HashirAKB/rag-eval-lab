#!/usr/bin/env bash
# Re-create data/corpus/ from the pinned commit of the MCP docs repository.
# The snapshot is already committed; run this only to verify or refresh it.
set -euo pipefail

REPO_URL="https://github.com/modelcontextprotocol/modelcontextprotocol.git"
COMMIT="3c4d91b7e832d9b74338d56eed03403ccc662123"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/corpus"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone --quiet --filter=blob:none "$REPO_URL" "$WORK/repo"
git -C "$WORK/repo" checkout --quiet "$COMMIT"

rm -rf "$DEST/specification" "$DEST/learn" "$DEST/security"
mkdir -p "$DEST"
SRC="$WORK/repo/docs"

# 1. The 2026-07-28 specification, minus the auto-generated schema reference.
(cd "$SRC/specification/2026-07-28" && find . -name '*.mdx' ! -name 'schema.mdx' -print0 |
  while IFS= read -r -d '' f; do mkdir -p "$DEST/specification/$(dirname "$f")"; cp "$f" "$DEST/specification/$f"; done)
# 2. Conceptual "learn" pages and security tutorials for the same version.
mkdir -p "$DEST/learn" "$DEST/security"
cp "$SRC/docs/2026-07-28/learn/"*.mdx "$DEST/learn/"
cp "$SRC/docs/2026-07-28/tutorials/security/"*.mdx "$DEST/security/"

echo "Corpus written to $DEST from commit $COMMIT"
