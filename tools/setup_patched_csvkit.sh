#!/usr/bin/env bash
# Recreate the patched csvkit checkout used by the test suite.
#
# The tests import csvkit from _csvkit_patched/ rather than site-packages so that
# a developer's global install is never modified. Regenerating it from a pinned
# tarball also means the patch is verified against clean upstream every time.
set -euo pipefail

CSVKIT_VERSION=2.2.0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/_csvkit_patched"

if [ -d "$TARGET" ]; then
    echo "$TARGET already exists; remove it to rebuild"
    exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "downloading csvkit $CSVKIT_VERSION..."
curl -sL "https://codeload.github.com/wireservice/csvkit/tar.gz/refs/tags/$CSVKIT_VERSION" -o "$TMP/csvkit.tar.gz"
tar xzf "$TMP/csvkit.tar.gz" -C "$TMP"
mv "$TMP/csvkit-$CSVKIT_VERSION" "$TARGET"

# The example corpus is 39 MB of real-world fixtures the suite does not need.
rm -rf "$TARGET/examples/realdata"

echo "applying patch..."
cd "$TARGET"
patch -p1 < "$ROOT/patches/0001-add-optional-ai-hook.patch"

echo "done: $TARGET"
