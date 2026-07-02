#!/usr/bin/env bash
# Fetch a spread of Gset Max-Cut benchmark instances into data/gset/.
#
# The Gset instances are a standard benchmark set for Max-Cut (Helmberg &
# Rendl; hosted by Y. Ye at Stanford). Each file is the plain
# "n m / u v w" format the parser in include/anneal/problems/maxcut.hpp
# reads. Files are gitignored (see data/.gitignore); this script downloads
# them on demand.
#
# Usage: ./data/download_gset.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)/gset"
mkdir -p "$DIR"
BASE="https://web.stanford.edu/~yyye/yyye/Gset"

# A spread of sizes: G1 (800 nodes, dense), G22 (2000), G39 (2000, signed
# weights), G55 (5000).
for g in G1 G22 G39 G55; do
  if [[ -s "$DIR/$g" ]]; then
    echo "have $g"
  else
    echo "downloading $g ..."
    curl -fsSL "$BASE/$g" -o "$DIR/$g"
  fi
done
echo "Gset instances in $DIR"
