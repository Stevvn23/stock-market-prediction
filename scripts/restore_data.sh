#!/usr/bin/env bash
# restore_data.sh
# Run at the start of a workflow job. Restores the append-only forecast files from the `data` branch so that
# predict.py extends the real history instead of starting an empty one.
#
# Raw prices and features are NOT restored: fetch_data.py downloads the full history again every run.
# If the data branch does not exist yet (very first run) there is nothing to restore, which is fine.

set -euo pipefail

DATA_BRANCH="${DATA_BRANCH:-data}"
REMOTE="${REMOTE:-origin}"
FILES=(
  "data/predictions/stock_predictions.csv"
  "data/predictions/pipeline_logs.csv"
)

mkdir -p data/predictions

if ! git ls-remote --exit-code --heads "$REMOTE" "$DATA_BRANCH" >/dev/null 2>&1; then
  echo "restore_data: branch '$DATA_BRANCH' does not exist yet; starting with empty prediction files."
  exit 0
fi

git fetch --no-tags --depth=1 "$REMOTE" "$DATA_BRANCH"

for f in "${FILES[@]}"; do
  if git cat-file -e "FETCH_HEAD:$f" 2>/dev/null; then
    git show "FETCH_HEAD:$f" > "$f"
    echo "restore_data: restored $f ($(wc -l < "$f") lines)"
  else
    echo "restore_data: $f not on '$DATA_BRANCH' yet."
  fi
done
