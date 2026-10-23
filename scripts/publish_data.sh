#!/usr/bin/env bash
# publish_data.sh
# Copies the generated CSV outputs onto the `data` branch and pushes it. The code lives on `main`; the
# generated data lives on `data`, so the main history stays clean and Power BI can read stable raw URLs:
#   https://raw.githubusercontent.com/<owner>/<repo>/data/data/predictions/stock_predictions.csv
#
# Safe to run when nothing changed (it then does nothing) and safe to run after a failed step (it only
# publishes files that exist). Pushes are retried after a rebase in case another writer got in first.

set -euo pipefail

DATA_BRANCH="${DATA_BRANCH:-data}"
REMOTE="${REMOTE:-origin}"
FILES=(
  "data/raw/stock_prices.csv"
  "data/processed/features.csv"
  "data/predictions/stock_predictions.csv"
  "data/predictions/pipeline_logs.csv"
  "data/predictions/live_scores.csv"
  "data/predictions/live_summary.csv"
)

ROOT="$(pwd)"
WT="$(mktemp -d)"
trap 'git worktree remove --force "$WT" >/dev/null 2>&1 || true; rm -rf "$WT"' EXIT

git config user.name  >/dev/null 2>&1 || git config user.name  "github-actions[bot]"
git config user.email >/dev/null 2>&1 || git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

if git ls-remote --exit-code --heads "$REMOTE" "$DATA_BRANCH" >/dev/null 2>&1; then
  git fetch --no-tags "$REMOTE" "+$DATA_BRANCH:refs/remotes/$REMOTE/$DATA_BRANCH"
  git worktree add --force -B "$DATA_BRANCH" "$WT" "$REMOTE/$DATA_BRANCH" >/dev/null
else
  echo "publish_data: creating new orphan branch '$DATA_BRANCH'."
  git worktree add --force --detach "$WT" HEAD >/dev/null
  (cd "$WT" && git checkout --orphan "$DATA_BRANCH" >/dev/null 2>&1 && git rm -rf . >/dev/null 2>&1 || true)
fi

# Safety net for the append-only files: the new version must still contain every line already published.
# (Protects the track record if a run started from a stale or empty copy.)
for f in "data/predictions/stock_predictions.csv" "data/predictions/pipeline_logs.csv"; do
  if [ -f "$WT/$f" ] && [ -f "$ROOT/$f" ]; then
    lost="$( { grep -vxFf "$ROOT/$f" "$WT/$f" || true; } | wc -l | tr -d ' ')"
    if [ "$lost" != "0" ]; then
      echo "publish_data: REFUSING to publish. $f would drop $lost previously published line(s)." >&2
      exit 1
    fi
  fi
done

copied=0
for f in "${FILES[@]}"; do
  if [ -f "$ROOT/$f" ]; then
    mkdir -p "$WT/$(dirname "$f")"
    cp "$ROOT/$f" "$WT/$f"
    copied=$((copied + 1))
  else
    echo "publish_data: $f not found, skipped."
  fi
done

if [ "$copied" -eq 0 ]; then
  echo "publish_data: nothing to publish."
  exit 0
fi

cd "$WT"
cat > README.md <<'EOT'
Generated data branch. Written only by the daily workflow; do not edit by hand.
Code and documentation are on the main branch.
EOT

git add -A
if git diff --cached --quiet; then
  echo "publish_data: no changes to publish."
  exit 0
fi

git commit -q -m "data: update $(date -u +%Y-%m-%dT%H:%MZ)"

for attempt in 1 2 3; do
  if git push "$REMOTE" "HEAD:refs/heads/$DATA_BRANCH"; then
    echo "publish_data: pushed to '$DATA_BRANCH'."
    exit 0
  fi
  echo "publish_data: push failed (attempt $attempt), rebasing and retrying."
  git fetch --no-tags "$REMOTE" "$DATA_BRANCH"
  git rebase "$REMOTE/$DATA_BRANCH" || { git rebase --abort || true; exit 1; }
  sleep $((attempt * 5))
done

echo "publish_data: giving up after 3 attempts." >&2
exit 1
