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
# Raw vendor prices (data/raw/stock_prices.csv) are deliberately NOT published: only data this project derives
# (features, forecasts, logs, scores) goes on the public data branch.
FILES=(
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

# Safety net for the append-only files: nothing already published may disappear.
#  - pipeline_logs.csv is only ever appended to, so every published line must still be there.
#  - stock_predictions.csv is compared by KEY (ticker, forecast_date, target_type, model = columns 1,2,3,6), so a
#    harmless difference in how a number is printed can never block a run, but a lost forecast always does.
#    If the content of an existing row changed, that is reported as a warning.
keys() { cut -d, -f1,2,3,6 "$1" | sort -u; }

pf="data/predictions/stock_predictions.csv"
if [ -f "$WT/$pf" ] && [ -f "$ROOT/$pf" ]; then
  lost="$( { comm -23 <(keys "$WT/$pf") <(keys "$ROOT/$pf") || true; } | wc -l | tr -d ' ')"
  if [ "$lost" != "0" ]; then
    echo "publish_data: REFUSING to publish. $pf would drop $lost previously published forecast key(s)." >&2
    exit 1
  fi
  changed="$( { grep -vxFf "$ROOT/$pf" "$WT/$pf" || true; } | wc -l | tr -d ' ')"
  if [ "$changed" != "0" ]; then
    echo "publish_data: WARNING: $changed previously published forecast line(s) differ in content from the new file."
  fi
fi

lf="data/predictions/pipeline_logs.csv"
if [ -f "$WT/$lf" ] && [ -f "$ROOT/$lf" ]; then
  lost="$( { grep -vxFf "$ROOT/$lf" "$WT/$lf" || true; } | wc -l | tr -d ' ')"
  if [ "$lost" != "0" ]; then
    echo "publish_data: REFUSING to publish. $lf would drop $lost previously published line(s)." >&2
    exit 1
  fi
fi

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

# The raw price file must never live on the data branch (also removes it if an earlier run published it).
git rm -q -f --ignore-unmatch data/raw/stock_prices.csv
git add -A

# One-time clean-up: replace the whole branch history with a single commit of the current content.
# Used to erase the earlier commits that contained raw prices. The guards above have already passed,
# so published forecasts are carried over unchanged.
if [ "${RESET_DATA_HISTORY:-0}" = "1" ]; then
  echo "publish_data: RESET_DATA_HISTORY=1, replacing the history of '$DATA_BRANCH' with one commit."
  git checkout -q --orphan "${DATA_BRANCH}-squashed"
  git commit -q -m "data: history reset $(date -u +%Y-%m-%dT%H:%MZ)"
  git push --force "$REMOTE" "HEAD:refs/heads/$DATA_BRANCH"
  echo "publish_data: history reset and pushed."
  exit 0
fi

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
