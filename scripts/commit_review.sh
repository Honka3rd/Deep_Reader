#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/commit_review.sh --message-file <file> [--dry-run]

Create a local review commit from already staged changes.
Stage and review the task's files explicitly before running this command.
--dry-run checks and lists the staged scope without committing or running hooks.
Normal commits retain Git hooks and signing configuration. No automatic staging,
amend, push, test execution or external approval is performed.
EOF
}

fail() { printf '%s\n' "$*" >&2; exit 1; }

message_file=''
dry_run=false
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --message-file)
      [[ "$#" -ge 2 && -n "$2" && -z "$message_file" ]] || {
        usage >&2; exit 2;
      }
      message_file="$2"
      shift 2
      ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

[[ -n "$message_file" ]] || { usage >&2; exit 2; }
[[ -f "$message_file" && -r "$message_file" && -s "$message_file" ]] ||
  fail 'Commit message must be a readable, nonempty file.'
# Resolve relative to the caller before changing into the script's repository.
message_file="$(cd "$(dirname "$message_file")" && pwd)/$(basename "$message_file")"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git symbolic-ref --quiet --short HEAD)" ||
  fail 'Detached HEAD: switch to the intended task branch first.'
git rev-parse --verify HEAD >/dev/null || fail 'An existing base commit is required.'
for state in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply sequencer; do
  [[ ! -e "$(git rev-parse --git-path "$state")" ]] ||
    fail "Unfinished Git operation ($state); resolve it before a review commit."
done
[[ -z "$(git ls-files --unmerged)" ]] || fail 'Resolve index conflicts before committing.'
if git diff --cached --quiet --exit-code; then
  fail 'No staged changes; stage only the intended task files first.'
else
  diff_status=$?
  [[ "$diff_status" -eq 1 ]] || exit "$diff_status"
fi
git diff --cached --check
printf 'Repository: %s\nBranch: %s\nStaged scope:\n' "$repo_root" "$branch"
git diff --cached --stat

if [[ "$dry_run" == true ]]; then
  printf 'Preview only; HEAD and staged changes are unchanged.\n'
  exit 0
fi

git commit --file "$message_file"
printf 'Local review commit: %s\n' "$(git rev-parse HEAD)"
printf 'Return this SHA for external review; no push or deployment was performed.\n'
