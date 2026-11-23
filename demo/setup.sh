#!/usr/bin/env bash
# Creates a scannable copy of the demo app WITH git history, so the git
# pipeline steps (log / diff / churn) and git-forensic planning have real
# commits to anchor on. Usage: ./demo/setup.sh [dest]   (default /tmp/code-xplorer-demo)
set -euo pipefail

DEST="${1:-/tmp/code-xplorer-demo}"
SRC="$(cd "$(dirname "$0")" && pwd)"
GIT=(git -c user.name=demo -c user.email=demo@example.com)

rm -rf "$DEST"
mkdir -p "$DEST"
cp "$SRC/auth.py" "$SRC/db.py" "$DEST/"
cd "$DEST"

git init -q

# Commit 1: the app without the password-reset function.
python3 - <<'EOF'
src = open('auth.py').read()
open('auth.py', 'w').write(src.split('# -- added in a later commit --')[0].rstrip() + '\n')
EOF
git add auth.py db.py
"${GIT[@]}" commit -qm "initial insecure app"

# Commit 2: the insecure password reset lands.
cp "$SRC/auth.py" auth.py
git add auth.py
"${GIT[@]}" commit -qm "add password reset (insecure)"

echo "Demo repo ready at: $DEST"
git --no-pager log --oneline
echo
echo "Next: python app.py  ->  scan '$DEST'  ->  open Pipelines  ->  run 'Security review'."
echo "Score the report against demo/README.md (5 planted findings)."
