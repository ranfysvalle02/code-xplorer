# demo/ — a tiny, deliberately insecure app for validating code-xplorer

This is **planted ground truth**: a two-file "app" with known, intentional
security problems. Index it, run the **Security review** pipeline (or plan your
own with AI), and score the results against the checklist below — exact recall,
not vibes.

> ⚠️ Every "secret" and vulnerability in here is fake and intentional.
> Do not copy any of this into real code.

## Validate in 60 seconds

```bash
# Option A: no git history (git pipeline steps will be hidden)
python app.py            # then scan:  demo/

# Option B: with git history (enables git log/diff/churn steps + git-forensic planning)
./demo/setup.sh          # creates /tmp/code-xplorer-demo with 2 commits
python app.py            # then scan:  /tmp/code-xplorer-demo
```

Once indexed, open **Pipelines** (top-right), pick the **Security review**
template (or type *"scan this repo and generate a security report"* and let the
AI plan it), and hit **Run**.

## Ground truth: what a good run must find

| # | File      | Finding                                      | Evidence the report should quote                                   |
|---|-----------|----------------------------------------------|--------------------------------------------------------------------|
| 1 | `auth.py` | Hardcoded admin password                     | `ADMIN_PASSWORD = "hunter2"`                                        |
| 2 | `auth.py` | Weak password hashing (MD5)                  | `hashlib.md5(password.encode()).hexdigest()`                        |
| 3 | `auth.py` | Password reset uses a fixed default password | `new_password = "changeme123"`                                      |
| 4 | `db.py`   | SQL injection via string formatting          | `"SELECT * FROM users WHERE id = '%s'" % user_id`                   |
| 5 | `db.py`   | Shell injection (`shell=True` + concat)      | `subprocess.call("tar -czf backup.tgz " + path, shell=True)`        |

Scoring: **recall** = findings reported out of 5. Then audit the trail — every
finding must cite a file and a verbatim quote (check the quote actually exists
in the file). A finding without evidence doesn't count; that's the whole point.

With Option B, git-grounded steps should also cite the two real commit shas
(`initial insecure app`, `add password reset (insecure)`).
