# Your coding agent is a black box with a meter running

### Why the big agent vendors may never optimize for your costs — and how a different architecture turns one-shot magic into cheap, auditable pipelines.

> A companion to [call-transcripts-ai](https://github.com/ranfysvalle02/call-transcripts-ai) (auditable call summaries) and [blackbox-ai](https://github.com/ranfysvalle02/blackbox-ai) (a flight recorder for coding agents). Same idea, third domain: when a machine is expensive and opaque, stop trusting its story — change the architecture so every claim leaves a receipt. The working code for everything below is in [code-xplorer](https://github.com/ranfysvalle02/code-xplorer).

## 1. The one-shot economy

Here's how most coding agents work today: you type *"scan this repo and tell me about security"*, and the tool stuffs as much of your codebase as it can into a frontier model's context window, lets it "think," lets it call tools in a loop, and bills you for every token of that inner monologue.

When it works, it feels like magic. When it doesn't, you get three problems at once:

1. **You can't see where it failed.** The answer is wrong, but *which part* of the reasoning was wrong? The retrieval? The analysis? The writing? The whole run is one undifferentiated blob of tokens.
2. **You can't fix just the broken part.** Your only lever is "try again" — which means paying for the entire run again, including everything that was already correct.
3. **You paid frontier-model prices for the whole job**, even though 80% of it — finding files, grepping, counting, formatting — never needed a frontier model at all.

Multi-step reasoning inside a single giant prompt is exactly where LLMs are weakest and where failures are hardest to localize. We've all watched an agent burn five minutes and a pile of tokens confidently going down the wrong path because step two of its invisible plan was wrong and nothing could stop it.

## 2. Follow the incentives

Now ask the uncomfortable question: **who profits when your agent is inefficient?**

I want to be careful here — this is about structural incentives, not accusations of bad faith. But the structure is worth staring at:

- **Token resale and partnerships.** Many agent companies sit between you and a model provider they have a commercial relationship with. Cursor's pricing, for instance, is deeply intertwined with the frontier labs it builds on. When a vendor's margins and partnerships are tied to the volume of frontier-model tokens flowing through their pipes, "make each task use dramatically fewer frontier tokens" is, at best, a complicated priority. Efficiency for you is revenue compression for them.
- **Architecture follows the business model.** If you monetize usage, the one-big-agent architecture is *great*: every retry, every bloated context window, every unnecessary re-read of your repo is billable. An architecture that routed 80% of the work to deterministic code and a free local 8B model would be better engineering — and worse quarterly numbers.
- **Opacity is a moat.** If you could see exactly which step consumed which tokens and produced which output, you could compare vendors, swap models per step, or run steps locally. A single opaque "agent run" resists exactly that kind of unbundling.

To be fair, incentives aren't the only force at work. One giant loop is also the *easiest architecture to ship* for arbitrary tasks: a general-purpose agent with a handful of tools generalizes to anything, while a pipeline of typed steps has to be designed, planned, and validated per class of ask. Horizontal products favor the loop for honest engineering reasons. But notice that this convenience and the incentive gradient point the *same direction* — which is exactly why you shouldn't expect the market to drift toward efficiency and transparency on its own.

None of this requires anyone to be a villain. It just means you shouldn't expect the cost problem and the black-box problem to be solved *for* you by companies whose architecture and partnerships quietly benefit from both. If you want efficiency and auditability, they have to be architectural properties you choose.

## 3. The alternative: break the ask into a pipeline

The fix isn't a better prompt or a bigger model. It's the same move that worked for auditable call summaries: **stop asking one model to do everything in one breath, and box the model in.**

In [code-xplorer](https://github.com/ranfysvalle02/code-xplorer), an ask like *"scan this repo and generate a report on security"* doesn't go to a mega-prompt. It goes to a **planner** that returns a plain JSON pipeline — a list of small, typed steps:

```json
{
  "name": "Security Scan Report",
  "steps": [
    { "id": "search_vulns",  "type": "search",    "params": { "query": "password secret sql injection subprocess shell" } },
    { "id": "extract_vulns", "type": "extract",   "inputs": ["search_vulns"], "model": "llama3.1:latest",
      "instructions": "Extract JSON facts: {claim, severity, file, evidence (verbatim quote)}. Only facts grounded in the inputs." },
    { "id": "rank",          "type": "transform", "inputs": ["extract_vulns"], "params": { "op": "count_by", "field": "severity" } },
    { "id": "report",        "type": "report",    "inputs": ["extract_vulns", "rank"], "model": "llama3.1:latest",
      "instructions": "Write a Markdown report citing the file paths and quotes provided. Do not invent anything." }
  ]
}
```

Five design decisions carry all the weight:

**Deterministic steps produce the objective artifacts.** `git`, `search`, and `transform` steps are plain code: real commits, real diffs, real hybrid-search hits, real counts. They cost zero tokens, they're byte-identical on re-run, and they form the ground truth that every LLM step must build on. The model never gets to invent the evidence — it only gets to interpret it.

**LLM steps extract, they don't free-associate.** An `extract` step returns small, typed JSON facts, and every fact must carry provenance — a file path, a commit sha, a verbatim quote. The final `report` step runs on a short leash: it may only use the structured facts handed to it. This is the "extract, don't summarize" rule from the call-transcripts work, applied to code.

**Every step is a decision record.** The runner logs each step's model, exact prompt, resolved inputs, raw output, parsed output, timing, and status to a persistent audit trail. When a run is wrong, you don't interrogate the model ("why did you say that?" just buys you a second magic trick). You open the trail, find the step whose *output* diverged from its *input*, fix that one step, and re-run. Saved runs load back into the editor — the trail is editable and re-runnable, not a post-mortem.

**You pick the model per step.** The planner can run on a fast local model. Extraction — a dumb, mechanical job by design — runs happily on a free 8B model. Only the final report *might* deserve a bigger model, and it's reading a page of structured facts, not your whole repo. The expensive model is invited to the last meeting, not cc'd on every email.

**The decomposition itself is contestable.** Because the plan is just JSON from an AI, you can ask for *different decompositions of the same ask* — a thorough plan, a minimal plan, a git-forensics plan that leads with commits and diffs, an evidence-first plan that demands quotes and deterministic counts. You compare pipelines, not vibes. Then you edit any step before spending a cent.

## 4. The receipts

I ran this on a deliberately insecure demo repo — hardcoded admin password, MD5 hashing, SQL injection via string formatting, `subprocess` with `shell=True`, a password reset that prints `"changeme123"`.

The AI-planned, six-step variant of this pipeline (the planner also led with a `git log` step and paired it with its own `extract`, so commit history got the same evidence treatment as the code) ran **entirely on local models (Llama 3.1 8B — free, on a laptop)** in about **12 seconds of LLM time**, and the final report contained findings like:

> **Insecure Backup Function** — `db.py` — Evidence: `subprocess.call("tar -czf backup.tgz " + path, shell=True)`
>
> **Insecure Password Hashing** — `auth.py` — Evidence: `return hashlib.md5(password.encode()).hexdigest()`
>
> **Hardcoded Admin Password** — `auth.py` — Evidence: `if username == "admin" and password == ADMIN_PASSWORD:`

Every finding cites a file and a verbatim quote that either exists in the code or doesn't. The git-derived claims cite real commit shas. And the full trail — which model ran which step, with which prompt, producing which output, in how many milliseconds — is saved and reloadable.

An 8B model did not get smarter than a frontier model. The *architecture* made its job small enough that an 8B model was sufficient, and made its mistakes cheap to find. That's the whole trick. A wrong label on a grounded fact costs a human three seconds to spot and one step to re-run — not an entire frontier-priced session.

Is the demo repo a toy? Completely — and deliberately, for the same reason the call-transcripts project uses synthetic calls: with planted ground truth you can score recall exactly instead of eyeballing vibes. A toy repo proves the *mechanism* (decomposition, grounding, per-step models, the audit trail), not real-world recall on a million-line monorepo — that's a claim only your own repo can settle, which is the point of shipping the tool instead of a benchmark table. On bigger codebases the deterministic steps are the ones doing the scaling work (hybrid search and git don't care how large the repo is; they only ever hand the model a bounded page of evidence), and when recall does fall short, the trail shows you *which search query* missed — so you fix a query, not a prompt. For heavier lifting, swap `voyage-code-3` embeddings and a frontier model into exactly the steps that need them — the point was never "local only," it's *your choice, per step*.

## 5. What this does *not* solve

Honesty section, same as always:

- **Grounding is not correctness.** A verbatim quote can still carry a wrong label or a wrong severity. The mechanical checks catch fabrication, not misclassification. Humans stay in the loop — the difference is the loop is now *possible*.
- **Planning is still an LLM call.** A weak planner produces a weak decomposition. That's why the plan is JSON you can read, edit, and re-generate with a different approach or model — the planner proposes, you dispose.
- **This covers the read-only half of the coding-agent loop.** The step types here (`git`, `search`, `extract`, `transform`, `report`) do comprehension and audit — security sweeps, architecture reviews, change forensics — not multi-file edits or fix-the-failing-test iteration. But nothing about the architecture stops there: a `patch` step that emits a diff (an objective artifact you can read before applying) and a `test` step whose pass/fail output is deterministic ground truth fit the same mold. The compiler and the test suite are the ultimate verbatim-quote checks — a patch either applies and passes or it doesn't. The write half is harder to box in, which is precisely why it deserves the box.
- **Not everything decomposes cleanly.** Tight, genuinely interdependent reasoning sometimes really does want one big context. Pipelines shine for the broad, auditable asks — security sweeps, architecture overviews, change forensics, "make me a report" — which, not coincidentally, are the asks where one-shot agents burn the most tokens for the least traceable output.
- **This is a teaching implementation.** Step types are intentionally few (`git`, `search`, `extract`, `transform`, `report`) so the system itself stays auditable.

## 6. The point

The big-vendor pitch is: *trust the agent, buy more tokens.* Both halves of that pitch get easier to sell when the product is a black box — and the companies selling it have, at minimum, no structural incentive to open the box or shrink the bill.

You don't have to wait for them. The architecture in this repo is not clever. It's the oldest idea in engineering — decompose, make interfaces explicit, log everything — applied to AI:

- **Determinism upstream, narrative last.** Code produces the facts; the model gets the last word, on a leash.
- **Small steps, small models.** Match the model to the step, not the invoice to the vendor.
- **Decision records, not confessions.** Never ask the model why. Read the trail.

Run `python app.py`, point it at a repo, hit **Pipelines**, and type an ask. Watch it get broken into steps you can see, edit, price, and audit. The models will still make mistakes — but every mistake becomes small, local, and traceable.

No more black box.
