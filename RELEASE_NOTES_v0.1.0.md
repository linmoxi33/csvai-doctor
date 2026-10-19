## csvai 0.1.0

Optional AI assistance for [csvkit](https://github.com/wireservice/csvkit) 2.2.0 — behind a
40-line opt-in patch, with a deterministic fallback for every path.

### The gap this fills

csvkit's 14 commands all require you to already know your column names, indices, or the exact
substring to match. `csvstat` will tell you a column is `Text` with 412 unique values; it cannot tell
you those values are company names. That is a semantic gap, and it is the one place a model helps.

### Three capabilities, each verified

| Capability | Model does | Verification |
|---|---|---|
| `--ai "…"` on any command | Proposes `{command, args}` steps | Every flag validated against csvkit's **live argparse**; every column against the real header |
| `csvai classify` | Labels columns from a closed vocabulary | Conformance measured in Python against sampled values |
| `csvai extract` | Pulls fields out of prose | Each value must appear in its source row, or it is discarded |

The model never emits a shell string, never chooses a file, and never runs SQL outside a single
read-only `SELECT` from one fixed table.

### Example

```
$ csvcut --ai "每个城市的订单总金额，从高到低" orders.csv
城市,总金额
上海,6860.0
成都,2118.4
重庆,540.5
北京,335.0
杭州,199.0
```

`--ai-show-plan` prints the compiled command so you can drop the flag and keep using plain csvkit.

Classification shows you the rows that disagree with the label:

```
联系邮箱   email   0.90   80%  Email addresses, one malformed entry
                non-conforming: ['wangqiang_at_example.com']
```

### Fallbacks are the feature

With no key, no network, or a hostile model, the rule engine serves the request instead:

| Request | Offline plan |
|---|---|
| `只要 A 和 B` / `drop A` | `csvcut --columns` / `--not-columns` |
| `A 等于 x` | `csvgrep --regex ^\s*x\s*$` (anchored, so `US` ≠ `USA`) |
| `A 大于 100` | `csvsql … WHERE "A" > 100` (a regex cannot express `>`) |
| `按 A 统计 B 的平均值` | `csvsql … AVG("B") GROUP BY "A"` |

Anything outside those patterns **returns nothing** rather than a plausible wrong answer. Exit code
2, clean stdout, and a list of what does work.

### Verification

- **151** package tests
- **346** upstream csvkit tests still pass after the patch
- `--help` **byte-identical** for 6 commands when csvai is not installed (compared against fixtures
  from a pristine checkout, run from a temp cwd)
- **5/5** deliberate mutations caught (flag allow-list, SQL guard, refusal contract, provenance
  check, conformance measurement)
- Patch applies cleanly to a fresh csvkit 2.2.0 tarball, then the suite passes there too

### Honest limits

- Profiles 40 rows by default; a column that is clean early and dirty later will be misread.
- Plans are capped at 3 steps.
- Offline coverage is narrow by design: selection, filtering, sorting, aggregation.
- `csvstack`/`csvjoin` are not plannable (they need multiple files).
- Ships as a patch, not a fork — you can read the entire integration surface in one screen.

See [`DIAGNOSIS.md`](https://github.com/linmoxi33/csvai-doctor/blob/main/DIAGNOSIS.md) for the
investigation, including the nine bugs found by running it (a date column that became a phone
column, `大于 abc` silently degrading to substring matching, GROUP BY disappearing, a pipeline that
died on `reconfigure`, and more).
