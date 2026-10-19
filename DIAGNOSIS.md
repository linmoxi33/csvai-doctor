# DIAGNOSIS — what was actually wrong, and what I measured

This file records the investigation behind the design. It exists because "csvkit needs AI" is a
claim that should be backed by evidence, not vibes, and because several of the bugs found along the
way are more instructive than the finished code.

Environment: Windows 11, Python 3.13.14, csvkit 2.2.0, agate 1.14.2, SQLAlchemy 2.1.5.

---

## 1. Is csvkit really missing anything?

csvkit 2.2.0: 6,416 stars, 14 commands, last push 2026-09-21, 49 open issues. Actively maintained
and genuinely good at what it does.

I read all the utilities to find where the friction actually is. It is not in the parsing — `agate`
handles types, encodings, sniffs delimiters, and does 80% of the work. The friction is at the
**entry point**:

| Command | What you must already know |
|---|---|
| `csvcut -c` | exact column name, or its 1-based index, or a range `3-5` |
| `csvgrep -c -m/-r` | exact column **and** a substring or a regex you wrote yourself |
| `csvsort -c` | exact column |
| `csvsql --query` | SQL, **and** that the table is named after the file |
| `csvstat -c` | exact column |

Fourteen commands, and not one of them can answer "which column holds the email addresses?" or
"the amount column, but only the big orders". `csvstat` will tell you a column is `Text` with 412
unique values; it cannot tell you those values are company names.

That is the gap. It is semantic, not mechanical — which is exactly the class of problem a language
model is good at and deterministic code is bad at.

## 2. The extension point that made this safe

Before writing any prompt, I checked whether csvkit's own `argparse` parsers could be introspected
at runtime. They can:

```python
# stub out parse_args, construct the utility, read .argparser._actions
csvcut flags: ['--add-bom', '--columns', '--delete-empty-rows', '--delimiter',
               '--encoding', '--escapechar', '--linenumbers', '--maxfieldsize',
               '--names', '--no-doublequote', '--no-header-row', '--not-columns',
               '--quotechar', '--quoting', '--skip-lines', '--skipinitialspace',
               '--tabs', '--zero']
```

This is the single most important decision in the project. It means:

- no hand-maintained flag table to drift out of date,
- the model's output is validated against **the same definition the real command will parse**,
- a hallucinated `--only-show-profit` cannot reach execution, because it does not exist in the
  parser that would receive it.

`csvkit.schema.describe()` rebuilds this per command, so a csvkit upgrade that adds a flag
automatically widens what the model may use — with no code change here.

**Trap found while doing this:** a naive "find the class in the module that has `add_arguments`"
picks up `CSVKitUtility` itself (imported into every utility module) and dies with
`NotImplementedError: add_arguments must be provided by each subclass`. The fix is to skip classes
whose `__module__` is not the module being inspected.

## 3. First real call: classification

`examples/real/orders.csv` — 10 rows, 8 columns, deliberately dirty: one malformed email
(`wangqiang_at_example.com`), one missing phone, one non-ISO date (`2026-3-5`), one row where 备注
says `无`.

Real DeepSeek call, 1.74 s, 564 prompt + 314 completion tokens:

| Column | Label | Model conf | Measured match |
|---|---|---|---|
| 订单号 | `identifier` | 0.95 | 100% |
| 客户 | `person_name` | 0.95 | — |
| 联系邮箱 | `email` | 0.90 | **80%** |
| 电话 | `phone` | 0.95 | 100% |
| 下单日期 | `date` | 0.97 | 100% |
| 金额 | `currency_amount` | 0.95 | 100% |
| 城市 | `city` | 0.95 | — |
| 备注 | `free_text` | 0.92 | — |

The 80% is the whole argument for this project in one number. The model correctly identified the
column as email; a local regex check then found the one broken value. Neither half could have
produced that line alone.

## 4. Everything that broke, and why

These are the parts worth reading. Each one was found by running the thing, not by reasoning about
it.

### 4.1 The date column that became a phone column

Offline classifier output, first run:

```
下单日期  ->  phone  0.95  conf=1.0  [100% of sampled values match the phone pattern]
```

The phone pattern was `^\+?[\d\s()\-.]{7,20}$`. `2026-03-02` is 10 characters of digits and dashes,
so it matched perfectly. **A pattern that is "obviously right" silently mislabels every date column
in every file.**

Fixes, in order: require the value to start with a digit, exclude the ISO-date shape explicitly, and
add a `_PRIORITY` order so specific interpretations (date, percentage, currency) are tried before
generic ones (phone). Lesson: with overlapping regexes, "which one matches" is not enough — you need
an explicit precedence.

### 4.2 `大于 abc` quietly became `contains "abc"`

Rule engine, early version:

```
'sepal_length 大于 abc'  ->  csvgrep --columns sepal_length --match abc
```

An ordered comparison against a non-number is unexpressible. The code fell through to the generic
"contains" branch and produced a plan that would run and return rows. **Plausible wrong output is
worse than an error**, because nobody checks output that looks like it worked.

The fix is a hard `return None` for ordered comparisons on non-numeric literals, plus a strict
`_parse_number` that requires the whole string to be a number. `_parse_number` previously used
`re.search` for digits, which meant `大于 abc` could still find a number in prose elsewhere.

### 4.3 Grouping silently disappeared

`按 species 统计 sepal_length 的平均值` compiled to:

```sql
SELECT AVG("sepal_length") AS avg_sepal_length FROM data
```

No `GROUP BY`. The group pattern ended with `(?:分组|统计|...)` and consumed `统计`, which is the
*aggregation verb*, not the end of the group clause. The answer was a single global average
presented as if it were per-species.

Fix: remove `统计`/`计算` from the group-clause terminators, and make an unresolvable group clause
return `None` instead of falling through to an ungrouped query. **An unknown group column must not
degrade into a different question.**

### 4.4 The pipeline that died on `reconfigure`

Multi-step plans chain commands by swapping `sys.stdin`. First attempt used `io.StringIO`:

```
csvsort failed: AttributeError: '_io.StringIO' object has no attribute 'reconfigure'
```

csvkit calls `sys.stdin.reconfigure(encoding=...)`. Adding `reconfigure` as a no-op surfaced the next
one:

```
'_StdinBuffer' object has no attribute 'buffer'      # BOM handling
'_io.BytesIO' object has no attribute 'peek'        # csv.Sniffer
```

The real fix was not to keep bolting on shims but to wrap a **`io.BufferedReader(io.BytesIO(...))`
in an `io.TextIOWrapper`** — that gives the full binary layer, `peek()` included, for free.

**Why this matters:** every one of these three bugs is invisible to a single-step test. They only
appear once two commands are chained, which is exactly the case the planner was built to produce.

### 4.5 Blocking "shell metacharacters" blocked the actual feature

The planner rejected `$`, `^`, `{}`, `|`, `>` in values. That is defensible-sounding until you try
to express real requests:

```
--regex ^[0-9]{4,}$                         rejected: shell metacharacter
SELECT * FROM data WHERE 金额 > 100         rejected: shell metacharacter
```

Two of the most natural encodings, both refused — while protecting against nothing, because **no
shell is ever involved**. Execution is in-process function calls.

Narrowed to: `$(`, backticks, and control characters. The residual risk was never shell injection;
it was `--query` taking arbitrary SQL, which is handled separately by the SQL guard (below).

### 4.6 Retry logic rewrote history

`test_json_mode_rejection_retries_without_response_format` failed:

```
assert 'response_format' in transport.calls[0]['body']   # it was already gone
```

The fallback path did `body.pop('response_format', None)` on the same dict that had already been
sent. The transport's recorded first request was retroactively mutated, so the log said we never
tried JSON mode. Fixed by copying the dict.

This is a test that caught a *diagnostic* bug rather than a behaviour bug — but a lying request log
is how "it worked yesterday" investigations go wrong.

### 4.7 `extract_json_object` salvaged garbage

Given truncated output `{"a": {"b": 1}`, the first implementation returned `{'b': 1}` — a valid inner
object from an invalid outer one. A caller cannot tell that from a complete answer. Now an unbalanced
outermost object returns `None`, which triggers the one repair attempt instead of pretending.

### 4.8 Small samples made everything a "category"

With three rows, `distinct <= len(values) // 2` is true for almost anything, so text columns were
labelled `category` with 0.4 confidence. Requiring at least four non-null values and a short mean
length fixed it. **A heuristic tuned on 40-row samples is not valid on 3-row samples.**

### 4.9 CRLF inflated the patch tenfold

After editing csvkit's files with Python's text I/O on Windows, the patch showed 654 changed lines
across three files. The real change is 40 lines; the rest was every line ending rewritten `LF` → `CRLF`.
Caught by `diff -u` line counts not matching the edit size, and fixed by normalising to LF before
generating the patch.

Anyone publishing patches from Windows will hit this. `diff -rq` will not flag it (the file legitimately
differs); only the *size* of the hunk reveals it.

## 5. What the fallback actually buys

Same file, same question, layer disabled:

```
$ CSVAI_DISABLE=1 csvcut --ai "只要城市和金额" orders.csv
城市,金额
成都,1280.00
...
```

and an unexpressible one:

```
$ CSVAI_DISABLE=1 csvcut --ai "做一个漂亮的图表" orders.csv
csvcut: could not interpret the request offline.
csvcut: offline patterns that do work:
...
csvcut: error: no plan could be produced; ...
```

Exit 2, clean stdout. The distinction that matters: **the fallback covers the common cases and
refuses the rest**, rather than covering everything badly.

## 6. Verifying the "no harm" claim

Three independent checks, because this is the claim most likely to be false:

1. **Help text, byte for byte**, from a temp cwd where `import csvai` fails, against fixtures
   captured from a pristine upstream checkout. Six commands, all identical.
2. **Output, byte for byte**, for `csvcut -c` and `csvgrep -c -m` with and without the patch applied.
3. **csvkit's own 346 tests**, after the patch: `346 passed, 31 subtests passed`.

Then the same for the publishable artefact: extract upstream to `/tmp`, `patch -p1 --dry-run`,
apply, run the suite (346 passed), and run `--ai` end-to-end. A patch that only works in the
directory it was built in is not a patch.

### A note on the three BOM failures

On this machine, csvkit's suite reports `3 failed` for `test_removes_bom`, `test_match_utf8_bom`, and
`test_unicode_bom`. These are **not caused by this patch**. The failing assertion is
`['\ufefffoo', ...] != ['foo', ...]` — csvkit's `--encoding` default reads `PYTHONIOENCODING`, which
this shell sets to `utf-8` (not `utf-8-sig`), so the BOM is not stripped.

```bash
$ PYTHONIOENCODING=utf-8-sig python -m pytest tests/ -q
346 passed
```

Worth knowing before concluding "the patch broke something": check the environment's
`PYTHONIOENCODING` first.

## 7. Mutation testing

A green suite proves nothing if the tests cannot fail. Five deliberate breaks:

| Mutation | Result | Interpretation |
|---|---|---|
| Remove flag allow-list | 2 failed | Security tests bite |
| Remove SQL read-only guard | 1 failed | SQL guard tests bite |
| `> abc` degrades to substring | 1 failed | The refusal contract is tested |
| Trust extracted values | 1 failed | Provenance check is tested |
| Report 1.0 conformance | 3 failed | Measurement is tested |

All five were caught. One caveat worth recording: during the mutation run, the restore of
`extract.py` silently failed (wrong backup filename) and the suite stayed at 1 failure. **Always
re-run the clean baseline after mutation testing** — a stale mutation looks exactly like a real bug.

## 8. Honest accounting

What this adds that csvkit cannot do: semantic labelling of columns, extraction from prose, and
tolerance for not knowing your own headers.

What it does not add: any new data-processing capability. Every AI-produced plan executes as
ordinary csvkit. The value is in the translation layer, and the translation layer is 90% validation
code by volume — deliberately.

Cost, measured: 3 real calls (classify + plan + extract on a 10-row file) = 11,119 prompt +
1,222 completion tokens, 6.7 s total. The second identical classify cost **0 tokens** thanks to the
cache. For a 40-column file the profile is capped at 40 rows, so prompt size is bounded by column
count, not by file size.
