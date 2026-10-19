# csvai

**Optional AI assistance for [csvkit](https://github.com/wireservice/csvkit) — natural-language queries,
semantic column classification, and field extraction from free text. Every one of them degrades to a
deterministic fallback instead of failing.**

```
$ csvcut --ai "每个城市的订单总金额，从高到低" orders.csv
城市,总金额
上海,6860.0
成都,2118.4
重庆,540.5
北京,335.0
杭州,199.0
```

That output came from `csvsql --tables data --query "SELECT 城市, SUM(金额) AS 总金额 FROM data
GROUP BY 城市 ORDER BY 总金额 DESC"` — real csvkit, real SQLite, hand-written SQL that the model
proposed and this package validated before a single row was read.

---

## Why this exists

csvkit is excellent and has 6,400 stars. It is also **purely mechanical**: all fourteen of its
commands require you to already know

- the exact column name or index (`-c 3`, `-c "省 份"`),
- the exact substring or regex to match (`-m`, `-r`),
- which command expresses your intent, and
- how to compose them with pipes.

For someone with a 40-column export from a CRM whose headers are `客户名称` / `cust_name` /
`Customer Name` depending on who exported it, that is the whole difficulty. The tool does exactly
what you say; the hard part is saying it.

So the gap is not "csvkit cannot process CSV". It is **"csvkit cannot cope with you not knowing
your own data's shape"**. That is a semantic gap, and it is the one place a language model genuinely
helps.

## What it does not do

No chat box. There is no conversation, no context to manage, no assistant persona. The model has
exactly three jobs, each with a typed output that gets validated:

| Capability | Model's job | What verifies it |
|---|---|---|
| **Natural language → command** | Propose a plan: a list of `{command, args}` | Every flag checked against csvkit's own live `argparse`; every column checked against the real header |
| **Column classification** | Label each column from a closed vocabulary | Conformance measured in Python against the actual sampled values |
| **Field extraction** | Pull requested fields out of prose | Each extracted value must appear in its source row, or it is discarded |

The model never produces a shell string, never picks a file, and never writes SQL that is not a
single read-only `SELECT` from one fixed table.

## Install

```bash
pip install csvkit==2.2.0
pip install -e .          # this package, no runtime dependencies

# patch your csvkit to expose --ai (see "Applying the patch")
```

Then configure a key:

```bash
export DEEPSEEK_API_KEY=sk-...       # or OPENAI_API_KEY / MOONSHOT_API_KEY
# or, for a local model, nothing at all:
export CSVAI_PROVIDER=ollama
```

Check it:

```
$ csvai doctor --check
csvai 0.1.0
python 3.13.14 on nt
csvkit 2.2.0
commands available: 10/10

provider     deepseek  (env:CSVAI_PROVIDER)
model        deepseek-flash  (provider:deepseek)
base_url     https://api.deepseek.com/v1  (provider:deepseek)
api_key      sk-9...2d9  (env:DEEPSEEK_API_KEY)
...
endpoint OK, model replied {"ok": true}
```

## Three ways to use it

### 1. `--ai` on the commands you already use

```bash
csvcut  --ai "只要城市和金额，按金额从高到低"  orders.csv
csvgrep --ai "找出金额大于1000的订单"          orders.csv
csvcut  --ai "把成都的订单挑出来"              orders.csv
```

`--ai-show-plan` prints the compiled command to stderr so you can copy it and drop the flag:

```
$ csvcut --ai "只看城市和金额按金额降序" --ai-show-plan orders.csv
csvcut: plan: csvcut --columns 城市,金额 | csvsort --columns 金额 --reverse
csvcut: model: deepseek-flash (1 call(s), 0 cache hit(s))
城市,金额
上海,4520.00
...
```

### 2. `csvai` for what csvkit has no command for

```bash
csvai classify -f orders.csv       # what are these columns, really?
csvai extract -f orders.csv --column 备注 --fields phone,amount,deadline
csvai ask -f orders.csv "按城市统计金额的平均值"
```

`classify` is the most useful of the three. It tells you that a column is a *company name*, not
"text", and — because conformance is measured rather than trusted — it shows you the rows that
disagree:

```
$ csvai classify -f orders.csv
source: model

column                   label             conf  match  reason
--------------------------------------------------------------
订单号                    identifier        0.95   100%  Sequential order IDs A-1001 format
客户                      person_name       0.95      -  Chinese personal names
联系邮箱                   email             0.90    80%  Email addresses, one malformed entry
                         non-conforming: ['wangqiang_at_example.com']
电话                      phone             0.95   100%  Mobile numbers with country code
下单日期                   date              0.98   100%  ISO dates YYYY-MM-DD
金额                      currency_amount   0.90   100%  Decimal monetary values
城市                      city              0.95      -  Chinese city names
备注                      free_text         0.90      -  Free-form Chinese delivery notes
```

The `80%` is the part that matters: the model said "email column", Python checked all five
non-empty values, and one of them is broken. You now know exactly which one.

### 3. As a library

```python
from csvai import config, provider, classify, profile

cfg = config.resolve()
table = profile.profile_file('orders.csv')
report = classify.classify(table, provider.ChatProvider(cfg))
for verdict in report.verdicts:
    print(verdict.name, verdict.label, verdict.conformance)
```

## The fallbacks are the feature

This is the part most "add AI to X" projects skip. Here, **the model being unavailable is a normal
operating mode**, not an error.

```
$ CSVAI_DISABLE=1 csvcut --ai "只要城市和金额" orders.csv
城市,金额
成都,1280.00
...
```

The offline engine handles the highest-frequency intents with rules:

| Request | Offline plan |
|---|---|
| `只要 A 和 B` / `drop A` / `select a,b` | `csvcut --columns` / `--not-columns` |
| `A 等于 x` | `csvgrep --regex ^\s*x\s*$` (anchored, so `US` ≠ `USA`) |
| `A 包含 x` | `csvgrep --match x` |
| `A 大于 100` | `csvsql ... WHERE "A" > 100` (a regex cannot express `>`) |
| `按 A 排序` / `按 A 降序` | `csvsort --columns A [--reverse]` |
| `按 A 统计 B 的平均值` | `csvsql ... AVG("B") GROUP BY "A"` |
| `统计` / `describe` | `csvstat` |

Anything outside those patterns **returns nothing**. That is deliberate: the rule engine is
designed to fail rather than produce a plausible wrong answer.

```
$ csvcut --ai "做一个漂亮的图表" orders.csv
csvcut: could not interpret the request offline.
csvcut: offline patterns that do work:
csvcut:   - select columns: 只要 A 和 B / 去掉 A / select a,b / drop a
csvcut:   - filter rows: A 等于 x / A 包含 x / A 大于 100 / A 不等于 x
csvcut:   - sort rows: 按 A 排序 / 按 A 降序 / sort by a desc
csvcut:   - aggregate: 按城市统计销售额的平均值 / 求 price 的最大值 / 有多少行
csvcut:   - statistics: 统计 / 统计 A 的平均值 / describe / summary
csvcut: error: no plan could be produced; see the offline patterns above, or configure a model
        to handle open-ended requests
```

Exit code 2, clean stderr, nothing on stdout. A pipeline that consumes this never sees garbage.

### What degradation looks like in practice

Measured on `examples/real/orders.csv`, same question, both paths:

| Column | Model says | Offline says |
|---|---|---|
| 订单号 | `identifier` (100% match) | `identifier` (100% match) |
| 客户 | `person_name` | **`unknown`** |
| 联系邮箱 | `email` (80%) | `email` (80%) |
| 备注 | `free_text` | **`unknown`** |

The two agree on everything a pattern can prove, and the model adds exactly two things a pattern
cannot: that `客户` holds people's names and `备注` is prose. The offline path admits it does not
know instead of guessing.

## Applying the patch

The three-file change to csvkit is 40 lines total: a lazy hook in `cli.py`, and `ai_capable = True`
on `CSVCut` and `CSVGrep`. The hook checks for `import csvai` and registers the flags only if it
succeeds, so a stock csvkit install never sees them.

```bash
python -c "import csvkit, os; print(os.path.dirname(os.path.dirname(csvkit.__file__)))"
cd /path/to/site-packages
patch -p1 < /path/to/csvai-doctor/patches/0001-add-optional-ai-hook.patch
```

The patch is against csvkit **2.2.0** exactly. `patch --dry-run` first; the test suite asserts it
applies cleanly, so a new csvkit release will fail CI rather than fail silently.

## Guarantees, and how they are tested

**Without the package, nothing changes.**

```
tests/test_integration.py::test_help_is_unchanged_without_csvai[csvcut]   PASSED
tests/test_integration.py::test_help_is_unchanged_without_csvai[csvgrep]  PASSED
tests/test_integration.py::test_help_is_unchanged_without_csvai[csvsort]  PASSED
tests/test_integration.py::test_help_is_unchanged_without_csvai[csvstat]  PASSED
tests/test_integration.py::test_help_is_unchanged_without_csvai[csvsql]   PASSED
tests/test_integration.py::test_help_is_unchanged_without_csvai[csvclean] PASSED
```

Those run `--help` from a temporary cwd where `import csvai` fails, and assert that **no AI flag
appears and the option set is identical to upstream's**. They do not compare raw bytes, because
argparse's rendering changed between Python 3.12 and 3.13 (`-d DELIMITER, --delimiter DELIMITER`
became `-d, --delimiter DELIMITER`) — that is upstream churn, not a leak, and asserting on it would
have made CI red on a different interpreter for reasons unrelated to this patch. A byte-for-byte
check still runs, gated on the fixtures' recorded interpreter version.

**Upstream behaviour is untouched.** After applying the patch:

```
346 passed, 31 subtests passed
```

That is csvkit's own full suite, unchanged, including the three BOM tests (which fail on a bare
`PYTHONIOENCODING=utf-8` environment for reasons unrelated to this patch — see
[`DIAGNOSIS.md`](DIAGNOSIS.md)).

**The safety boundary holds.** 44 tests in `tests/test_planner.py` drive a hostile model: invented
flags, `--file /etc/passwd`, `$(id)`, `ATTACH DATABASE`, `DROP TABLE`, an out-of-range column index,
an unknown command. All are rejected, and rejection falls back rather than crashing.

**The tests can fail.** Mutation testing — deliberately breaking each guarantee and checking the
suite notices:

| Mutation | Tests that caught it |
|---|---|
| Remove the flag allow-list check | 2 failed |
| Remove the read-only SQL check | 1 failed |
| Degrade `A > abc` to substring matching | 1 failed |
| Trust extracted values without checking the source | 1 failed |
| Report 100% conformance instead of measuring it | 3 failed |

## Design notes worth knowing

**Why no `openai` SDK.** csvkit users install globally. Adding an HTTP client plus its
httpx/anyio tree to an optional extra that most will never enable is a bad trade for a wire format
that is four fields wide. The client is ~100 lines of `urllib` with injectable transport, which also
makes every failure branch testable without network.

**Why `csvsql` is exposed at all.** Grouping — "total amount per city" — is among the most common
real requests and csvkit has no non-SQL way to express it. Exposing it required three guards: single
statement, read-only `SELECT`/`WITH`, must read from the fixed table `data`, and a keyword blocklist
covering `ATTACH`/`PRAGMA`/`LOAD_EXTENSION`/DML/DDL. The model fills in one guarded template; it does
not get arbitrary SQL.

**Why the cache exists.** Not for speed. Classifying the same 60 columns while iterating on a prompt
is a pointless bill, and a cached answer replays byte-identically, so tests that exercise the AI path
do not depend on the model's mood. Measured: second identical question → 0 additional tokens.

**Why conformance is computed in Python.** A model that says "this is an email column" at 0.95
confidence is often right and sometimes wrong. Checking every sampled value against a local pattern
converts that opinion into a measurement, and the `non-conforming` list turns a silent data problem
into a visible one. When model and data disagree, **the data wins and the report says so**.

**Why `$`, `^`, `{}`, `|`, `>` are allowed in values.** An early version blocked them as "shell
metacharacters" and thereby rejected `--regex ^[0-9]{4,}$` and `WHERE 金额 > 100` — the two most
natural ways to express those requests. Nothing here is ever handed to a shell, so the only real
threat is command substitution. The check was narrowed to `$(`, backticks, and control characters.

## Configuration

Precedence: **CLI flags > environment > config file > defaults**, and `csvai config` prints which
layer each value came from.

| Variable | Meaning |
|---|---|
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `MOONSHOT_API_KEY` | Provider key, picked up per provider |
| `CSVAI_PROVIDER` | `deepseek` (default), `openai`, `moonshot`, `ollama`, `custom` |
| `CSVAI_MODEL`, `CSVAI_BASE_URL`, `CSVAI_API_KEY` | Overrides |
| `CSVAI_TIMEOUT`, `CSVAI_MAX_RETRIES`, `CSVAI_SAMPLE_ROWS` | Tuning |
| `CSVAI_CACHE`, `CSVAI_CACHE_DIR` | Cache on/off and location |
| `CSVAI_DISABLE=1` | **Kill switch.** Wins over everything, survives a broken config file |
| `CSVAI_CONFIG` | Path to a TOML config |

```toml
# .csvai.toml
[csvai]
provider = "deepseek"
model = "deepseek-flash"
sample_rows = 40
cache = true
```

## Known limits

- **The planner cannot see past the sample.** It profiles 40 rows by default. A column that is clean
  in the first 40 and dirty afterwards will be classified from the clean part.
- **Three steps maximum.** Plans are bounded at 3 commands. Longer compositions are refused, not
  silently truncated.
- **Offline coverage is deliberate, not accidental.** The rule engine covers selection, filtering,
  sorting, and aggregation. It does not attempt anything paraphrase-heavy, and it says so.
- **`csvstack` / `csvjoin` are not plannable.** They need multiple input files; the planner sees one
  table, so any plan it produced would be wrong.
- **Not a general SQL interface.** The `csvsql` path exists for aggregation within one file. It is
  not a query engine over your database.
- **Patch, not fork.** This ships as a 40-line patch plus a separate package, so you can read the
  entire integration surface in one screen. If csvkit adds a plugin mechanism upstream, this patch
  should be replaced by it.

## Layout

```
csvai/
  config.py       four-layer config resolution, with provenance
  provider.py     OpenAI-compatible client, retries, JSON rescue, usage accounting
  cache.py        content-addressed response cache
  profile.py      token-efficient table profiling (sampling happens here)
  schema.py       introspects csvkit's own argparse -> the flag allow-list
  planner.py      NL -> validated plan; the security boundary
  rules.py        deterministic offline planner
  classify.py     column labelling + measured conformance
  extract.py      field extraction + provenance checking
  engine.py       executes plans by driving csvkit in-process
  integration.py  the only place csvkit and csvai meet
  cli.py          the `csvai` command
patches/
  0001-add-optional-ai-hook.patch    40 lines against csvkit 2.2.0
tests/           164 tests
tools/live_e2e.py  end-to-end proof against a real endpoint
```

## License

MIT for this package. csvkit itself is MIT, Copyright (c) Christopher Groskopf and James McKinney —
see [NOTICE](NOTICE). This repository contains a patch, not a redistribution of csvkit.
