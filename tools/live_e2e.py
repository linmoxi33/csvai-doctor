"""End-to-end proof that the AI path really calls a model.

Run this with a real key to produce evidence:

    set DEEPSEEK_API_KEY=sk-...
    python tools/live_e2e.py

It performs four live checks and prints a table. Nothing is asserted, because a
provider outage should not look like a code failure - the point is to *show* what
happened, including token counts, latency, and cache behaviour. Exit code is 0
if every step produced usable output via either path.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '_csvkit_patched'))

from csvai import cache as cache_module  # noqa: E402
from csvai import classify, engine, extract, planner, rules  # noqa: E402
from csvai import config as config_module  # noqa: E402
from csvai import provider as provider_module  # noqa: E402

ORDERS = ROOT / 'examples' / 'real' / 'orders.csv'


def banner(title):
    print(f'\n{"=" * 68}\n{title}\n{"=" * 68}')


def main():
    cfg = config_module.resolve()
    configured = config_module.is_configured(cfg)

    banner('environment')
    print(f'provider : {cfg["provider"]}')
    print(f'model    : {cfg.model}')
    print(f'base_url : {cfg.base_url}')
    print(f'api_key  : {"set" if cfg.api_key else "MISSING"} ({cfg.source_of("api_key")})')
    print(f'usable   : {configured}')
    if not configured:
        print('\nNo key found. Set DEEPSEEK_API_KEY (or CSVAI_API_KEY) and re-run.')
        print('Every step below still works through the offline path, which this')
        print('script demonstrates by falling back automatically.')

    provider = provider_module.ChatProvider(cfg)
    cache = cache_module.ResponseCache(cfg.cache_path()) if cfg.get('cache') else None
    from csvai.profile import profile_file

    table = profile_file(str(ORDERS))
    print(f'\ninput    : {ORDERS.relative_to(ROOT)} '
          f'({table.row_count} rows x {len(table.columns)} columns)')

    results = []

    # 1. Semantic classification
    banner('1. classify: what are these columns, really?')
    started = time.time()
    try:
        report = classify.classify(table, provider, cache=cache)
        elapsed = time.time() - started
        for verdict in report.verdicts:
            match = '-' if verdict.conformance is None else f'{verdict.conformance:.0%}'
            print(f'  {verdict.name:<10} {verdict.label:<18} conf={verdict.confidence:.2f} '
                  f'match={match:<5} {verdict.reason}')
        results.append(('classify', 'model', elapsed, report.source))
    except Exception as exc:
        elapsed = time.time() - started
        print(f'  model failed: {type(exc).__name__}: {exc}')
        report = classify.classify_offline(table)
        for verdict in report.verdicts:
            print(f'  {verdict.name:<10} {verdict.label:<18} conf={verdict.confidence:.2f} '
                  f'{verdict.reason}')
        results.append(('classify', 'rules', elapsed, report.source))

    # 2. Natural language planning
    banner('2. plan: "每个城市的订单总金额，从高到低"')
    request = '每个城市的订单总金额，从高到低'
    started = time.time()
    plan = None
    try:
        plan = planner.plan(request, table, provider, cache=cache)
        source = 'model'
    except Exception as exc:
        print(f'  model failed: {type(exc).__name__}: {exc}')
    if plan is None:
        plan = rules.plan(request, table)
        source = 'rules'
    elapsed = time.time() - started
    if plan is None:
        print('  neither path could express the request')
        results.append(('plan', 'none', elapsed, 'none'))
    else:
        print(f'  plan : {plan.describe()}')
        print(f'  why  : {plan.reason}')
        print(f'  source: {source}')
        result = engine.execute_plan(plan, str(ORDERS))
        for line in result.stdout.splitlines()[:6]:
            print(f'    {line}')
        results.append(('plan', source, elapsed, plan.command))

    # 3. Free-text extraction
    banner('3. extract: fields out of the free-text 备注 column')
    import csv as csv_module

    with open(ORDERS, encoding='utf-8', newline='') as handle:
        reader = csv_module.reader(handle)
        header = next(reader)
        rows = list(reader)
    notes = [row[header.index('备注')] for row in rows]
    fields = ['contact_phone', 'amount', 'delivery_request', 'invoice_note']

    started = time.time()
    try:
        extraction = extract.extract(notes, fields, '备注', provider, cache=cache)
        source = 'model'
    except Exception as exc:
        print(f'  model failed: {type(exc).__name__}: {exc}')
        extraction, unsupported = extract.extract_offline(notes, fields, '备注')
        source = 'rules'
        if unsupported:
            print(f'  offline cannot extract: {unsupported}')
    elapsed = time.time() - started
    for item in extraction.rows[:4]:
        print(f'  row {item["index"]}: {json.dumps(item["fields"], ensure_ascii=False)}')
    print(f'  verified={extraction.verified} unverified={len(extraction.unverified)}')
    results.append(('extract', source, elapsed, f'{extraction.verified} fields'))

    # 4. Cache behaviour, which is what makes iteration affordable
    banner('4. cache: same question twice')
    if cache is None:
        print('  cache disabled, skipping')
    else:
        before = provider.usage.as_dict()
        classify.classify(table, provider, cache=cache)
        after = provider.usage.as_dict()
        print(f'  calls before={before["calls"]} after={after["calls"]} '
              f'(cache hits {after["cache_hits"]})')
        print('  a second identical question cost 0 tokens' if after['calls'] == before['calls']
              else '  cache miss: prompt or model changed')

    banner('usage')
    usage = provider.usage.as_dict()
    print(json.dumps(usage, indent=2))

    banner('summary')
    print(f'{"step":<10} {"path":<7} {"seconds":>8}  detail')
    for name, path, elapsed, detail in results:
        print(f'{name:<10} {path:<7} {elapsed:>8.2f}  {detail}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
