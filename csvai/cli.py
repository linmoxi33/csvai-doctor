"""``csvai`` - a thin CLI over the optional AI layer.

Two audiences:

* someone who wants the extra capabilities on their own (``classify``,
  ``extract``), without going through csvkit's flags;
* someone debugging a deployment (``doctor``, ``config``, ``cache``).

Every command has a deterministic path, so ``csvai classify file.csv`` works on
a machine with no key at all - it just says so in the output.
"""

import argparse
import csv
import json
import sys

from . import __version__, engine, rules
from . import cache as cache_module
from . import classify as classify_module
from . import config as config_module
from . import extract as extract_module
from . import profile as profile_module
from . import provider as provider_module
from .errors import CsvaiError


def _safe_print(text):
    """Print without dying on a non-UTF-8 console.

    Windows CI defaults to cp1252 and Chinese column names then raise
    UnicodeEncodeError mid-report, which looks like an application bug instead
    of a terminal limitation. Degrade the glyphs, never the exit code.
    """
    try:
        stream = sys.stdout
        encoding = getattr(stream, 'encoding', None) or 'utf-8'
        if encoding.lower().replace('-', '') not in ('utf8', 'utf8sig'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'backslashreplace').decode('ascii'))


def _build_context(args):
    cli = {'provider': getattr(args, 'provider', None),
           'model': getattr(args, 'model', None),
           'base_url': getattr(args, 'base_url', None),
           'cache': None if getattr(args, 'no_cache', False) else None}
    cfg = config_module.resolve(cli_values=cli)
    provider = provider_module.ChatProvider(cfg)
    cache = None
    if cfg.get('cache') and not getattr(args, 'no_cache', False):
        cache = cache_module.ResponseCache(cfg.cache_path())
    return cfg, provider, cache


def _load_table(args, table_profile=None):
    if table_profile is not None:
        return table_profile
    if args.input in ('-', None):
        text = sys.stdin.read()
        return profile_module.profile_text(text, delimiter=args.delimiter or ',')
    return profile_module.profile_file(
        args.input,
        sample_size=args.sample_rows,
        encoding=args.encoding,
        delimiter=args.delimiter,
    )


def _read_rows(path, encoding='utf-8-sig', delimiter=','):
    handle = sys.stdin if path in ('-', None) else open(path, encoding=encoding, newline='')
    try:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        rows = list(reader)
    finally:
        if handle is not sys.stdin:
            handle.close()
    return header or [], rows


# -- commands ----------------------------------------------------------

def cmd_classify(args):
    table = _load_table(args)
    if not table.columns:
        raise CsvaiError('no columns found; is the file empty?')

    cfg, provider, cache = _build_context(args)
    report = None
    degraded = None

    if not args.offline and config_module.is_configured(cfg):
        try:
            report = classify_module.classify(table, provider, cache=cache)
        except Exception as exc:
            degraded = f'{type(exc).__name__}: {exc}'
    elif not args.offline:
        degraded = 'no usable model configured'

    if report is None:
        report = classify_module.classify_offline(table)
        if degraded:
            report.note = f'model unavailable ({degraded}); ' + report.note

    payload = report.as_dict()
    payload['model'] = cfg.model if report.source == 'model' else None

    if args.json:
        _safe_print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    _safe_print(f'source: {report.source}')
    if report.note:
        _safe_print(f'note:   {report.note}')
    _safe_print('')
    header = f'{"column":<24} {"label":<16} {"conf":>5} {"match":>6}  reason'
    _safe_print(header)
    _safe_print('-' * len(header))
    for verdict in report.verdicts:
        match = '-' if verdict.conformance is None else f'{verdict.conformance:.0%}'
        _safe_print(f'{verdict.name[:24]:<24} {verdict.label:<16} '
                    f'{verdict.confidence:>5.2f} {match:>6}  {verdict.reason}')
        if verdict.offending:
            _safe_print(f'{"":<24} non-conforming: {verdict.offending[:2]}')
    return 0


def cmd_extract(args):
    header, rows = _read_rows(args.input, encoding=args.encoding,
                              delimiter=args.delimiter or ',')
    if not header:
        raise CsvaiError('no header row found')
    if args.column not in header:
        raise CsvaiError(f'column {args.column!r} not found; available: {header}')
    index = header.index(args.column)
    texts = [row[index] if index < len(row) else '' for row in rows]

    fields = [f.strip() for f in args.fields.split(',') if f.strip()]
    if not fields:
        raise CsvaiError('--fields needs at least one field name')

    cfg, provider, cache = _build_context(args)
    result = None
    note = ''
    if not args.offline and config_module.is_configured(cfg):
        try:
            result = extract_module.extract(texts, fields, args.column, provider, cache=cache)
        except Exception as exc:
            note = f'model unavailable ({type(exc).__name__}: {exc}); '
    elif not args.offline:
        note = 'no usable model configured; '

    if result is None:
        result, unsupported = extract_module.extract_offline(texts, fields, args.column)
        if unsupported:
            note += f'offline mode cannot extract: {", ".join(unsupported)}; '

    out_header = list(header) + list(fields)
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(out_header)
    for position, row in enumerate(rows):
        extra = ['' for _ in fields]
        for item in result.rows:
            if item['index'] == position:
                extra = [item['fields'].get(name) or '' for name in fields]
                break
        writer.writerow(list(row) + extra)

    if note or result.unverified:
        summary = f'extracted via {result.source}'
        if result.unverified:
            summary += f'; {len(result.unverified)} value(s) discarded as not present in source'
        if note:
            summary += '; ' + note.rstrip('; ')
        _safe_print(summary, )
    return 0


def cmd_ask(args):
    """Natural-language control from the csvai CLI, same engine as --ai."""
    from . import planner

    table = _load_table(args)
    cfg, provider, cache = _build_context(args)

    plan = None
    note = ''
    if not args.offline and config_module.is_configured(cfg):
        try:
            plan = planner.plan(args.request, table, provider, cache=cache)
        except Exception as exc:
            note = f'{type(exc).__name__}: {exc}'

    if plan is None:
        plan = rules.plan(args.request, table)
        if plan is None:
            message = 'could not interpret the request offline'
            if note:
                message += f' (model said: {note})'
            message += '\noffline patterns that work:\n  - ' + '\n  - '.join(rules.capabilities())
            raise CsvaiError(message)
        if note:
            _safe_print(f'note: model unavailable ({note}); used offline rules')

    _safe_print(f'plan: {plan.describe()}')
    _safe_print(f'why : {plan.reason}')
    if args.dry_run:
        return 0

    result = engine.execute_plan(plan, args.input)
    sys.stdout.write(result.stdout)
    return 0


def cmd_doctor(args):
    env = engine.describe_environment()
    cfg = None
    cfg_error = None
    try:
        cfg = config_module.resolve(cli_values={'provider': args.provider,
                                                'model': args.model,
                                                'base_url': args.base_url})
    except CsvaiError as exc:
        cfg_error = str(exc)

    _safe_print(f'csvai {__version__}')
    _safe_print(f'python {env["python"]} on {env["platform"]}')
    _safe_print(f'csvkit {env["csvkit"] or "NOT INSTALLED"}')
    missing = [name for name, ok in env['commands'].items() if not ok]
    if missing:
        _safe_print(f'commands unavailable: {", ".join(missing)}')
    else:
        _safe_print(f'commands available: {len(env["commands"])}/{len(env["commands"])}')
    _safe_print('')

    if cfg_error:
        _safe_print(f'config: INVALID - {cfg_error}')
        return 1

    _safe_print(f'provider     {cfg["provider"]}  ({cfg.source_of("provider")})')
    _safe_print(f'model        {cfg.model}  ({cfg.source_of("model")})')
    _safe_print(f'base_url     {cfg.base_url}  ({cfg.source_of("base_url")})')
    key = cfg.api_key
    shown = 'not set' if not key else (key[:4] + '...' + key[-4:])
    _safe_print(f'api_key      {shown}  ({cfg.source_of("api_key")})')
    _safe_print(f'timeout      {cfg["timeout"]}s   retries {cfg["max_retries"]}')
    _safe_print(f'cache        {"on at " + str(cfg.cache_path()) if cfg.get("cache") else "off"}')
    if cfg.config_path:
        _safe_print(f'config file  {cfg.config_path}')

    if not cfg.enabled:
        _safe_print('\nstatus: DISABLED (CSVAI_DISABLE or enabled=false)')
        _safe_print('        offline rule engine and regex extraction still work')
        return 0
    if not config_module.is_configured(cfg):
        _safe_print('\nstatus: NOT CONFIGURED - set an API key, or use --offline')
        return 1

    if args.check:
        _safe_print('\nprobing endpoint...')
        try:
            payload = provider_module.ChatProvider(cfg).complete_json(
                'You reply with JSON only.', 'Return {"ok": true}.')
            _safe_print(f'endpoint OK, model replied {json.dumps(payload, ensure_ascii=False)[:120]}')
            return 0
        except CsvaiError as exc:
            _safe_print(f'endpoint FAILED: {exc}')
            return 1

    _safe_print('\nstatus: configured (pass --check to verify the endpoint)')
    return 0


def cmd_config(args):
    cfg = config_module.resolve()
    _safe_print(json.dumps(cfg.as_public_dict(), ensure_ascii=False, indent=2))
    _safe_print('')
    _safe_print('source of each value:')
    for key in sorted(cfg.sources):
        _safe_print(f'  {key:<14} {cfg.source_of(key)}')
    _safe_print('')
    _safe_print('precedence: cli > env (CSVAI_*) > config file > defaults')
    return 0


def cmd_cache(args):
    cfg = config_module.resolve()
    store = cache_module.ResponseCache(cfg.cache_path())
    if args.clear:
        removed = store.clear()
        _safe_print(f'removed {removed} entrie(s) from {cfg.cache_path()}')
        return 0
    stats = store.stats()
    _safe_print(f'cache dir : {cfg.cache_path()}')
    _safe_print(f'entries   : {stats["entries"]}')
    _safe_print(f'bytes     : {stats["bytes"]}')
    return 0


def cmd_capabilities(args):
    _safe_print('offline (no model, no key):')
    for line in rules.capabilities():
        _safe_print(f'  - {line}')
    _safe_print('')
    _safe_print('offline extractors: ' + ', '.join(extract_module.offline_supported_fields()))
    _safe_print('')
    _safe_print('requires a model:')
    _safe_print('  - open-ended phrasing of the above, in any language')
    _safe_print('  - semantic column classification (beyond patterns and names)')
    _safe_print('  - extracting arbitrary fields from free text')
    _safe_print('  - grouping/aggregation phrased loosely ("who bought the most")')
    return 0


# -- parser ------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog='csvai',
        description='Optional AI assistance for csvkit. Every command has an '
                    'offline path, so nothing here requires a model.')
    parser.add_argument('-V', '--version', action='version', version=f'csvai {__version__}')
    sub = parser.add_subparsers(dest='command_name', required=True)

    def add_common(target, with_input=True):
        if with_input:
            # The input is a flag, not a positional: `ask` needs a positional for
            # the prose request, and two optional-or-required positionals in a
            # row make `ask --dry-run "按 city 排序"` swallow the request as the
            # filename. Explicit is worth the extra six characters.
            target.add_argument('-f', '--file', dest='input', default='-',
                                help='CSV file, or - for stdin (default: -)')
        target.add_argument('--provider', help='override provider')
        target.add_argument('--model', help='override model')
        target.add_argument('--base-url', dest='base_url', help='override base URL')
        target.add_argument('--encoding', default='utf-8-sig', help='input encoding')
        target.add_argument('--delimiter', default=None, help='input delimiter')
        target.add_argument('--sample-rows', dest='sample_rows', type=int, default=40,
                            help='rows to sample for profiling (default: 40)')
        target.add_argument('--offline', action='store_true',
                            help='never call a model; use the local path only')
        target.add_argument('--no-cache', dest='no_cache', action='store_true',
                            help='bypass the response cache for this run')

    p = sub.add_parser('classify', help='label columns by what their values are')
    add_common(p)
    p.add_argument('--json', action='store_true', help='machine-readable output')
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser('extract', help='pull structured fields out of a text column')
    add_common(p)
    p.add_argument('--column', required=True, help='the free-text column to read')
    p.add_argument('--fields', required=True,
                   help='comma-separated field names, e.g. "phone,amount,deadline"')
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser('ask', help='run a natural-language request against a CSV')
    p.add_argument('request', help='what you want, in prose')
    add_common(p)
    p.add_argument('--dry-run', dest='dry_run', action='store_true',
                   help='show the compiled plan without running it')
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser('doctor', help='report what is installed and configured')
    p.add_argument('--provider')
    p.add_argument('--model')
    p.add_argument('--base-url', dest='base_url')
    p.add_argument('--check', action='store_true', help='make one real call to verify')
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser('config', help='print the resolved configuration')
    p.set_defaults(func=cmd_config)

    p = sub.add_parser('cache', help='inspect or clear the response cache')
    p.add_argument('--clear', action='store_true')
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser('capabilities', help='what works offline vs what needs a model')
    p.set_defaults(func=cmd_capabilities)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CsvaiError as exc:
        sys.stderr.write(f'csvai: {exc}\n')
        return 2
    except BrokenPipeError:  # piping into head
        return 0


if __name__ == '__main__':
    sys.exit(main())
