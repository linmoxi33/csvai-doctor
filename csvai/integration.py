"""The glue between csvkit commands and the optional AI layer.

This module is the *only* place where csvkit's runtime meets csvai, and it is
imported lazily: ``csvkit.cli`` looks for it only when a subclass sets
``ai_capable`` or when ``--ai`` is on the command line.

Responsibilities, in order:

1. Register the ``--ai*`` flags (called from the utility's parser setup).
2. Decide whether to use the model at all - honouring ``CSVAI_DISABLE``, a
   missing key, and ``--no-ai``.
3. Try the model, and on *any* failure fall back to the deterministic rule
   engine, printing one line to stderr explaining what happened.
4. Refuse to pretend: if neither path can serve the request, exit non-zero with
   a message that says what was tried.

Rule 3 is the whole point. An optional feature that turns a working command into
a stack trace the moment the network blips is worse than no feature.
"""

import sys


def add_arguments(parser, help_suffix=''):
    """Attach --ai flags to a utility's parser.

    Kept in one function so every opted-in command exposes the same surface, and
    so removing the layer is a single deletion in ``csvkit/cli.py``.
    """
    group = parser.add_argument_group(
        'optional AI assistance',
        'Requires the csvai package (pip install csvkit[ai]) and an API key. '
        'Without them these flags are not registered at all.'
    )
    group.add_argument(
        '--ai', dest='ai_request', metavar='REQUEST', default=None,
        help='Describe what you want in natural language, e.g. '
             '--ai "keep city and amount, sorted by amount desc". '
             'The request is compiled to ordinary csvkit flags and executed by csvkit.'
    )
    group.add_argument(
        '--ai-show-plan', dest='ai_show_plan', action='store_true',
        help='Print the compiled command to stderr before running it.')
    group.add_argument(
        '--ai-model', dest='ai_model', default=None,
        help='Override the configured model name for this invocation.')
    group.add_argument(
        '--ai-provider', dest='ai_provider', default=None,
        help='Override the configured provider (deepseek, openai, ollama, custom...).')
    group.add_argument(
        '--ai-base-url', dest='ai_base_url', default=None,
        help='Override the API base URL, e.g. a local OpenAI-compatible endpoint.')
    group.add_argument(
        '--ai-off', dest='ai_off', action='store_true',
        help='Never call the model; use the offline rule engine only. '
             'Useful for reproducible runs and air-gapped machines.')


#: Flags that mean "the user explicitly asked for offline behaviour".
_OFFLINE_SWITCHES = ('ai_off',)


def _cli_overrides(args):
    return {
        'provider': getattr(args, 'ai_provider', None),
        'model': getattr(args, 'ai_model', None),
        'base_url': getattr(args, 'ai_base_url', None),
    }


def _warn(utility, message):
    """One line on stderr. Never stdout: stdout is data and must stay clean."""
    stream = getattr(utility, 'error_file', None) or sys.stderr
    stream.write(f'{utility.__class__.__name__.lower()}: {message}\n')
    stream.flush()


def _profile_for(utility):
    """Build a table profile from the utility's own input handling.

    Reuses csvkit's reader so delimiters, encodings and skip-lines behave
    exactly as they will during execution. Taking a different path here would
    let the model reason about a different file than the one we then run.
    """
    from csvai.profile import profile_rows

    rows, column_names, _ = utility.get_rows_and_column_names_and_column_ids(
        **utility.reader_kwargs
    )
    sampled = []
    total = 0
    for row in rows:
        total += 1
        if len(sampled) < 40:
            sampled.append(row)
    return profile_rows(column_names, sampled, path=utility.args.input_path or '<stdin>',
                        delimiter=utility.reader_kwargs.get('delimiter', ','),
                        total_rows=total, sample_size=40)


def _make_provider(cfg):
    from csvai.provider import ChatProvider

    return ChatProvider(cfg)


def _make_cache(cfg):
    from csvai.cache import ResponseCache

    if not cfg.get('cache'):
        return None
    return ResponseCache(cfg.cache_path())


def _execute(utility, plan, buffer):
    """Run a plan and write its output into the utility's output stream."""
    from csvai import engine

    if getattr(utility.args, 'ai_show_plan', False):
        _warn(utility, f'plan: {plan.describe()}')

    result = engine.execute_plan(plan, utility.args.input_path)
    buffer.write(result.stdout)
    buffer.flush()
    return result


def run(utility):
    """Serve a ``--ai`` request, degrading to rules when the model is unusable."""
    from csvai import config as config_module
    from csvai import planner, rules
    from csvai.errors import CsvaiError

    request = utility.args.ai_request
    if not isinstance(request, str) or not request.strip():
        utility.argparser.error('--ai needs a non-empty request')

    table = _profile_for(utility)

    # Layer 1: configuration. A broken config must not break the command, so a
    # failure here means "offline only", not "crash".
    try:
        cfg = config_module.resolve(cli_values=_cli_overrides(utility.args))
    except CsvaiError as exc:
        _warn(utility, f'AI configuration unusable ({exc}); using offline rules')
        cfg = None

    offline_only = bool(cfg is None) or any(
        getattr(utility.args, name, False) for name in _OFFLINE_SWITCHES
    )
    if cfg is not None and not offline_only:
        from csvai.config import is_configured

        if not is_configured(cfg):
            _warn(utility, 'no usable model configured; using offline rules')
            offline_only = True

    # Layer 2: the model. Any exception is a fallback, never a crash.
    if not offline_only:
        try:
            provider = _make_provider(cfg)
            cache = _make_cache(cfg)
            plan = planner.plan(request, table, provider, cache=cache)
            result = _execute(utility, plan, utility.output_file)
            usage = provider.usage.as_dict()
            if usage['calls'] or usage['cache_hits']:
                _warn(utility, f'model: {cfg.model} '
                               f'({usage["calls"]} call(s), {usage["cache_hits"]} cache hit(s))')
            return result
        except CsvaiError as exc:
            _warn(utility, f'model could not serve this request ({exc}); using offline rules')
        except Exception as exc:  # pragma: no cover - unexpected, still degrade
            _warn(utility, f'model failed ({type(exc).__name__}: {exc}); using offline rules')

    # Layer 3: deterministic rules. This is the path that must always work.
    plan = rules.plan(request, table)
    if plan is None:
        _warn(utility, 'could not interpret the request offline.')
        _warn(utility, 'offline patterns that do work:')
        for line in rules.capabilities():
            _warn(utility, f'  - {line}')
        utility.argparser.error(
            'no plan could be produced; see the offline patterns above, '
            'or configure a model to handle open-ended requests'
        )
    return _execute(utility, plan, utility.output_file)
