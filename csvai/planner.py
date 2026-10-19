"""Compile a natural-language request into a validated csvkit pipeline.

The contract that makes this safe rather than decorative:

* The model proposes a *plan* - a list of steps, each a command with argv. It
  never produces a command string, so it cannot reach a shell.
* Every flag it names must exist in the live argparse schema for that command.
* Every column it names must exist in the actual header (after fuzzy repair).
* Steps are limited in number and run through csvkit's own reader/writer, not
  through ``sh``. A pipe here is a Python file object, not a subprocess.
* Anything ambiguous or unknown is rejected, and rejection is a normal outcome
  that the caller answers with a deterministic fallback.

A plan is therefore data, and the executor is still csvkit itself.
"""

import difflib
import json
import re
from dataclasses import dataclass

from .errors import PlanningError, UnsafeCommandError
from .schema import COLUMN_FLAGS, describe

#: More than this and the plan is almost certainly a misunderstanding rather
#: than a genuine request. Three covers select-then-sort, filter-then-cut, and
#: similar two/three-stage shapes.
MAX_STEPS = 3

SYSTEM_PROMPT_TEMPLATE = """You translate a request about a CSV file into a short pipeline of Unix-style CLI commands.

You MUST reply with a single JSON object, no prose, with this exact shape:
{{"steps": [{{"command": "<allowed command>", "args": ["--flag", "value", ...]}}], "reason": "<short>"}}

Rules you must follow:
- Use ONLY the commands and flags listed below. Never invent a flag.
- Boolean flags appear alone in the list, with no following value.
- Value flags are followed by exactly one string value.
- Do NOT include the input filename; the caller supplies it and pipes each step into the next.
- Do NOT use "|", "@file", redirection, semicolons or backticks - the caller builds the pipe itself.
- Maximum {max_steps} steps. Use one step whenever one step suffices.
- For column selection, use the EXACT column names given to you, joined by commas.
- To keep some columns AND sort or filter, use two steps: csvcut first, then csvsort/csvgrep.
- csvsql runs the file as an in-memory SQLite table named "data". Use it for GROUP BY,
  aggregation, and WHERE comparisons on numbers. The query must be a single SELECT
  from "data"; never write INSERT/UPDATE/DELETE/ATTACH/PRAGMA.
- If the request genuinely cannot be expressed with the listed commands, return
  {{"steps": [], "reason": "<why>"}} instead of guessing or approximating.
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(max_steps=MAX_STEPS)

#: Nothing here is ever handed to a shell, so the only thing worth blocking is
#: command substitution. ``$``/``^``/``{}``/``|``/``>``/``<`` all stay allowed
#: because they are regex anchors, regex quantifiers, SQL operators, and SQL
#: alternation - and none of them means anything to Python's subprocess-free
#: execution. Blocking them rejected legitimate filters while protecting
#: nothing, because no shell ever parses these strings.
#:
#: What genuinely matters is ``$(...)``/backticks (command substitution in the
#: model's imagination) and control characters, which could split a token.
_SHELLISH = re.compile(r'\$\(|`|\x00|\r|\n')

#: Flags whose value legitimately contains shell-looking characters:
#: ``--regex`` uses ``|`` for alternation, and ``--query`` is SQL, which needs
#: ``||`` for concatenation. These are validated by their own rules instead.
_FREEFORM_FLAGS = {'--regex', '--query'}

#: Statements that must never appear in a model-authored query. The database
#: is an in-memory SQLite copy of one user-supplied CSV, so the danger is not
#: corrupting data - it is reading *other* files (ATTACH) or loading native
#: code (LOAD_EXTENSION).
_FORBIDDEN_SQL = re.compile(
    r'\b(?:attach|detach|pragma|vacuum|load_extension|insert|update|delete|'
    r'drop|alter|create|replace\s+into|reindex)\b',
    re.I,
)


def validate_sql(query):
    """Check that a model-authored query is a single read-only SELECT.

    We allow SQL planning because grouping ("total amount per city") is one of
    the most common real requests and has no non-SQL expression in csvkit.
    The guard is narrow and mechanical: one statement, read-only, no file
    attachment, no extension loading.
    """
    text = str(query).strip()
    if not text:
        raise UnsafeCommandError('empty SQL query')

    # A trailing semicolon is harmless; an embedded one starts a second query.
    trimmed = text.rstrip(';').strip()
    if ';' in trimmed:
        raise UnsafeCommandError('only a single SQL statement is allowed')

    if not re.match(r'^(?:select|with)\b', trimmed, re.I):
        raise UnsafeCommandError('only SELECT queries are allowed')

    hit = _FORBIDDEN_SQL.search(trimmed)
    if hit:
        raise UnsafeCommandError(f'forbidden SQL keyword: {hit.group(0).upper()}')

    if '--' in trimmed or '/*' in trimmed:
        raise UnsafeCommandError('SQL comments are not allowed')

    return trimmed

_MAX_ARGS = 40
_MAX_VALUE_LEN = 512


@dataclass
class Step:
    command: str
    args: list

    def as_dict(self):
        return {'command': self.command, 'args': list(self.args)}

    def describe(self):
        return ' '.join([self.command] + list(self.args))


@dataclass
class Plan:
    steps: list
    reason: str = ''
    source: str = 'model'          # 'model' | 'rules'

    @property
    def command(self):
        """First command: convenient when the plan is a single step."""
        return self.steps[0].command if self.steps else None

    @property
    def args(self):
        return list(self.steps[0].args) if self.steps else []

    def argv(self, input_path=None):
        """Single-step argv, kept for callers that only handle one command."""
        out = list(self.args)
        if input_path:
            out.append(str(input_path))
        return out

    def as_dict(self):
        return {
            'steps': [s.as_dict() for s in self.steps],
            'reason': self.reason,
            'source': self.source,
        }

    def describe(self):
        return ' | '.join(step.describe() for step in self.steps)


def _schema_block(commands):
    chunks = []
    for name, schema in commands.items():
        chunks.append(json.dumps(schema.as_prompt_dict(), ensure_ascii=False))
    return '\n'.join(chunks)


def build_prompt(request, table, commands):
    """Assemble the user message: schema, real header, and the request."""
    return (
        f'CSV file: {table.path}\n'
        f'Rows: {table.row_count}. Columns ({len(table.columns)}):\n'
        f'{table.header_block()}\n\n'
        f'Available commands and their flags (JSON lines):\n{_schema_block(commands)}\n\n'
        f'Request: {request}\n\n'
        'Reply with the JSON object only.'
    )


def resolve_column(token, header):
    """Map a model-supplied column reference onto a real header name.

    Accepts an exact name, a 1-based index, or a close-enough spelling. Returns
    ``None`` when nothing is confident enough - a wrong column is worse than a
    rejected plan, because it silently produces plausible but incorrect output.
    """
    token = str(token).strip().strip('"\'')
    if not token:
        return None
    if token in header:
        return token
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < len(header):
            return header[index]
    folded = {name.strip().lower(): name for name in header}
    if token.lower() in folded:
        return folded[token.lower()]
    matches = difflib.get_close_matches(token, header, n=1, cutoff=0.75)
    if matches:
        return matches[0]
    lowered = difflib.get_close_matches(token.lower(), list(folded), n=1, cutoff=0.75)
    if lowered:
        return folded[lowered[0]]
    return None


def _validate_columns(value, header, flag):
    """Resolve a comma-separated column list, or raise if any part is unknown."""
    parts = [p for p in str(value).split(',') if p.strip() != '']
    if not parts:
        raise UnsafeCommandError(f'{flag} needs at least one column')
    resolved = []
    for part in parts:
        # Preserve ranges like 1-3 and 2:4, which csvkit parses itself.
        if re.fullmatch(r'\d+[-:]\d+', part.strip()):
            resolved.append(part.strip())
            continue
        name = resolve_column(part, header)
        if name is None:
            raise UnsafeCommandError(f'unknown column {part!r} for {flag}')
        resolved.append(name)
    return ','.join(resolved)


def _validate_step(raw_step, table, allowed_commands):
    if not isinstance(raw_step, dict):
        raise PlanningError('each step must be a JSON object')

    command = raw_step.get('command')
    if not isinstance(command, str) or not command.strip():
        raise PlanningError('each step needs a command')
    command = command.strip()
    if command not in allowed_commands:
        raise UnsafeCommandError(f'command {command!r} is not allowed')

    args = raw_step.get('args') or []
    if not isinstance(args, list):
        raise PlanningError('args must be a list')
    if len(args) > _MAX_ARGS:
        raise UnsafeCommandError('too many arguments')

    schema = allowed_commands[command]
    header = table.names
    cleaned = []
    index = 0
    while index < len(args):
        token = args[index]
        token = token if isinstance(token, str) else str(token)
        token = token.strip()
        if not token:
            index += 1
            continue

        if _SHELLISH.search(token):
            raise UnsafeCommandError(f'shell metacharacter in {token!r}')

        if not token.startswith('-'):
            # A bare word is either a stray positional (the filename belongs to
            # the caller) or a value we already consumed; both mean malformed.
            raise UnsafeCommandError(f'unexpected positional token {token!r}')

        flag = schema.flag(token)
        if flag is None:
            alt = None
            for candidate in schema.flags.values():
                if token in candidate.names:
                    alt = candidate
                    break
            if alt is None:
                raise UnsafeCommandError(f'flag {token!r} is not defined by {command}')
            flag = alt

        if flag.takes_file:
            raise UnsafeCommandError(f'{flag.primary} takes a file path and is not allowed')

        if flag.kind == 'bool':
            cleaned.append(flag.primary)
            index += 1
            continue

        if index + 1 >= len(args):
            raise UnsafeCommandError(f'{flag.primary} requires a value')
        value = args[index + 1]
        value = value if isinstance(value, str) else str(value)
        if len(value) > _MAX_VALUE_LEN:
            raise UnsafeCommandError(f'value for {flag.primary} is too long')

        if flag.primary in COLUMN_FLAGS:
            value = _validate_columns(value, header, flag.primary)

        elif flag.primary == '--query':
            value = validate_sql(value)
            # The table name must be fixed so the query can only see this file.
            if not re.search(r'\bfrom\s+["\'`]?data\b', value, re.I):
                raise UnsafeCommandError(
                    'query must read from the table named "data" '
                    '(pass --tables data; the caller loads the CSV there)'
                )

        freeform = flag.primary in _FREEFORM_FLAGS
        if not freeform and _SHELLISH.search(value):
            raise UnsafeCommandError(f'control character in value of {flag.primary}')

        if flag.choices:
            if value not in [str(c) for c in flag.choices]:
                raise UnsafeCommandError(
                    f'{flag.primary} must be one of {list(flag.choices)}, got {value!r}'
                )

        cleaned.extend([flag.primary, value])
        index += 2

    if not cleaned:
        raise PlanningError('a step contained no arguments')

    # csvsql names its table after the input file, which the model never sees.
    # Pinning it to "data" makes any query the model writes predictable, and it
    # is what :func:`validate_sql` requires the FROM clause to reference.
    if command == 'csvsql' and '--tables' not in cleaned:
        cleaned = ['--tables', 'data'] + cleaned

    return Step(command=command, args=cleaned)


def validate_plan(raw, table, allowed_commands):
    """Turn a model payload into a :class:`Plan`, or raise.

    This is the security boundary. Everything downstream may assume every step
    contains only flags that its real command defines, and values that were
    checked against real data.
    """
    if not isinstance(raw, dict):
        raise PlanningError('plan must be a JSON object')

    # An explicit decline is a valid answer, not a malformed one. Check it
    # before anything else so the caller reports the model's own reason instead
    # of a confusing validation error.
    raw_steps = raw.get('steps')
    if raw_steps is None:
        # Tolerate the single-command shape some models fall back to.
        if raw.get('command'):
            raw_steps = [{'command': raw['command'], 'args': raw.get('args') or []}]
        else:
            reason = str(raw.get('reason') or '').strip() or 'model declined the request'
            raise PlanningError(reason)

    if not isinstance(raw_steps, list):
        raise PlanningError('steps must be a list')
    if not raw_steps:
        reason = str(raw.get('reason') or '').strip() or 'model returned an empty plan'
        raise PlanningError(reason)
    if len(raw_steps) > MAX_STEPS:
        raise UnsafeCommandError(f'plan has {len(raw_steps)} steps; the limit is {MAX_STEPS}')

    steps = [_validate_step(step, table, allowed_commands) for step in raw_steps]
    return Plan(steps=steps, reason=str(raw.get('reason') or ''), source='model')


def plan(request, table, provider, cache=None, commands=None):
    """Ask the model for a plan and validate it. Raises on any failure."""
    if commands is None:
        from .schema import PLANNABLE

        commands = {name: describe(name) for name in PLANNABLE}
    user = build_prompt(request, table, commands)
    raw = provider.complete_json(SYSTEM_PROMPT, user, cache=cache)
    return validate_plan(raw, table, commands)
