"""Introspect csvkit's own command surface at runtime.

The AI never invents flags. We build an allow-list directly from each utility's
live ``argparse`` parser, which means the model's output is validated against
the same definition the real command uses - no duplicated hand-written table to
drift out of date, and no chance of a hallucinated ``--only-show-profit``
reaching a shell.

This module deliberately imports csvkit lazily so that ``csvai`` stays usable
(and testable) without csvkit installed.
"""

import contextlib
from dataclasses import dataclass, field

from .errors import PlanningError

#: Utilities we expose to natural-language planning.
#:
#: Excluded on purpose:
#: ``csvlook``/``csvpy`` are display/interactive only, so prose-to-argv is a
#: poor fit. ``csvstack``/``csvjoin`` need two or more input files, but the
#: planner receives a single table, so any plan it produced would be wrong.
#:
#: ``csvsql`` IS included, because grouping and aggregation ("total amount per
#: city") are among the most common real requests and csvkit has no non-SQL way
#: to express them. It is safe to expose only because of three constraints
#: enforced in :mod:`csvai.planner`: ``--query`` must be a single read-only
#: SELECT, it must read from the fixed table name ``data``, and a list of
#: mutating/attaching keywords is rejected. The model never gets to run
#: arbitrary SQL; it gets to fill in one guarded template.
PLANNABLE = (
    'csvcut',
    'csvgrep',
    'csvsort',
    'csvstat',
    'csvsql',
    'csvjson',
    'csvformat',
    'csvclean',
)

#: Commands our own code may execute even though the model may not plan them.
RULE_ONLY = ('csvsql',)

#: Flags that make no sense to auto-generate: they either need a real terminal,
#: change the tool's own behaviour, or would let the model escape the sandbox.
DENIED_FLAGS = {
    '-h',
    '--help',
    '-V',
    '--version',
    '-v',
    '--verbose',
    # A model must not be able to point the tool at another file.
    '-f',
    '--file',
}

#: Value-taking flags whose argument we never let the model choose freely.
#: These are validated separately (columns against the real header).
COLUMN_FLAGS = {'-c', '--columns', '-C', '--not-columns', '--group'}

_FLAG_KIND = {
    'store_true': 'bool',
    'store_false': 'bool',
    'store': 'value',
    'append': 'value',
    'count': 'bool',
}


@dataclass
class Flag:
    names: tuple
    dest: str
    kind: str            # 'bool' | 'value' | 'list'
    help: str = ''
    choices: tuple = ()
    takes_file: bool = False

    @property
    def primary(self):
        """Prefer the long form: more readable in logs and argv."""
        longs = [n for n in self.names if n.startswith('--')]
        return longs[0] if longs else self.names[0]

    def as_prompt_dict(self):
        data = {'flag': self.primary, 'help': (self.help or '').strip()}
        if self.kind == 'bool':
            data['type'] = 'boolean switch'
        elif self.kind == 'list':
            data['type'] = 'list of values'
        else:
            data['type'] = 'takes one value'
        if self.choices:
            data['allowed'] = list(self.choices)
        if self.takes_file:
            data['note'] = 'takes a file path; not selectable by the model'
        return data


@dataclass
class CommandSchema:
    name: str
    description: str = ''
    flags: dict = field(default_factory=dict)   # primary name -> Flag
    positional: tuple = ()

    def allowed(self):
        return sorted(self.flags)

    def flag(self, name):
        return self.flags.get(name)

    def as_prompt_dict(self):
        return {
            'command': self.name,
            'purpose': self.description,
            'flags': [f.as_prompt_dict() for f in self.flags.values()],
        }


class _NullNamespace:
    """Stands in for parsed args while we only want the parser's shape."""

    def __getattr__(self, name):
        return None


def _capture_parser(utility_name):
    """Build a utility with parsing stubbed out, then hand back its parser.

    csvkit utilities parse argv inside ``__init__`` and some call
    ``argparser.error()`` eagerly, so we replace ``parse_args`` for the
    duration of construction. No side effects beyond object creation occur.
    """
    import argparse

    from csvkit.utilities import (
        csvclean,
        csvcut,
        csvformat,
        csvgrep,
        csvjson,
        csvsort,
        csvsql,
        csvstat,
    )

    modules = {
        'csvclean': csvclean,
        'csvcut': csvcut,
        'csvformat': csvformat,
        'csvgrep': csvgrep,
        'csvjson': csvjson,
        'csvsort': csvsort,
        'csvsql': csvsql,
        'csvstat': csvstat,
    }
    module = modules.get(utility_name)
    if module is None:
        raise PlanningError(f'unknown csvkit utility: {utility_name}')

    cls = None
    for attribute in dir(module):
        candidate = getattr(module, attribute)
        if not isinstance(candidate, type):
            continue
        # Skip the base class: it lives in csvkit.cli, but a star-import would
        # otherwise surface it here and it raises NotImplementedError.
        if candidate.__module__ != module.__name__:
            continue
        if not hasattr(candidate, 'add_arguments'):
            continue
        cls = candidate
        break
    if cls is None:  # pragma: no cover - defensive
        raise PlanningError(f'cannot locate utility class in {utility_name}')

    original = argparse.ArgumentParser.parse_args

    def _stub(self, *args, **kwargs):
        return _NullNamespace()

    argparse.ArgumentParser.parse_args = _stub
    try:
        with contextlib.redirect_stderr(_DevNull()):
            instance = cls(args=[])
    finally:
        argparse.ArgumentParser.parse_args = original

    return instance.argparser, getattr(cls, 'description', '') or ''


class _DevNull:
    def write(self, *_args):
        return 0

    def flush(self):
        return None


def _classify(action):
    if action.option_strings in ([], None):
        return None
    if action.nargs == 0:
        return 'bool'
    if action.nargs in ('+', '*') or action.__class__.__name__ in (
        'ExtendAction',
        '_AppendAction',
    ):
        return 'list'
    kind = _FLAG_KIND.get(action.__class__.__name__.replace('_', '').lower(), 'value')
    return kind


def describe(utility_name):
    """Return a :class:`CommandSchema` for one csvkit command."""
    parser, description = _capture_parser(utility_name)
    flags = {}
    positionals = []

    for action in parser._actions:  # argparse exposes no public equivalent
        names = tuple(action.option_strings or ())
        if not names:
            if action.dest not in ('help',):
                positionals.append(action.dest)
            continue
        if set(names) & DENIED_FLAGS:
            continue
        kind = _classify(action)
        if kind is None:
            continue
        flag = Flag(
            names=names,
            dest=action.dest,
            kind=kind,
            help=(action.help or '').replace('%default', ''),
            choices=tuple(action.choices or ()),
            takes_file=action.type is not None
            and getattr(action.type, '__name__', '') == 'FileType',
        )
        flags[flag.primary] = flag

    return CommandSchema(
        name=utility_name,
        description=description,
        flags=flags,
        positional=tuple(positionals),
    )


def all_schemas():
    schemas = {}
    for name in PLANNABLE:
        try:
            schemas[name] = describe(name)
        except Exception:  # pragma: no cover - a missing optional dep
            continue
    return schemas
