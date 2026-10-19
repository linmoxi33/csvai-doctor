"""Execute a validated plan by driving csvkit itself, in-process.

Why in-process rather than ``subprocess``:

* No shell, so no quoting bugs and no injection surface.
* A "pipe" is a Python text buffer passed to the next utility's stdin, which
  means a multi-step plan is still plain function calls.
* Startup cost stays low: importing csvkit once beats spawning an interpreter
  per step.

The executor is intentionally dumb. All judgement happened in the planner; here
we only translate steps into argv, run them, and normalise failures into
:class:`~csvai.errors.PlanningError`.
"""

import contextlib
import io
import os
import sys
from dataclasses import dataclass, field

from .errors import PlanningError

#: (command, module path, class name) - resolved lazily so that importing this
#: module does not require every optional csvkit dependency (dbf, xls, ...).
_COMMANDS = {
    'csvcut': ('csvkit.utilities.csvcut', 'CSVCut'),
    'csvgrep': ('csvkit.utilities.csvgrep', 'CSVGrep'),
    'csvsort': ('csvkit.utilities.csvsort', 'CSVSort'),
    'csvstat': ('csvkit.utilities.csvstat', 'CSVStat'),
    'csvjson': ('csvkit.utilities.csvjson', 'CSVJSON'),
    'csvformat': ('csvkit.utilities.csvformat', 'CSVFormat'),
    'csvstack': ('csvkit.utilities.csvstack', 'CSVStack'),
    'csvjoin': ('csvkit.utilities.csvjoin', 'CSVJoin'),
    'csvclean': ('csvkit.utilities.csvclean', 'CSVClean'),
    'csvsql': ('csvkit.utilities.csvsql', 'CSVSQL'),
}

#: Commands whose argv already names its inputs, so we must not append one.
_NO_INPUT_ARG = {'csvstack', 'csvjoin'}


@dataclass
class Result:
    command: str
    argv: list
    stdout: str
    stderr: str = ''
    steps: list = field(default_factory=list)

    @property
    def rows(self):
        lines = [line for line in self.stdout.splitlines() if line != '']
        return max(0, len(lines) - 1)

    @property
    def line_count(self):
        return len([line for line in self.stdout.splitlines() if line != ''])

    def preview(self, limit=5):
        lines = [line for line in self.stdout.splitlines() if line != '']
        head = lines[: limit + 1]
        suffix = ''
        if len(lines) > limit + 1:
            suffix = f'\n... ({len(lines) - limit - 1} more lines)'
        return '\n'.join(head) + suffix


def _load(command):
    if command not in _COMMANDS:
        raise PlanningError(f'cannot execute unknown command {command!r}')
    module_name, class_name = _COMMANDS[command]
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as exc:  # pragma: no cover - optional csvkit extras
        raise PlanningError(f'{command} is unavailable: {exc}') from None
    return getattr(module, class_name)


class _StdinBuffer(io.TextIOWrapper):
    """Stand-in for ``sys.stdin`` when chaining steps.

    csvkit's utilities assume a real console stream: they call
    ``reconfigure(encoding=...)``, and the CSV sniffer reaches for ``.buffer``
    and ``.peek()``. A ``StringIO`` provides none of that, and every gap showed
    up as a different AttributeError in a different command - reconfigure for
    csvsort, peek for the sniffer. Wrapping a BytesIO gives the binary layer for
    free, so the whole surface behaves like the stdin csvkit expects.
    """

    def __init__(self, text):
        # BufferedReader, not BytesIO: only the buffered wrapper exposes
        # ``peek()``, which csv.Sniffer uses to look ahead without consuming.
        raw = io.BufferedReader(io.BytesIO(text.encode('utf-8')))
        super().__init__(raw, encoding='utf-8', newline='', write_through=True)

    def reconfigure(self, *args, **kwargs):
        # Encoding is already fixed; csvkit only wants to be sure.
        return None


class _Buffer(io.TextIOWrapper):
    """Text stream over a byte buffer, so csvkit can write and we can decode."""

    def __init__(self):
        super().__init__(io.BytesIO(), encoding='utf-8', newline='', write_through=True)

    def value(self):
        self.flush()
        return self.buffer.getvalue().decode('utf-8', 'replace')


def _run_once(command, argv, input_path=None, stdin_text=None):
    cls = _load(command)
    args = [str(a) for a in argv]
    if input_path is not None and command not in _NO_INPUT_ARG:
        args.append(str(input_path))

    out = _Buffer()
    err = io.StringIO()

    original_stdin = sys.stdin
    if stdin_text is not None:
        # csvkit's _open_input_file uses sys.stdin when no path is given, so a
        # multi-step plan is chained by swapping stdin for a buffer.
        sys.stdin = _StdinBuffer(stdin_text)

    try:
        with contextlib.redirect_stderr(err):
            utility = cls(args, out)
            utility.run()
        stdout = out.value()
    except SystemExit as exc:
        raise PlanningError(
            f'{command} rejected the arguments: {err.getvalue().strip() or exc}'
        ) from None
    except Exception as exc:
        raise PlanningError(f'{command} failed: {type(exc).__name__}: {exc}') from None
    finally:
        sys.stdin = original_stdin
        try:
            out.close()
        except Exception:
            pass

    return stdout, err.getvalue()


def execute(command, argv, input_path=None):
    """Run one csvkit command with *argv* against *input_path*.

    Returns a :class:`Result`. Raises :class:`PlanningError` if the command
    rejects the arguments, so the caller can fall back or report.
    """
    stdout, stderr = _run_once(command, argv, input_path=input_path)
    args = [str(a) for a in argv]
    if input_path is not None and command not in _NO_INPUT_ARG:
        args.append(str(input_path))
    return Result(command=command, argv=args, stdout=stdout, stderr=stderr,
                  steps=[{'command': command, 'args': [str(a) for a in argv]}])


def execute_plan(plan, input_path=None):
    """Run every step, feeding each step's output into the next.

    Fails loudly on the first step that errors: returning partial output from a
    half-executed pipeline would look like a complete answer.
    """
    if not plan.steps:
        raise PlanningError('plan has no steps')

    text = None
    last_command = None
    last_argv = []
    trail = []
    stderr = ''

    for position, step in enumerate(plan.steps):
        stdin_text = text if position > 0 else None
        # Only the first step reads the real file.
        path = input_path if position == 0 else None
        text, stderr = _run_once(step.command, step.args, input_path=path,
                                 stdin_text=stdin_text)
        last_command, last_argv = step.command, list(step.args)
        trail.append(step.as_dict())

    return Result(command=last_command, argv=last_argv, stdout=text or '',
                  stderr=stderr, steps=trail)


def command_available(command):
    try:
        _load(command)
        return True
    except PlanningError:
        return False


def describe_environment():
    """Diagnostics for ``csvai doctor``: what can actually run here."""
    info = {'csvkit': None, 'commands': {}, 'python': sys.version.split()[0],
            'platform': os.name}
    try:
        from importlib.metadata import version

        info['csvkit'] = version('csvkit')
    except Exception:
        try:
            import csvkit as _csvkit

            info['csvkit'] = getattr(_csvkit, '__version__', 'unknown')
        except Exception:
            info['csvkit'] = None
    for command in _COMMANDS:
        info['commands'][command] = command_available(command)
    return info
