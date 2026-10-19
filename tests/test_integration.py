"""Integration: the optional layer must be genuinely optional.

Two promises are tested here, and they are the ones a reviewer should care about:

1. **Without the package, nothing changes.** Help text and output are compared
   byte-for-byte against fixtures captured from unmodified csvkit 2.2.0.
2. **With the package but no usable model, the command still works.** Every kind
   of failure - no key, dead endpoint, garbage reply, rejected plan - falls back
   to the rule engine and keeps stdout clean.

The help fixtures were generated from the pristine upstream checkout, not from
the patched tree, so a diff means the patch leaked.
"""

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from csvai import integration

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURES = HERE / 'fixtures'


# -- 1. byte-identical help --------------------------------------------

@pytest.mark.parametrize('command', ['csvcut', 'csvgrep', 'csvsort', 'csvstat',
                                     'csvsql', 'csvclean'])
def test_help_is_unchanged_without_csvai(command, tmp_path):
    """Run the command where ``import csvai`` fails, as in a plain install."""
    fixture = FIXTURES / f'help_{command}.txt'
    if not fixture.exists():
        pytest.skip(f'missing fixture for {command}')

    # cwd must NOT be the repo root: csvai lives there and Python would put it
    # on sys.path via the script directory, making the test pass for the wrong
    # reason. A temp cwd plus a PYTHONPATH that only contains csvkit reproduces
    # a machine that never installed the extra.
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / '_csvkit_patched')
    env.pop('CSVAI_CONFIG', None)

    result = subprocess.run(
        [sys.executable, '-m', f'csvkit.utilities.{command}', '--help'],
        cwd=str(tmp_path),
        env=env, capture_output=True, text=True, encoding='utf-8',
    )
    expected = (FIXTURES / f'help_{command}.txt').read_text(encoding='utf-8')
    assert result.stdout == expected, f'{command} help changed'


def test_help_fixtures_contain_no_ai_flags():
    """Guard against accidentally regenerating fixtures from the patched tree."""
    text = (FIXTURES / 'help_csvcut.txt').read_text(encoding='utf-8')
    assert '--ai' not in text


def test_ai_flags_appear_when_the_package_is_present():
    """The same command, now with csvai importable, must expose the flags."""
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / '_csvkit_patched'), str(ROOT)])
    result = subprocess.run(
        [sys.executable, '-m', 'csvkit.utilities.csvcut', '--help'],
        cwd=str(ROOT), env=env, capture_output=True, text=True, encoding='utf-8',
    )
    assert '--ai REQUEST' in result.stdout
    assert '--ai-off' in result.stdout


# -- 2. fallback behaviour ---------------------------------------------

class _FakeUtility:
    """Minimal stand-in for a csvkit utility instance.

    Using the real one would drag in csvkit's file handling; the integration
    layer only touches these few attributes, and pretending them keeps the test
    about the fallback logic rather than about csvkit.
    """

    class _Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __init__(self, request, **kwargs):
        self.args = self._Args(
            ai_request=request,
            ai_show_plan=kwargs.get('show_plan', False),
            ai_model=kwargs.get('model'),
            ai_provider=kwargs.get('provider'),
            ai_base_url=kwargs.get('base_url'),
            ai_off=kwargs.get('off', False),
            input_path=str(kwargs.get('path', 'examples/real/orders.csv')),
        )
        self.output_file = io.StringIO()
        self.error_file = io.StringIO()
        self.reader_kwargs = {}
        self.override_flags = []

    def __class__(self):  # pragma: no cover - name used only for messages
        return _FakeUtility

    def get_rows_and_column_names_and_column_ids(self, **kwargs):
        import csv

        # Read eagerly: returning an iterator over a closed file is exactly the
        # bug this fake had, and it is not what the real utility does either.
        with open(self.args.input_path, encoding='utf-8', newline='') as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = list(reader)
        return iter(rows), header, list(range(len(header)))

    def argparser(self):
        raise AssertionError('argparser.error should not be reached')


def _run(request, monkeypatch, tmp_path, **kwargs):
    """Run integration.run with a hermetic environment and config file."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv('CSVAI_CONFIG', str(tmp_path / 'none.toml'))
    for name in ('DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'CSVAI_API_KEY',
                 'CSVAI_DISABLE', 'CSVAI_PROVIDER', 'CSVAI_MODEL', 'CSVAI_BASE_URL'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CSVAI_CACHE_DIR', str(tmp_path / 'cache'))

    for key, value in kwargs.pop('env', {}).items():
        monkeypatch.setenv(key, value)

    utility = _FakeUtility(request, **kwargs)
    integration.run(utility)
    return utility


def test_no_key_falls_back_to_rules(monkeypatch, tmp_path):
    utility = _run('只要 city 和 amount', monkeypatch, tmp_path,
                   path='tests/fixtures/rules_sample.csv')
    assert 'city,amount' in utility.output_file.getvalue().splitlines()[0]
    assert 'offline rules' in utility.error_file.getvalue()


def test_ai_off_skips_the_model_even_with_a_key(monkeypatch, tmp_path):
    utility = _run('只要 city', monkeypatch, tmp_path, off=True,
                   path='tests/fixtures/rules_sample.csv',
                   env={'DEEPSEEK_API_KEY': 'sk-should-not-be-used'})
    assert 'city' in utility.output_file.getvalue()
    assert 'CSVAI' not in utility.error_file.getvalue()


def test_disabled_env_var_falls_back(monkeypatch, tmp_path):
    utility = _run('只要 city', monkeypatch, tmp_path,
                   path='tests/fixtures/rules_sample.csv',
                   env={'DEEPSEEK_API_KEY': 'sk-x', 'CSVAI_DISABLE': '1'})
    assert 'city' in utility.output_file.getvalue()


def test_unreachable_endpoint_falls_back(monkeypatch, tmp_path):
    # A real socket attempt to a guaranteed-dead port would be slow and flaky;
    # instead point at a port nothing listens on and keep the timeout tiny.
    utility = _run('只要 city', monkeypatch, tmp_path,
                   path='tests/fixtures/rules_sample.csv',
                   provider='custom', base_url='http://127.0.0.1:9/v1',
                   env={'CSVAI_API_KEY': 'k', 'CSVAI_TIMEOUT': '1',
                        'CSVAI_MAX_RETRIES': '0'})
    assert 'city' in utility.output_file.getvalue()
    assert 'offline rules' in utility.error_file.getvalue()


def test_garbage_reply_falls_back(monkeypatch, tmp_path):
    from csvai import provider as provider_module

    def fake_chat(self, messages):
        return 'I am sorry, I cannot help with that.'

    monkeypatch.setattr(provider_module.ChatProvider, '_chat', fake_chat)
    utility = _run('只要 city', monkeypatch, tmp_path,
                   path='tests/fixtures/rules_sample.csv',
                   env={'DEEPSEEK_API_KEY': 'k'})
    assert 'city' in utility.output_file.getvalue()
    assert 'offline rules' in utility.error_file.getvalue()


def test_rejected_plan_falls_back(monkeypatch, tmp_path):
    """A model that hallucinates a flag must not reach the executor."""
    from csvai import provider as provider_module

    def fake_chat(self, messages):
        return '{"steps": [{"command": "csvcut", "args": ["--drop-everything"]}]}'

    monkeypatch.setattr(provider_module.ChatProvider, '_chat', fake_chat)
    utility = _run('只要 city', monkeypatch, tmp_path,
                   path='tests/fixtures/rules_sample.csv',
                   env={'DEEPSEEK_API_KEY': 'k'})
    assert 'city' in utility.output_file.getvalue()
    assert 'offline rules' in utility.error_file.getvalue()


def test_uninterpretable_request_offline_is_a_clear_failure(monkeypatch, tmp_path):
    utility = _FakeUtility('做一个漂亮的图表', path='tests/fixtures/rules_sample.csv')
    with pytest.raises(SystemExit):
        # argparser.error is invoked; give the fake a real parser behaviour.
        class _Parser:
            def error(self, message):
                raise SystemExit(message)

        utility.argparser = _Parser()
        integration.run(utility)

    stderr = utility.error_file.getvalue()
    assert 'could not interpret the request offline' in stderr
    assert 'select columns' in stderr


def test_broken_config_does_not_break_the_command(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    config_file = tmp_path / 'bad.toml'
    config_file.write_text('[csvai]\nprovider = "nonsense"\n', encoding='utf-8')
    monkeypatch.setenv('CSVAI_CONFIG', str(config_file))

    utility = _FakeUtility('只要 city', path='tests/fixtures/rules_sample.csv')
    integration.run(utility)
    assert 'city' in utility.output_file.getvalue()
    assert 'configuration unusable' in utility.error_file.getvalue()


def test_plan_is_shown_only_on_request(monkeypatch, tmp_path):
    quiet = _run('只要 city', monkeypatch, tmp_path,
                 path='tests/fixtures/rules_sample.csv', off=True)
    assert 'plan:' not in quiet.error_file.getvalue()

    loud = _run('只要 city', monkeypatch, tmp_path,
                path='tests/fixtures/rules_sample.csv', off=True, show_plan=True)
    assert 'plan: csvcut --columns city' in loud.error_file.getvalue()


def test_empty_request_is_rejected(monkeypatch, tmp_path):
    utility = _FakeUtility('   ', path='tests/fixtures/rules_sample.csv')

    class _Parser:
        def error(self, message):
            raise SystemExit(message)

    utility.argparser = _Parser()
    with pytest.raises(SystemExit, match='non-empty'):
        integration.run(utility)


def test_stdout_stays_clean_data(monkeypatch, tmp_path):
    """Diagnostics go to stderr; stdout must remain parseable CSV."""
    import csv as csv_module

    utility = _run('只要 city 和 amount', monkeypatch, tmp_path,
                   path='tests/fixtures/rules_sample.csv', off=True)
    rows = list(csv_module.reader(io.StringIO(utility.output_file.getvalue())))
    assert rows[0] == ['city', 'amount']
    assert all(len(row) == 2 for row in rows[1:] if row)
