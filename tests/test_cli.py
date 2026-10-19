"""Command-line surface: argument parsing and exit codes.

The failure this guards against is quiet and embarrassing: `ask` originally took
two positionals (`[input] request`), so `csvai ask --dry-run "按 city 排序"`
parsed the request as a *filename* and reported "could not interpret". Nothing
crashed; the tool simply ignored what the user asked. Only running it revealed
that, which is why these tests exist.
"""

import subprocess
import sys

import pytest

from csvai import cli

ROOT = __import__('pathlib').Path(__file__).resolve().parent.parent
SAMPLE = ROOT / 'examples' / 'real' / 'orders.csv'


def run(argv, env_extra=None):
    """Invoke the CLI in-process and capture the return code."""
    return cli.main(argv)


def run_subprocess(argv, env_extra=None):
    import os

    env = dict(os.environ)
    env['CSVAI_DISABLE'] = '1'
    env['PYTHONPATH'] = os.pathsep.join(
        [str(ROOT / '_csvkit_patched'), str(ROOT)]
    )
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, '-m', 'csvai.cli'] + argv,
                          cwd=str(ROOT), env=env,
                          capture_output=True, text=True, encoding='utf-8')


def test_ask_request_is_a_positional_not_a_filename(tmp_path, capsys):
    """The regression: a quoted request must reach the planner verbatim."""
    assert run(['ask', '-f', str(SAMPLE), '--dry-run', '--offline', '按 城市 排序']) == 0
    captured = capsys.readouterr()
    assert 'csvsort --columns 城市' in captured.out
    assert 'could not interpret' not in captured.err + captured.out


def test_ask_reads_from_stdin_by_default(capsys, monkeypatch):
    monkeypatch.setattr('sys.stdin', open(SAMPLE, encoding='utf-8'))
    assert run(['ask', '--dry-run', '--offline', '只要 城市']) == 0
    assert 'csvcut --columns 城市' in capsys.readouterr().out


def test_classify_offline_reports_its_source(capsys):
    assert run(['classify', '-f', str(SAMPLE), '--offline']) == 0
    out = capsys.readouterr().out
    assert 'source: rules' in out
    assert 'no semantic understanding' in out


def test_classify_json_is_machine_readable(capsys):
    import json

    assert run(['classify', '-f', str(SAMPLE), '--offline', '--json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['source'] == 'rules'
    assert isinstance(payload['columns'], list)
    assert all('label' in column for column in payload['columns'])


def test_extract_reports_unsupported_offline_fields(capsys):
    assert run(['extract', '-f', str(SAMPLE), '--column', '备注', '--offline',
                '--fields', 'phone,sentiment']) == 0
    out = capsys.readouterr().out
    assert 'phone' in out.splitlines()[0]
    assert 'offline mode cannot extract' in out


def test_uninterpretable_request_exits_nonzero():
    assert run(['ask', '-f', str(SAMPLE), '--offline', '做一个漂亮的图表']) == 2


def test_doctor_reports_not_configured_without_a_key(capsys, monkeypatch):
    for name in ('DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'CSVAI_API_KEY',
                 'CSVAI_PROVIDER', 'CSVAI_MODEL'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CSVAI_DISABLE', '1')
    code = run(['doctor'])
    out = capsys.readouterr().out
    # Disabled is its own state, distinct from "not configured": the offline
    # paths still work, so this must not look like a failure.
    assert code == 0
    assert 'DISABLED' in out


def test_config_never_prints_the_raw_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('CSVAI_API_KEY', 'sk-super-secret-value-123')
    monkeypatch.setenv('CSVAI_PROVIDER', 'custom')
    monkeypatch.setenv('CSVAI_BASE_URL', 'https://example.invalid/v1')
    monkeypatch.setenv('CSVAI_MODEL', 'm')
    monkeypatch.chdir(tmp_path)
    assert run(['config']) == 0
    out = capsys.readouterr().out
    assert 'sk-super-secret-value-123' not in out
    assert 'sk-s...-123' in out


def test_capabilities_lists_both_modes(capsys):
    assert run(['capabilities']) == 0
    out = capsys.readouterr().out
    assert 'offline (no model, no key)' in out
    assert 'requires a model' in out


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        run(['--version'])
    assert exc.value.code == 0


def test_missing_required_option_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        run(['extract', '-f', str(SAMPLE)])   # --column / --fields missing
    assert exc.value.code == 2


def test_subprocess_offline_smoke():
    """End-to-end in a fresh interpreter, the way CI runs it."""
    doctor = run_subprocess(['doctor'])
    assert doctor.returncode == 0
    assert 'DISABLED' in doctor.stdout

    ask = run_subprocess(['ask', '-f', str(SAMPLE), '--dry-run', '按 城市 排序'])
    assert ask.returncode == 0, ask.stderr
    assert 'csvsort --columns 城市' in ask.stdout
    assert ask.stderr == ''
