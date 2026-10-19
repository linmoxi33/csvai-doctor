"""The security boundary: what a hostile or confused model is allowed to do.

These tests are the reason the layer can be trusted. A model that hallucinates a
flag, invents a column, or tries to smuggle SQL through a value must be
*rejected*, not "best-effort executed". Each rejection is also a fallback
trigger, so the command still produces a correct answer via the rule engine.
"""

import pytest

from csvai import schema
from csvai.errors import PlanningError, UnsafeCommandError
from csvai.planner import MAX_STEPS, Plan, resolve_column, validate_plan, validate_sql
from tests.fakes import table_from

TABLE = table_from(
    ['订单号', '城市', '金额', 'notes'],
    [['A-1', '成都', '100', 'x'], ['A-2', '北京', '250', 'y']],
)

COMMANDS = {name: schema.describe(name) for name in schema.PLANNABLE}


def plan_of(*steps, reason='test'):
    return {'steps': [{'command': c, 'args': a} for c, a in steps], 'reason': reason}


# -- happy path --------------------------------------------------------

def test_single_step_plan_is_accepted():
    plan = validate_plan(plan_of(('csvcut', ['--columns', '城市,金额'])), TABLE, COMMANDS)
    assert isinstance(plan, Plan)
    assert plan.command == 'csvcut'
    assert plan.args == ['--columns', '城市,金额']
    assert plan.steps[0].command == 'csvcut'


def test_multi_step_plan_is_accepted():
    raw = plan_of(('csvcut', ['--columns', '城市,金额']),
                  ('csvsort', ['--columns', '金额', '--reverse']))
    plan = validate_plan(raw, TABLE, COMMANDS)
    assert len(plan.steps) == 2
    assert plan.describe().count('|') == 1


def test_legacy_single_command_shape_is_tolerated():
    raw = {'command': 'csvcut', 'args': ['--columns', '城市'], 'reason': 'old shape'}
    plan = validate_plan(raw, TABLE, COMMANDS)
    assert plan.command == 'csvcut'


def test_column_name_is_repaired_from_a_near_miss():
    # A model that writes the English word for the column should still land on
    # the real header rather than producing an empty result.
    plan = validate_plan(plan_of(('csvcut', ['--columns', '城市'])), TABLE, COMMANDS)
    assert plan.args == ['--columns', '城市']


def test_column_index_is_resolved():
    plan = validate_plan(plan_of(('csvcut', ['--columns', '2'])), TABLE, COMMANDS)
    assert plan.args == ['--columns', '城市']


def test_short_flag_is_normalised_to_long_form():
    plan = validate_plan(plan_of(('csvcut', ['-c', '城市'])), TABLE, COMMANDS)
    assert plan.args == ['--columns', '城市']


# -- rejections --------------------------------------------------------

def test_unknown_flag_is_rejected():
    with pytest.raises(UnsafeCommandError, match='not defined by csvcut'):
        validate_plan(plan_of(('csvcut', ['--only-show-profit'])), TABLE, COMMANDS)


def test_unknown_command_is_rejected():
    with pytest.raises(UnsafeCommandError, match='not allowed'):
        validate_plan(plan_of(('rm', ['-rf', '/'])), TABLE, COMMANDS)


def test_unknown_column_is_rejected():
    with pytest.raises(UnsafeCommandError, match='unknown column'):
        validate_plan(plan_of(('csvcut', ['--columns', '利润'])), TABLE, COMMANDS)


def test_positional_token_is_rejected():
    # A bare path would let the model choose a different input file.
    with pytest.raises(UnsafeCommandError, match='positional'):
        validate_plan(plan_of(('csvcut', ['/etc/passwd'])), TABLE, COMMANDS)


def test_file_flag_is_rejected():
    with pytest.raises(UnsafeCommandError, match='csvcut|not defined'):
        validate_plan(plan_of(('csvcut', ['--file', '/etc/passwd'])), TABLE, COMMANDS)


def test_command_substitution_is_rejected():
    with pytest.raises(UnsafeCommandError, match='control character'):
        validate_plan(plan_of(('csvgrep', ['--columns', 'notes', '--match', '$(id)'])),
                      TABLE, COMMANDS)


def test_newline_in_value_is_rejected():
    with pytest.raises(UnsafeCommandError, match='control character'):
        validate_plan(plan_of(('csvgrep', ['--columns', 'notes', '--match', 'a\nb'])),
                      TABLE, COMMANDS)


def test_too_many_steps_is_rejected():
    steps = [('csvcut', ['--columns', '城市'])] * (MAX_STEPS + 1)
    with pytest.raises(UnsafeCommandError, match='limit'):
        validate_plan(plan_of(*steps), TABLE, COMMANDS)


def test_missing_value_for_flag_is_rejected():
    with pytest.raises(UnsafeCommandError, match='requires a value'):
        validate_plan(plan_of(('csvcut', ['--columns'])), TABLE, COMMANDS)


def test_empty_plan_is_rejected_with_the_model_reason():
    with pytest.raises(PlanningError, match='cannot express'):
        validate_plan({'steps': [], 'reason': 'cannot express that'}, TABLE, COMMANDS)


def test_null_command_is_rejected_with_the_model_reason():
    with pytest.raises(PlanningError, match='cannot express'):
        validate_plan({'command': None, 'args': [], 'reason': 'cannot express that'},
                      TABLE, COMMANDS)


def test_non_dict_plan_is_rejected():
    with pytest.raises(PlanningError, match='JSON object'):
        validate_plan(['csvcut'], TABLE, COMMANDS)


def test_flag_choices_are_enforced():
    with pytest.raises(UnsafeCommandError, match='must be one of'):
        validate_plan(plan_of(('csvformat', ['--quoting', '99'])), TABLE, COMMANDS)


def test_value_length_is_capped():
    with pytest.raises(UnsafeCommandError, match='too long'):
        validate_plan(plan_of(('csvgrep', ['--columns', 'notes', '--match', 'x' * 600])),
                      TABLE, COMMANDS)


# -- regex values must remain usable -----------------------------------

def test_regex_with_anchors_and_quantifiers_is_allowed():
    # Blocking $ ^ { } would reject the most natural way to express "4+ digits".
    plan = validate_plan(
        plan_of(('csvgrep', ['--columns', '金额', '--regex', r'^[0-9]{3,}$'])),
        TABLE, COMMANDS,
    )
    assert plan.args == ['--columns', '金额', '--regex', r'^[0-9]{3,}$']


def test_sql_comparison_operators_are_allowed():
    plan = validate_plan(
        plan_of(('csvsql', ['--query', 'SELECT * FROM data WHERE 金额 > 100'])),
        TABLE, COMMANDS,
    )
    assert plan.args[0] == '--tables'
    assert plan.args[1] == 'data'


# -- SQL guard ---------------------------------------------------------

@pytest.mark.parametrize('query', [
    'DELETE FROM data',
    'UPDATE data SET 金额 = 1',
    'INSERT INTO data VALUES (1)',
    'DROP TABLE data',
    'ATTACH DATABASE "/tmp/x" AS x',
    'PRAGMA table_info(data)',
    'SELECT * FROM data; DROP TABLE data',
    'SELECT load_extension("evil")',
    'CREATE TABLE y (a int)',
])
def test_dangerous_sql_is_rejected(query):
    with pytest.raises(UnsafeCommandError):
        validate_sql(query)


@pytest.mark.parametrize('query', [
    'SELECT * FROM data',
    'select 城市, sum(金额) from data group by 城市',
    'WITH t AS (SELECT * FROM data) SELECT * FROM t',
    'SELECT * FROM data WHERE 金额 > 100',
    'SELECT * FROM data;',
])
def test_safe_sql_is_accepted(query):
    assert validate_sql(query)


def test_sql_comment_is_rejected():
    with pytest.raises(UnsafeCommandError, match='comments'):
        validate_sql('SELECT * FROM data -- sneaky')


def test_query_must_reference_the_fixed_table():
    with pytest.raises(UnsafeCommandError, match='table named "data"'):
        validate_plan(plan_of(('csvsql', ['--query', 'SELECT * FROM other'])),
                      TABLE, COMMANDS)


def test_csvsql_gets_the_table_name_injected():
    plan = validate_plan(plan_of(('csvsql', ['--query', 'SELECT * FROM data'])),
                         TABLE, COMMANDS)
    assert plan.args[:2] == ['--tables', 'data']


# -- column resolution -------------------------------------------------

def test_resolve_prefers_exact_match():
    assert resolve_column('城市', ['城市', '城市名']) == '城市'


def test_resolve_is_case_insensitive():
    assert resolve_column('NOTES', ['notes', 'amount']) == 'notes'


def test_resolve_returns_none_when_nothing_is_close():
    assert resolve_column('qqq', ['notes', 'amount']) is None


def test_resolve_handles_out_of_range_index():
    assert resolve_column('99', ['a', 'b']) is None


def test_empty_column_token_returns_none():
    assert resolve_column('  ', ['a']) is None
