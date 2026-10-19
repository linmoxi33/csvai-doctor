"""The offline engine: what it handles, and what it refuses.

The refusals matter as much as the successes. A rule that "helpfully" degrades an
unexpressible request into a nearby-but-wrong one produces plausible output that
nobody checks - the worst outcome for a data tool.
"""

import pytest

from csvai import rules
from tests.fakes import table_from

TABLE = table_from(
    ['city', 'amount', 'product', 'note'],
    [
        ['成都', '1280.00', '耳机', '希望周五前送到'],
        ['北京', '320.00', '键盘', ''],
        ['成都', '78.50', '鼠标垫', '发票抬头写公司名'],
        ['上海', '4520.00', '显示器', '无'],
    ],
)


def plan(request):
    return rules.plan(request, TABLE)


def describe(request):
    result = plan(request)
    return result.describe() if result else None


# -- selection ---------------------------------------------------------

@pytest.mark.parametrize('phrase,expected', [
    ('只要 city 和 amount', 'csvcut --columns city,amount'),
    ('保留 city、amount', 'csvcut --columns city,amount'),
    ('select city,amount', 'csvcut --columns city,amount'),
    ('去掉 note', 'csvcut --not-columns note'),
    ('drop note', 'csvcut --not-columns note'),
    ('排除 note 列', 'csvcut --not-columns note'),
])
def test_column_selection(phrase, expected):
    assert describe(phrase) == expected


# -- filtering ---------------------------------------------------------

def test_text_equality_is_anchored():
    # "US" must not match "USA", so equality compiles to an anchored regex.
    result = plan('city 等于 成都')
    assert result.command == 'csvgrep'
    assert result.args[:2] == ['--columns', 'city']
    assert result.args[2] == '--regex'
    assert result.args[3].startswith('^') and result.args[3].endswith('$')


def test_contains_matching_is_used_for_loose_phrasing():
    result = plan('product 包含 耳机')
    assert result.args == ['--columns', 'product', '--match', '耳机']


def test_numeric_comparison_goes_through_sql():
    # A substring matcher cannot express "> 1000"; only SQL can.
    result = plan('amount 大于 1000')
    assert result.command == 'csvsql'
    query = result.args[result.args.index('--query') + 1]
    assert '>' in query and '1000' in query
    assert '"amount"' in query


@pytest.mark.parametrize('phrase,operator', [
    ('amount 大于 100', '>'),
    ('amount 大于等于 100', '>='),
    ('amount 小于 100', '<'),
    ('amount 小于等于 100', '<='),
    ('amount 超过 100', '>'),
    ('amount 不低于 100', '>='),
])
def test_comparator_variants(phrase, operator):
    query = plan(phrase).args[-1]
    assert f'{operator} 100' in query


def test_thousands_separator_is_accepted():
    query = plan('amount 大于 1,000').args[-1]
    assert '1000' in query


def test_non_numeric_threshold_is_refused_not_matched():
    # The dangerous case: silently degrading to "contains abc".
    assert plan('amount 大于 abc') is None


def test_not_equal_uses_inverse_match():
    result = plan('city 不等于 成都')
    assert '--invert-match' in result.args
    assert result.args[result.args.index('--match') + 1] == '成都'


def test_trailing_conversational_padding_is_stripped():
    result = plan('把 city 是 成都 的行筛出来')
    assert result.args[result.args.index('--regex') + 1] == r'^\s*成都\s*$'


# -- sorting -----------------------------------------------------------

def test_sort_ascending_is_default():
    assert describe('按 amount 排序') == 'csvsort --columns amount'


def test_sort_descending_adds_reverse():
    assert describe('按 amount 降序') == 'csvsort --columns amount --reverse'


def test_sort_english_phrasing():
    assert describe('sort by amount desc') == 'csvsort --columns amount --reverse'


# -- aggregation -------------------------------------------------------

def test_grouped_sum_keeps_the_group_column():
    # Regression: "按 city 统计 amount 的平均值" once dropped the GROUP BY
    # because the trailing verb "统计" was consumed by the group pattern.
    query = plan('按 city 统计 amount 的平均值').args[-1]
    assert 'AVG("amount")' in query
    assert 'GROUP BY "city"' in query


def test_grouped_max_with_per():
    query = plan('每个 city 的 amount 最大值').args[-1]
    assert 'MAX("amount")' in query
    assert 'GROUP BY "city"' in query


def test_count_rows():
    query = plan('有多少行').args[-1]
    assert 'COUNT(*)' in query


def test_ungrouped_sum():
    query = plan('求 amount 的总和').args[-1]
    assert 'SUM("amount")' in query
    assert 'GROUP BY' not in query


def test_unknown_column_in_aggregate_is_refused():
    assert plan('求 profit 的总和') is None


# -- refusal -----------------------------------------------------------

@pytest.mark.parametrize('phrase', [
    '随便说点什么',
    '',
    '把数据整理一下',
    '做一个漂亮的图表',
    '预测下个月的销量',
])
def test_unrecognised_requests_return_none(phrase):
    assert plan(phrase) is None


def test_unknown_column_anywhere_is_refused():
    assert plan('只要 profit 和 amount') is None


def test_capabilities_are_documented():
    text = '\n'.join(rules.capabilities())
    assert 'select columns' in text
    assert 'aggregate' in text
