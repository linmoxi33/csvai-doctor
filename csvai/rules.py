"""A deterministic, offline planner.

This is the fallback that makes the AI layer genuinely optional. It covers the
handful of intents that account for most real CSV work, using the same
:class:`~csvai.planner.Plan` object so the executor cannot tell the difference.

Design rules, learned the hard way:

* **Never guess silently.** If the request falls outside the supported
  patterns, return ``None`` and let the caller say so. A plausible-but-wrong
  filter is far more damaging than an honest "I cannot do that offline".
* **Only emit constructs that can actually express the intent.** A substring
  matcher cannot express ``> 100``; numeric comparisons are routed through
  ``csvsql``, which loads the file into in-memory SQLite and lets the database
  do the comparison properly.
* **Report what was matched.** ``Plan.reason`` explains the rule that fired, so
  the user can see why they got this result.
"""

import re

from .planner import Plan, Step, resolve_column

#: Comparators expressed in prose. Order matters: longer phrases first so that
#: "大于等于" is not consumed by "大于".
_COMPARATORS = [
    (r'(?:大于等于|不小于|不低于|>=|≥)', 'ge'),
    (r'(?:小于等于|不大于|不高于|<=|≤)', 'le'),
    (r'(?:不等于|!=|≠)', 'ne'),
    (r'(?:大于|超过|高于|多于|>|greater than|more than|above)', 'gt'),
    (r'(?:小于|低于|少于|<|less than|below)', 'lt'),
    # "contains" must be tried before equality: "包含" has no 'is' in it, but a
    # loose equality pattern would otherwise consume the whole phrase and then
    # fail to resolve, silently losing a perfectly expressible request.
    (r'(?:包含|含有|带有|含有|contains?|includes?|has)', 'contains'),
    (r'(?:等于|==|=|equals?|is|是)', 'eq'),
]

_OP_SQL = {'gt': '>', 'ge': '>=', 'lt': '<', 'le': '<=', 'ne': '!=', 'eq': '='}

_QUOTES = '"\'“”‘’「」『』'

_DROP_VERBS = r'(?:去掉|删除|不要|排除|移除|drop|exclude|remove|without)'
_KEEP_VERBS = r'(?:只要|只需要|保留|仅保留|选择|选取|取出|选出|select|keep|only|just)'

_LEADING_NOISE = re.compile(
    r'^(?:请|帮我|把|筛选|过滤|查找|找出|显示|列出|仅|只|rows?\s+where|filter|where|show|find|'
    r'给|将|从|其中)\s*',
    re.I,
)


def _plan(command, args, reason):
    """Build a single-step rule plan. Kept tiny so every rule reads the same."""
    return Plan(steps=[Step(command=command, args=list(args))],
                reason=reason, source='rules')


def _strip_quotes(text):
    return text.strip().strip(_QUOTES).strip()


def _split_columns(text):
    parts = re.split(r'\s*(?:,|、|和|与|及|以及|/|\band\b)\s*', text)
    return [_strip_quotes(p) for p in parts if _strip_quotes(p)]


def _resolve_all(tokens, header):
    resolved = []
    for token in tokens:
        cleaned = re.sub(r'(?:这几列|这些列|列|columns?|fields?)$', '', token).strip()
        name = resolve_column(cleaned, header) if cleaned else None
        if name is None:
            return None
        if name not in resolved:
            resolved.append(name)
    return resolved or None


def _parse_number(text):
    """Extract a numeric literal, or ``None`` when *text* is not numeric.

    Deliberately strict: the whole value must be a number (allowing thousands
    separators, a trailing percent, and surrounding whitespace). A loose search
    for digits would turn "大于 abc" into "match 5", which is the exact class of
    silent misinterpretation this module exists to avoid.
    """
    stripped = _strip_quotes(text)
    stripped = re.sub(r'[\s,，]', '', stripped)
    stripped = stripped.rstrip('%')
    if not stripped:
        return None
    if not re.fullmatch(r'-?\d+(?:\.\d+)?', stripped):
        return None
    return stripped


def _sort_direction(text):
    if re.search(r'(?:降序|倒序|从大到小|从高到低|desc|descending|reverse)', text, re.I):
        return 'desc'
    return 'asc'


#: Suffixes people append to a comparison literal. Applied repeatedly so that
#: "的行筛出来" peels off in two steps: "的行" then "筛出来".
_LITERAL_SUFFIXES = (
    r'筛选?出来', r'选出来', r'找出来', r'挑出来', r'列出来',
    r'的行', r'的记录', r'的数据', r'的', r'给我', r'即可', r'就行',
)


def _clean_literal(text):
    """Strip trailing conversational padding from a filter value.

    Peeling is repeated because these suffixes stack. A single pass leaves
    "Iris-setosa 的行" behind after removing "筛出来", which then matches
    nothing - a silent empty result, the worst possible failure mode.
    """
    value = _strip_quotes(text)
    changed = True
    while changed:
        changed = False
        for suffix in _LITERAL_SUFFIXES:
            new = re.sub(rf'\s*(?:{suffix})\s*$', '', value)
            if new != value:
                value = new.strip()
                changed = True
    return _strip_quotes(value)


def _quote_identifier(name):
    """Quote a column name for SQL. Doubling embedded quotes is the SQL rule."""
    return '"' + str(name).replace('"', '""') + '"'


def _quote_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def plan_select(request, header):
    """Handle keep/drop column selection."""
    for verb in (_DROP_VERBS, _KEEP_VERBS):
        pattern = rf'{verb}\s*(?:掉|了)?\s*(.+?)(?:这几列|这些列|列|即可|就行|$|，|。)'
        match = re.search(pattern, request, re.I)
        if not match:
            continue
        tokens = _split_columns(match.group(1))
        resolved = _resolve_all(tokens, header)
        if not resolved:
            continue
        is_drop = verb is _DROP_VERBS
        flag = '--not-columns' if is_drop else '--columns'
        return _plan('csvcut', [flag, ','.join(resolved)], f'rule: {"drop" if is_drop else "keep"} columns {resolved}')
    return None


def plan_filter(request, header):
    """Handle ``<column> <comparator> <value>`` filters."""
    for pattern, op in _COMPARATORS:
        match = re.search(rf'(.+?)\s*(?:{pattern})\s*(.+)', request, re.I)
        if not match:
            continue
        column_token = _strip_quotes(match.group(1))
        column_token = _LEADING_NOISE.sub('', column_token).strip()
        column = resolve_column(column_token, header) if column_token else None
        if column is None:
            continue

        value = _strip_quotes(match.group(2))
        value = _clean_literal(value)
        if not value:
            return None

        number = _parse_number(value)

        if op in ('gt', 'ge', 'lt', 'le') and number is not None:
            # Only SQL can express an ordered comparison on real numbers.
            sql = (
                f'SELECT * FROM data WHERE {_quote_identifier(column)} '
                f'{_OP_SQL[op]} {number}'
            )
            return _plan('csvsql', ['--tables', 'data', '--query', sql], f'rule: numeric filter {column} {_OP_SQL[op]} {number} (via in-memory SQLite)')

        if op in ('eq', 'ne') and number is not None:
            sql = (
                f'SELECT * FROM data WHERE {_quote_identifier(column)} '
                f'{_OP_SQL[op]} {number}'
            )
            return _plan('csvsql', ['--tables', 'data', '--query', sql], f'rule: numeric comparison {column} {_OP_SQL[op]} {number}')

        if op == 'ne':
            return _plan('csvgrep', ['--columns', column, '--invert-match', '--match', value], f'rule: exclude {column} == {value!r}')

        if op == 'eq':
            # Exact equality on text: anchor the regex so "US" does not match
            # "USA". csvkit passes this straight to re.search.
            return _plan('csvgrep', ['--columns', column, '--regex', rf'^\s*{re.escape(value)}\s*$'], f'rule: {column} equals {value!r}')

        if op in ('gt', 'ge', 'lt', 'le'):
            # Ordered comparison against a non-number cannot be expressed at
            # all. Refuse instead of silently degrading to substring matching.
            return None

        return _plan('csvgrep', ['--columns', column, '--match', value],
                     f'rule: {column} contains {value!r}')
    return None


def plan_sort(request, header):
    """Handle ``sort by <column>`` in either language."""
    match = re.search(
        r'(?:按|根据|依|sort\s+by|order\s+by|sorted\s+by)\s*(.+?)\s*'
        r'(?:排序|排列|升序|降序|倒序|sort|asc|desc)?\s*(?:即可|就行)?$',
        request, re.I)
    if not match:
        return None
    tokens = _split_columns(match.group(1))
    resolved = _resolve_all(tokens, header)
    if not resolved:
        return None
    args = ['--columns', ','.join(resolved)]
    direction = _sort_direction(request)
    if direction == 'desc':
        args.append('--reverse')
    return _plan('csvsort', args, f'rule: sort by {resolved} ({direction})')


_AGGREGATES = (
    (r'(?:平均|均值|average|mean|avg)', 'AVG'),
    (r'(?:合计|总和|求和|sum|total)', 'SUM'),
    (r'(?:最大|最高|max|largest|highest)', 'MAX'),
    (r'(?:最小|最低|min|smallest|lowest)', 'MIN'),
    (r'(?:数量|计数|多少行|多少条|count|how many)', 'COUNT'),
)


def plan_aggregate(request, header):
    """Handle ``<aggregate> of <column> [grouped by <column>]``.

    This is where a model is genuinely hard to replace: the phrasing space is
    huge. Offline we cover the common explicit shapes and refuse the rest.
    """
    aggregate = None
    for pattern, func in _AGGREGATES:
        if re.search(pattern, request, re.I):
            aggregate = func
            break
    if aggregate is None:
        return None

    # Grouping: "按 X" / "每个 X" / "group by X". The trailing verb list
    # excludes "统计" and "计算" on purpose - they name the aggregation
    # action, and consuming them here would swallow the target column
    # ("按 species 统计 sepal_length 的平均值" must group by species).
    group = None
    group_match = re.search(
        r'(?:按|根据|每个|各|per|group\s+by|grouped\s+by)\s*'
        r'(.+?)\s*(?:分组|group)?\s*(?:统计|汇总|计算|求|算|的|$|，|。)',
        request, re.I)
    if group_match:
        group_token = _LEADING_NOISE.sub('', _strip_quotes(group_match.group(1))).strip()
        # A grouping clause that resolves to nothing means we do not understand
        # the request; proceeding ungrouped would answer a different question.
        if group_token:
            group = resolve_column(group_token, header)
            if group is None:
                return None

    target = None
    target_match = re.search(
        r'([^\s，。的]+?)\s*的?\s*'
        r'(?:平均|均值|合计|总和|求和|最大|最高|最小|最低|数量|计数)',
        request, re.I)
    if target_match:
        target_token = _LEADING_NOISE.sub('', _strip_quotes(target_match.group(1))).strip()
        # Drop a leading grouping clause if the two spans overlapped.
        if group and target_token.startswith(group):
            target_token = target_token[len(group):].strip()
        target_token = re.sub(r'^(?:按|根据|每个|各)\s*', '', target_token).strip()
        if target_token:
            target = resolve_column(target_token, header)
            if target is None:
                return None

    if aggregate == 'COUNT':
        expression = 'COUNT(*)'
    else:
        if target is None:
            return None
        expression = f'{aggregate}({_quote_identifier(target)})'

    select_parts = []
    if group:
        select_parts.append(_quote_identifier(group))
    select_parts.append(f'{expression} AS {_alias(aggregate, target)}')
    sql = f'SELECT {", ".join(select_parts)} FROM data'
    if group:
        sql += f' GROUP BY {_quote_identifier(group)} ORDER BY {_quote_identifier(group)}'

    described = f'{aggregate.lower()}'
    if target:
        described += f' of {target}'
    if group:
        described += f' grouped by {group}'
    return _plan('csvsql', ['--tables', 'data', '--query', sql], f'rule: aggregate {described}')


def _alias(aggregate, target):
    base = {'AVG': 'avg', 'SUM': 'sum', 'MAX': 'max', 'MIN': 'min', 'COUNT': 'count'}[aggregate]
    if target:
        safe = re.sub(r'\W+', '_', str(target)).strip('_').lower()
        if safe:
            return f'{base}_{safe}'
    return base


def plan_stat(request, header):
    """Handle plain descriptive-statistics requests."""
    if not re.search(r'(?:统计|汇总|概况|描述|summary|describe|statistics?|stats)', request, re.I):
        return None
    # If the request names a column we cannot resolve, answering with a
    # whole-table summary would be silently wrong. Refuse instead.
    for token in re.findall(r'(?:按|根据|每个|各)\s*([^\s，。的]+)', request):
        if resolve_column(token, header) is None:
            return None
    args = []
    match = re.search(r'(?:统计|汇总|describe|stats?)\s*(?:一下)?\s*([^的，。]+?)\s*的',
                      request, re.I)
    if match:
        resolved = _resolve_all(_split_columns(match.group(1)), header)
        if resolved:
            args = ['--columns', ','.join(resolved)]
        elif match.group(1).strip():
            return None
    operation = None
    for key, flag in (('mean', '--mean'), ('median', '--median'), ('sum', '--sum'),
                      ('max', '--max'), ('min', '--min'), ('stdev', '--stdev'),
                      ('unique', '--unique'), ('count', '--count')):
        if re.search(key, request, re.I):
            operation = flag
            break
    if operation:
        args.append(operation)
    return _plan('csvstat', args, f'rule: statistics{"" if not operation else " " + operation}')


#: Order matters: the first rule that both matches and resolves wins.
#: Aggregate precedes filter so "按城市统计销售额的平均值" is not mistaken for a
#: filter on 城市.
_RULES = (plan_select, plan_aggregate, plan_filter, plan_sort, plan_stat)


def plan(request, table):
    """Return a :class:`Plan` or ``None`` when no rule applies confidently."""
    header = table.names
    if not header:
        return None
    for rule in _RULES:
        try:
            result = rule(request, header)
        except re.error:  # pragma: no cover - defensive
            continue
        if result is not None:
            return result
    return None


def capabilities():
    """Human-readable list of what works without a model."""
    return [
        'select columns: 只要 A 和 B / 去掉 A / select a,b / drop a',
        'filter rows: A 等于 x / A 包含 x / A 大于 100 / A 不等于 x',
        'sort rows: 按 A 排序 / 按 A 降序 / sort by a desc',
        'aggregate: 按城市统计销售额的平均值 / 求 price 的最大值 / 有多少行',
        'statistics: 统计 / 统计 A 的平均值 / describe / summary',
    ]
