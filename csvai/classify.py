"""Semantic column classification.

This is the task that motivated the whole layer. Heuristics can tell you a
column is "text"; they cannot tell you it is a *company name*, or that it is
mostly emails but 12% of rows are junk. A model can, and - crucially - it can
explain *which* rows it is unsure about.

Three properties make this trustworthy rather than decorative:

* **The model sees real values**, sampled locally, never the whole file.
* **Labels are constrained** to a closed vocabulary, so output is filterable.
* **Conformance is measured in Python**, not trusted from the model. If the
  model says "this is an email column", we check every sampled value against a
  local pattern and report the actual share that matches. When the model and
  the data disagree, the data wins and we say so.
"""

import re
from dataclasses import dataclass, field

#: Closed label vocabulary. Kept small on purpose: a free-text taxonomy produces
#: labels that cannot be compared across columns or runs.
LABELS = (
    'person_name',
    'organization',
    'email',
    'phone',
    'url',
    'address',
    'city',
    'country',
    'postal_code',
    'date',
    'datetime',
    'currency_amount',
    'quantity',
    'percentage',
    'identifier',
    'category',
    'free_text',
    'product',
    'title',
    'code',
    'unknown',
)

#: Local validators. These are the ground truth used to score a model's claim,
#: and they are also what the offline fallback uses on its own.
#:
#: The phone pattern is deliberately narrow: an early version accepted any
#: 7-20 run of digits, spaces and dashes, which swallowed ISO dates
#: ("2026-03-02") and mislabelled a date column as phone numbers. A phone must
#: carry at least 7 actual digits and must not look like a calendar date.
_PATTERNS = {
    'email': re.compile(r'^[\w.+-]+@[\w-]+\.[\w.-]+$'),
    'phone': re.compile(
        r'^(?!\d{4}-\d{1,2}-\d{1,2}$)'      # not an ISO-ish date
        r'\+?[\d][\d\s()\-.]{6,19}$'        # starts with a digit
    ),
    'url': re.compile(r'^https?://\S+$', re.I),
    'postal_code': re.compile(r'^\d{5}(-\d{4})?$|^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$', re.I),
    'date': re.compile(r'^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{4}$|^\d{4}/\d{1,2}/\d{1,2}$'),
    'datetime': re.compile(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}'),
    'currency_amount': re.compile(
        r'^[$€£¥]\s?-?[\d,]+(?:\.\d+)?$'          # symbol-prefixed
        r'|^-?[\d,]+(?:\.\d+)?\s?(?:USD|EUR|CNY|RMB|元)$'  # suffixed
        r'|^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$'      # thousands-grouped: 1,280.00
        r'|^-?\d+\.\d{2}$'                        # exactly two decimals: 89.90
    ),
    'percentage': re.compile(r'^-?[\d.]+\s?%$'),
    'identifier': re.compile(r'^[A-Z]{1,6}[-_]?\d{2,}$'),
}

#: Checked before the keyed patterns above, because these overlap. A date and a
#: bare decimal number are both "digits and separators"; the more specific
#: interpretation has to win or every date column becomes a phone column.
_PRIORITY = ('email', 'url', 'datetime', 'date', 'percentage', 'currency_amount',
             'postal_code', 'identifier', 'phone')

def _fullmatch(pattern, value):
    """Anchored match. ``re.match`` only pins the start, which let a trailing
    "元" or " 的行" slip past validators that looked anchored."""
    return pattern.fullmatch(value) is not None


SYSTEM_PROMPT_HEAD = """You classify the columns of a CSV file by what their values actually are.

Reply with ONLY a JSON object:
{"columns": [{"name": "<exact column name>", "label": "<one label>", "confidence": 0.0-1.0, "reason": "<10 words max>"}]}

The allowed labels are: """

SYSTEM_PROMPT = (
    SYSTEM_PROMPT_HEAD
    + ', '.join(LABELS)
    + """

Rules:
- Use the EXACT column name as given. Do not rename, translate or reorder.
- Classify by the VALUES shown, not only by the column name. A column called
  "amount" holding dates is a date column.
- If the sample is too small or genuinely ambiguous, use "unknown" and a low
  confidence. Do not invent certainty.
- Do not add columns that were not given, and do not omit any.
"""
)


@dataclass
class ColumnVerdict:
    name: str
    label: str
    confidence: float
    reason: str = ''
    source: str = 'model'
    conformance: float = None       # measured share of values matching the label
    offending: list = field(default_factory=list)

    def as_dict(self):
        data = {
            'column': self.name,
            'label': self.label,
            'confidence': round(float(self.confidence), 2),
            'source': self.source,
        }
        if self.reason:
            data['reason'] = self.reason
        if self.conformance is not None:
            data['conformance'] = round(self.conformance, 2)
            if self.offending:
                data['nonconforming_examples'] = self.offending[:3]
        return data

    @property
    def trustworthy(self):
        """Low conformance means the label does not describe the data."""
        return self.conformance is None or self.conformance >= 0.6


@dataclass
class ClassificationReport:
    verdicts: list
    source: str
    note: str = ''

    def as_dict(self):
        return {
            'source': self.source,
            'note': self.note,
            'columns': [v.as_dict() for v in self.verdicts],
        }

    def labelled(self, label):
        return [v for v in self.verdicts if v.label == label]


def build_prompt(table, columns=None):
    target = table.columns if columns is None else [
        c for c in table.columns if c.name in set(columns)
    ]
    lines = []
    for column in target:
        samples = ', '.join(repr(s) for s in column.samples[:6]) or '(all empty)'
        lines.append(f'- {column.name!r}: {samples}')
    return (
        f'File: {table.path} ({table.row_count} rows)\n'
        f'Classify these {len(target)} columns:\n' + '\n'.join(lines) +
        '\n\nReply with the JSON object only.'
    )


def score(verdict, table):
    """Attach measured conformance to a verdict.

    Never raises: an unknown label simply gets no score, because inventing a
    number would be worse than admitting we did not check.
    """
    pattern = _PATTERNS.get(verdict.label)
    column = next((c for c in table.columns if c.name == verdict.name), None)
    if pattern is None or column is None or not column.samples:
        return verdict

    values = [v for v in column.samples if v and v.strip()]
    if not values:
        return verdict
    hits = sum(1 for v in values if _fullmatch(pattern, v.strip()))
    verdict.conformance = hits / len(values)
    if verdict.conformance < 1.0:
        verdict.offending = [v for v in values if not _fullmatch(pattern, v.strip())][:3]
    return verdict


def validate(raw, table):
    """Turn a model reply into verdicts, dropping anything malformed."""
    payload = raw.get('columns')
    if not isinstance(payload, list):
        return []

    header = set(table.names)
    seen = set()
    verdicts = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get('name')
        if not isinstance(name, str) or name not in header or name in seen:
            continue
        seen.add(name)
        label = str(item.get('label') or 'unknown').strip().lower()
        if label not in LABELS:
            label = 'unknown'
        try:
            confidence = float(item.get('confidence', 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))
        verdicts.append(
            ColumnVerdict(
                name=name,
                label=label,
                confidence=confidence,
                reason=str(item.get('reason') or '')[:80],
                source='model',
            )
        )
    return verdicts


def classify(table, provider, cache=None):
    """Classify via the model, then verify each verdict against the data."""
    raw = provider.complete_json(SYSTEM_PROMPT, build_prompt(table), cache=cache)
    verdicts = validate(raw, table)
    if not verdicts:
        raise ValueError('model returned no usable column verdicts')
    for verdict in verdicts:
        score(verdict, table)
    return ClassificationReport(verdicts=verdicts, source='model')


# -- offline fallback ---------------------------------------------------

_NAME_HINTS = (
    ('email', ('email', 'mail', '邮箱', '邮件')),
    ('phone', ('phone', 'tel', 'mobile', '手机', '电话')),
    ('url', ('url', 'link', 'website', '网址', '链接')),
    ('city', ('city', '城市')),
    ('country', ('country', '国家', 'nation')),
    ('person_name', ('name', '姓名', '人名', 'contact')),
    ('postal_code', ('zip', 'postal', '邮编')),
    ('date', ('date', '日期', 'time', '时间')),
    ('currency_amount', ('amount', 'price', 'cost', 'revenue', 'salary', '金额', '价格', '销售额')),
    ('quantity', ('qty', 'quantity', 'count', '数量')),
    ('percentage', ('pct', 'percent', 'rate', 'ratio', '比例', '率')),
    ('identifier', ('id', 'uuid', 'sku', '编号', '代码')),
    ('product', ('product', 'item', '商品', '产品')),
    ('title', ('title', 'headline', '标题')),
)


def classify_offline(table):
    """Label columns using value patterns first, then name hints.

    Value evidence outranks the column name, matching what we ask the model to
    do, so the two paths agree on the easy cases.
    """
    verdicts = []
    for column in table.columns:
        values = [v.strip() for v in column.samples if v and v.strip()]
        label, confidence, reason = 'unknown', 0.2, 'no clear signal in sample'

        if values:
            best, best_share = None, 0.0
            for candidate in _PRIORITY:
                pattern = _PATTERNS[candidate]
                share = sum(1 for v in values if _fullmatch(pattern, v)) / len(values)
                if share > best_share:
                    best, best_share = candidate, share
            # A pure-number column that matched "phone" by accident should not
            # become a phone column; 0.9 is the bar for a pattern-based call.
            if best and best_share >= 0.9:
                label, confidence = best, min(0.95, best_share)
                reason = f'{best_share:.0%} of sampled values match the {best} pattern'

        if label == 'unknown':
            lowered = column.name.lower()
            for candidate, hints in _NAME_HINTS:
                if any(hint in lowered for hint in hints):
                    label, confidence = candidate, 0.45
                    reason = f'column name suggests {candidate}'
                    break

        if label == 'unknown':
            if column.numeric_ratio >= 0.9:
                label, confidence, reason = 'quantity', 0.5, 'values are numeric'
            elif column.looks_free_text:
                label, confidence, reason = 'free_text', 0.5, 'long free-form values'
            elif column.looks_like_category:
                label, confidence, reason = 'category', 0.4, 'few repeated values'

        verdict = ColumnVerdict(name=column.name, label=label,
                                confidence=confidence, reason=reason, source='rules')
        verdicts.append(score(verdict, table))

    return ClassificationReport(
        verdicts=verdicts,
        source='rules',
        note='labels derived from local value patterns and column names; '
             'no semantic understanding of the domain',
    )
