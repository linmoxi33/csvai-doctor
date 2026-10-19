"""Pull structured fields out of free-text columns.

The classic case: a CSV has a human-readable column ("notes", "address",
"review") and you need a field that is *in there* but not separated. Regex is
hopeless across variation, and a model is genuinely good at it - with two
caveats this module enforces:

* **Only the requested fields are returned**, validated against the declared
  schema. No free-form keys leaking into your data.
* **Every value is traced back to the source text.** If the model returns a
  value that does not appear (even fuzzily) in the row it claims to describe,
  we drop it and count it. Silent invention in a data pipeline is the worst
  possible failure, so it is measured rather than assumed.

Offline, a small set of regex extractors covers the common cases (email, phone,
amount, date) so extraction is never entirely unavailable.
"""

import re
from dataclasses import dataclass, field

from .errors import PlanningError

#: Batch size for extraction calls. Each row is independent, so batching is
#: purely a cost/latency trade-off; 12 keeps prompts small enough that a single
#: malformed item does not poison a large batch.
BATCH_SIZE = 12

SYSTEM_PROMPT = """You extract structured fields from short text entries.

Reply with ONLY a JSON object:
{"rows": [{"index": <the integer index given>, "fields": {"<field>": "<value or null>"}}]}

Rules:
- Return one object per input entry, using the SAME index values.
- Use only the requested field names. Never add extra keys.
- Copy values from the text. Do not compute, normalise, translate or invent.
- If a field is absent from the text, use null. Do not guess.
- Keep values short: the exact substring that answers the field, nothing more.
"""


@dataclass
class ExtractionResult:
    rows: list                       # [{index, fields}]
    source: str
    requested: list
    verified: int = 0
    unverified: list = field(default_factory=list)

    def as_dict(self):
        data = {'source': self.source, 'fields': list(self.requested),
                'rows': len(self.rows), 'verified': self.verified}
        if self.unverified:
            data['unverified_count'] = len(self.unverified)
            data['unverified_examples'] = self.unverified[:3]
        return data


def _normalise(text):
    """Lowercase and collapse whitespace/punctuation for fuzzy containment."""
    return re.sub(r'[\s\u3000]+', '', str(text or '')).lower()


def appears_in(value, source):
    """Is *value* plausibly a substring of *source*?

    Compared after stripping whitespace and case, because models routinely
    tidy up spacing ("138 0013 8000" -> "13800138000") without inventing
    anything. Digit-for-digit differences still count as unverified.
    """
    if value is None:
        return True
    candidate = _normalise(value)
    if not candidate:
        return True
    return candidate in _normalise(source)


def build_prompt(texts, fields, column_name):
    lines = []
    for index, text in enumerate(texts):
        snippet = re.sub(r'\s+', ' ', str(text or '')).strip()[:400]
        lines.append(f'{index}: {snippet}')
    return (
        f'Column: {column_name}\n'
        f'Fields to extract: {", ".join(fields)}\n\n'
        f'Entries:\n' + '\n'.join(lines) +
        '\n\nReply with the JSON object only.'
    )


def validate(raw, texts, fields):
    """Convert a model reply into verified rows.

    Anything outside the declared field list, or with an index that was not
    sent, is discarded. Values not traceable to their source are recorded as
    unverified and blanked, so downstream code can never consume an invented
    value by accident.
    """
    payload = raw.get('rows')
    if not isinstance(payload, list):
        raise PlanningError('model returned no rows')

    allowed = list(fields)
    out = []
    unverified = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get('index'))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(texts):
            continue
        raw_fields = item.get('fields')
        if not isinstance(raw_fields, dict):
            continue

        clean = {}
        for name in allowed:
            value = raw_fields.get(name)
            if value is None:
                clean[name] = None
                continue
            text = str(value).strip()
            if not text or text.lower() in ('null', 'none', 'n/a', '-'):
                clean[name] = None
                continue
            # Provenance check: a value the model produced but which does not
            # appear in the source text is treated as an invention and blanked.
            if not appears_in(text, texts[index]):
                unverified.append({'index': index, 'field': name, 'value': text[:40]})
                clean[name] = None
                continue
            clean[name] = text[:200]

        out.append({'index': index, 'fields': clean})

    return out, unverified


def extract(texts, fields, column_name, provider, cache=None):
    """Extract *fields* from each text, batching and verifying."""
    if not fields:
        raise PlanningError('no fields requested')
    if not texts:
        return ExtractionResult(rows=[], source='model', requested=list(fields))

    rows = []
    unverified = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        user = build_prompt(batch, fields, column_name)
        raw = provider.complete_json(SYSTEM_PROMPT, user, cache=cache)
        parsed, bad = validate(raw, batch, fields)
        # Re-base the indices onto the whole column.
        for item in parsed:
            item['index'] += start
        rows.extend(parsed)
        for entry in bad:
            entry['index'] += start
        unverified.extend(bad)

    rows.sort(key=lambda item: item['index'])
    verified = sum(
        1 for item in rows for value in item['fields'].values() if value is not None
    )
    return ExtractionResult(rows=rows, source='model', requested=list(fields),
                            verified=verified, unverified=unverified)


# -- offline fallback ---------------------------------------------------

_OFFLINE_PATTERNS = {
    'email': re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),
    'phone': re.compile(r'(?<!\d)(\+?\d[\d\s()-]{6,18}\d)(?!\d)'),
    'amount': re.compile(r'(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?元)'),
    'url': re.compile(r'https?://\S+'),
    'date': re.compile(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}'),
}


def extract_offline(texts, fields, column_name=''):
    """Regex-only extraction for a handful of well-defined fields.

    Only fires for fields with a local pattern; anything else returns null,
    which is honest: "unknown" is a usable answer, a guess is not.
    """
    rows = []
    for index, text in enumerate(texts):
        values = {}
        for name in fields:
            pattern = _OFFLINE_PATTERNS.get(name.strip().lower())
            if pattern is None:
                values[name] = None
                continue
            match = pattern.search(str(text or ''))
            values[name] = match.group(0).strip() if match else None
        rows.append({'index': index, 'fields': values})

    unsupported = [n for n in fields if n.strip().lower() not in _OFFLINE_PATTERNS]
    return ExtractionResult(
        rows=rows,
        source='rules',
        requested=list(fields),
        verified=sum(
            1 for r in rows for v in r['fields'].values() if v is not None
        ),
        unverified=[],
    ), unsupported


def offline_supported_fields():
    return sorted(_OFFLINE_PATTERNS)
