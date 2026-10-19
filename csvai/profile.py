"""Build a compact, token-efficient profile of a CSV file.

The whole point of an AI assist here is that the model sees the *data*, not just
the header. But sending a 200k-row file is both expensive and pointless: the
shape of a column is decided by the first few dozen rows.

So we send a fixed-size sample plus deterministic statistics computed locally.
That split matters - anything arithmetic is done in Python, so the model can
never be the reason a count is wrong.
"""

import csv
import io
from dataclasses import dataclass, field

from .errors import CsvaiError

#: Hard cap on characters of any single cell we echo to the model. A runaway
#: blob column should not eat the whole context window.
MAX_CELL_CHARS = 120
#: Cap on columns profiled in full detail; beyond this we only keep names.
MAX_PROFILED_COLUMNS = 80


@dataclass
class ColumnProfile:
    name: str
    index: int
    samples: list = field(default_factory=list)
    nulls: int = 0
    distinct: int = 0
    numeric_ratio: float = 0.0
    date_ratio: float = 0.0
    mean_length: float = 0.0
    looks_identifier: bool = False
    looks_free_text: bool = False
    looks_like_category: bool = False

    def as_prompt_dict(self):
        data = {
            'name': self.name,
            'index_1_based': self.index + 1,
            'samples': self.samples,
            'null_count': self.nulls,
            'distinct_in_sample': self.distinct,
        }
        if self.numeric_ratio >= 0.8:
            data['type'] = 'numeric'
        elif self.date_ratio >= 0.8:
            data['type'] = 'date'
        elif self.looks_free_text:
            data['type'] = 'free_text'
        elif self.looks_like_category:
            data['type'] = 'category'
        else:
            data['type'] = 'text'
        return data


@dataclass
class TableProfile:
    path: str
    columns: list
    row_count: int
    sample_rows: int
    delimiter: str = ','
    encoding: str = 'utf-8'

    @property
    def names(self):
        return [c.name for c in self.columns]

    def as_prompt_dict(self):
        return {
            'file': self.path,
            'row_count': self.row_count,
            'sampled_rows': self.sample_rows,
            'column_count': len(self.columns),
            'columns': [c.as_prompt_dict() for c in self.columns[:MAX_PROFILED_COLUMNS]],
        }

    def header_block(self, limit=60):
        """A terse one-line-per-column rendering used inside prompts."""
        lines = []
        for column in self.columns[:limit]:
            kind = 'text'
            if column.numeric_ratio >= 0.8:
                kind = 'number'
            elif column.date_ratio >= 0.8:
                kind = 'date'
            sample = ', '.join(repr(s) for s in column.samples[:3])
            lines.append(f'- {column.name!r} ({kind}) e.g. {sample}')
        if len(self.columns) > limit:
            lines.append(f'- ... and {len(self.columns) - limit} more columns')
        return '\n'.join(lines)


def _is_number(text):
    try:
        float(text.replace(',', '').replace('%', '').strip())
        return True
    except (TypeError, ValueError):
        return False


def _is_date(text):
    import re

    if re.fullmatch(r'\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?', text.strip()):
        return True
    if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}', text.strip()):
        return True
    return False


def profile_rows(header, rows, path='<stdin>', delimiter=',', encoding='utf-8',
                 total_rows=None, sample_size=40):
    """Profile already-parsed rows.

    Split from :func:`profile_file` so that callers holding a parsed reader
    (like csvkit utilities) can reuse it without a second file pass.
    """
    rows = list(rows)
    sampled = rows[:sample_size]
    columns = []
    for index, name in enumerate(header):
        values = []
        nulls = 0
        for row in sampled:
            raw = row[index] if index < len(row) else ''
            if raw is None or str(raw).strip() == '':
                nulls += 1
                continue
            values.append(str(raw)[:MAX_CELL_CHARS])
        numeric = sum(1 for v in values if _is_number(v))
        dates = sum(1 for v in values if _is_date(v))
        distinct = len(set(values))
        mean_length = (sum(len(v) for v in values) / len(values)) if values else 0.0
        denominator = max(1, len(values))
        columns.append(
            ColumnProfile(
                name=name,
                index=index,
                samples=values[:5],
                nulls=nulls,
                distinct=distinct,
                numeric_ratio=numeric / denominator,
                date_ratio=dates / denominator,
                mean_length=mean_length,
                looks_identifier=bool(values) and distinct == len(values) and mean_length < 40,
                looks_free_text=mean_length > 40,
                # A category needs evidence of repetition. With fewer than four
                # non-null values every column looks like "a few distinct
                # values", so a tiny sample must not be called categorical -
                # that produced confident-sounding 'category' labels for text
                # columns in three-row files.
                looks_like_category=(
                    len(values) >= 4
                    and 1 < distinct <= max(2, len(values) // 2)
                    and mean_length <= 40
                ),
            )
        )

    return TableProfile(
        path=path,
        columns=columns,
        row_count=total_rows if total_rows is not None else len(rows),
        sample_rows=len(sampled),
        delimiter=delimiter,
        encoding=encoding,
    )


def profile_file(path, sample_size=40, encoding='utf-8-sig', delimiter=None):
    """Profile a CSV on disk without loading it all into memory twice.

    Counting rows still requires one full pass, but we stop retaining data
    after the sample, so peak memory stays proportional to the sample.
    """
    try:
        handle = open(path, 'r', encoding=encoding, newline='')
    except OSError as exc:
        raise CsvaiError(f'cannot read {path}: {exc}') from None

    with handle:
        sample_text = handle.read(64 * 1024)
        handle.seek(0)
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(sample_text[:4096]).delimiter
            except csv.Error:
                delimiter = ','
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return TableProfile(path=str(path), columns=[], row_count=0, sample_rows=0,
                                delimiter=delimiter, encoding=encoding)
        sampled = []
        total = 0
        for row in reader:
            total += 1
            if len(sampled) < sample_size:
                sampled.append(row)

    return profile_rows(header, sampled, path=str(path), delimiter=delimiter,
                        encoding=encoding, total_rows=total, sample_size=sample_size)


def profile_text(text, sample_size=40, encoding='utf-8', delimiter=','):
    """Profile from an in-memory string (used by tests and stdin paths)."""
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        return TableProfile(path='<stdin>', columns=[], row_count=0, sample_rows=0,
                            delimiter=delimiter, encoding=encoding)
    sampled = []
    total = 0
    for row in reader:
        total += 1
        if len(sampled) < sample_size:
            sampled.append(row)
    return profile_rows(header, sampled, path='<stdin>', delimiter=delimiter,
                        encoding=encoding, total_rows=total, sample_size=sample_size)
