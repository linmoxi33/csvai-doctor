"""Classification and extraction: verification over trust.

The central claim of these two modules is that a model's output is *checked*
against the data before it becomes a result. These tests pin that behaviour,
including the cases where the model is confidently wrong.
"""

import pytest

from csvai import classify, extract
from tests.fakes import make_provider, table_from

TABLE = table_from(
    ['city', 'email', 'amount', 'note'],
    [
        ['成都', 'a@example.com', '1280.00', '希望周五前送到'],
        ['北京', 'b.example.com', '320.00', ''],
        ['上海', 'c@example.com', '78.50', '发票抬头写公司名'],
    ],
)


# -- classification, model path ----------------------------------------

def test_model_labels_are_applied():
    provider, _ = make_provider([{
        'columns': [
            {'name': 'city', 'label': 'city', 'confidence': 0.9, 'reason': 'city names'},
            {'name': 'email', 'label': 'email', 'confidence': 0.95, 'reason': 'addresses'},
            {'name': 'amount', 'label': 'currency_amount', 'confidence': 0.9,
             'reason': 'money'},
            {'name': 'note', 'label': 'free_text', 'confidence': 0.8, 'reason': 'prose'},
        ]
    }])
    report = classify.classify(TABLE, provider)
    labels = {v.name: v.label for v in report.verdicts}
    assert labels['city'] == 'city'
    assert labels['note'] == 'free_text'
    assert report.source == 'model'


def test_conformance_is_measured_not_trusted():
    # The model claims all three values are emails; only two are. The report
    # must show the measured 67%, not the model's 95% confidence.
    provider, _ = make_provider([{
        'columns': [{'name': 'email', 'label': 'email', 'confidence': 0.95}]
    }])
    report = classify.classify(TABLE, provider)
    verdict = report.verdicts[0]
    assert verdict.conformance == pytest.approx(2 / 3, abs=0.01)
    assert verdict.offending == ['b.example.com']
    assert verdict.trustworthy is True   # 67% still clears the 60% bar


def test_low_conformance_marks_the_verdict_untrustworthy():
    table = table_from(['x'], [['nonsense'], ['more nonsense'], ['???']])
    provider, _ = make_provider([{
        'columns': [{'name': 'x', 'label': 'email', 'confidence': 0.99}]
    }])
    report = classify.classify(table, provider)
    assert report.verdicts[0].conformance == 0.0
    assert report.verdicts[0].trustworthy is False


def test_unknown_label_from_model_is_coerced():
    provider, _ = make_provider([{
        'columns': [{'name': 'city', 'label': 'geopolitical_region', 'confidence': 0.9}]
    }])
    report = classify.classify(TABLE, provider)
    assert report.verdicts[0].label == 'unknown'


def test_model_cannot_invent_a_column():
    provider, _ = make_provider([{
        'columns': [
            {'name': 'city', 'label': 'city', 'confidence': 0.9},
            {'name': 'profit', 'label': 'currency_amount', 'confidence': 0.9},
        ]
    }])
    report = classify.classify(TABLE, provider)
    assert [v.name for v in report.verdicts] == ['city']


def test_confidence_is_clamped():
    provider, _ = make_provider([{
        'columns': [{'name': 'city', 'label': 'city', 'confidence': 7.5}]
    }])
    report = classify.classify(TABLE, provider)
    assert report.verdicts[0].confidence == 1.0


def test_model_returning_nothing_raises_for_fallback():
    provider, _ = make_provider([{'columns': []}])
    with pytest.raises(ValueError):
        classify.classify(TABLE, provider)


# -- classification, offline path --------------------------------------

def test_offline_classification_uses_value_patterns():
    report = classify.classify_offline(TABLE)
    labels = {v.name: v.label for v in report.verdicts}
    assert labels['email'] == 'email'
    assert labels['amount'] == 'currency_amount'
    assert report.source == 'rules'


def test_offline_says_unknown_rather_than_guessing():
    report = classify.classify_offline(TABLE)
    labels = {v.name: v.label for v in report.verdicts}
    # 'note' is short-ish prose; the offline path must not claim to know it is
    # free_text with high confidence, and must not invent a domain label.
    assert labels['note'] in ('free_text', 'unknown')


def test_offline_does_not_mistake_dates_for_phones():
    # Regression: a 7-20 digit pattern matched "2026-03-02" and turned every
    # date column into a phone column.
    table = table_from(['d'], [['2026-03-02'], ['2026-03-03'], ['2026-03-04']])
    report = classify.classify_offline(table)
    assert report.verdicts[0].label == 'date'


def test_offline_handles_empty_samples():
    table = table_from(['x'], [[''], ['']])
    report = classify.classify_offline(table)
    assert report.verdicts[0].label in ('unknown', 'category')


# -- extraction, model path --------------------------------------------

TEXTS = [
    '希望周五前送到，电话 13800138000',
    '发票抬头写公司名，金额 1280.00 元',
    '',
]


def test_extraction_returns_declared_fields():
    provider, _ = make_provider([{
        'rows': [
            {'index': 0, 'fields': {'phone': '13800138000', 'amount': None}},
            {'index': 1, 'fields': {'phone': None, 'amount': '1280.00 元'}},
            {'index': 2, 'fields': {'phone': None, 'amount': None}},
        ]
    }])
    result = extract.extract(TEXTS, ['phone', 'amount'], 'note', provider)
    assert result.rows[0]['fields']['phone'] == '13800138000'
    assert result.rows[1]['fields']['amount'] == '1280.00 元'
    assert result.verified == 2


def test_invented_values_are_discarded():
    # "99999" is nowhere in the source text, so it must not survive.
    provider, _ = make_provider([{
        'rows': [{'index': 0, 'fields': {'phone': '99999', 'amount': None}}]
    }])
    result = extract.extract(TEXTS[:1], ['phone', 'amount'], 'note', provider)
    assert result.rows[0]['fields']['phone'] is None
    assert len(result.unverified) == 1
    assert result.unverified[0]['value'] == '99999'


def test_extra_fields_from_model_are_dropped():
    provider, _ = make_provider([{
        'rows': [{'index': 0, 'fields': {'phone': '13800138000', 'secret': 'leak'}}]
    }])
    result = extract.extract(TEXTS[:1], ['phone'], 'note', provider)
    assert 'secret' not in result.rows[0]['fields']


def test_out_of_range_index_is_dropped():
    provider, _ = make_provider([{
        'rows': [{'index': 99, 'fields': {'phone': 'x'}}]
    }])
    result = extract.extract(TEXTS[:1], ['phone'], 'note', provider)
    assert result.rows == []


def test_null_like_strings_become_none():
    provider, _ = make_provider([{
        'rows': [{'index': 0, 'fields': {'phone': 'null', 'amount': 'N/A'}}]
    }])
    result = extract.extract(TEXTS[:1], ['phone', 'amount'], 'note', provider)
    assert result.rows[0]['fields'] == {'phone': None, 'amount': None}


def test_whitespace_differences_are_still_considered_present():
    # Models routinely tidy "138 0013 8000" into "13800138000" without inventing.
    provider, _ = make_provider([{
        'rows': [{'index': 0, 'fields': {'phone': '13800138000'}}]
    }])
    result = extract.extract(['电话 138 0013 8000'], ['phone'], 'note', provider)
    assert result.rows[0]['fields']['phone'] == '13800138000'
    assert result.unverified == []


def test_batching_rebases_indices():
    texts = [f'行 {i} 电话 1380013800{i}' for i in range(8)]
    texts += [f'行 {i} 电话 1390013900{i}' for i in range(8)]
    assert len(texts) > extract.BATCH_SIZE

    first = {'rows': [{'index': i, 'fields': {'phone': None}}
                      for i in range(extract.BATCH_SIZE)]}
    second = {'rows': [{'index': i, 'fields': {'phone': None}}
                       for i in range(len(texts) - extract.BATCH_SIZE)]}
    provider, transport = make_provider([first, second])
    result = extract.extract(texts, ['phone'], 'note', provider)
    assert len(transport.calls) == 2
    assert [r['index'] for r in result.rows] == list(range(len(texts)))


# -- extraction, offline path ------------------------------------------

def test_offline_extraction_finds_patterns():
    result, unsupported = extract.extract_offline(TEXTS, ['phone', 'amount'])
    assert result.rows[0]['fields']['phone'] == '13800138000'
    assert result.rows[1]['fields']['amount'] == '1280.00 元'
    assert unsupported == []


def test_offline_extraction_reports_unsupported_fields():
    result, unsupported = extract.extract_offline(TEXTS, ['phone', 'sentiment'])
    assert unsupported == ['sentiment']
    assert all(r['fields']['sentiment'] is None for r in result.rows)


def test_offline_supported_fields_are_listed():
    assert 'email' in extract.offline_supported_fields()
    assert 'phone' in extract.offline_supported_fields()
