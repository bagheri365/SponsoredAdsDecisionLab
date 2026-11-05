import numpy as np
import pytest

from sponsored_ads.behavioral_retrieval import BehavioralIndex, ObservedAd, RetrievedAd
from sponsored_ads.ctr_contract import make_contract
from sponsored_ads.preranking_evaluation import (
    EvaluationCase, evaluate_case, evaluate_cases, summarize_cases,
)


class Model:
    def predict(self, matrix):
        return np.where(matrix[:, 0] == 0, 0.9, 0.1)


@pytest.fixture
def sample():
    index = BehavioralIndex([ObservedAd('a', 'cat', 'brand', 3),
                             ObservedAd('b', 'cat', 'brand', 2),
                             ObservedAd('c', 'cat', 'brand', 1)])
    contract = make_contract(('adgroup_id', 'cate_id', 'brand', 'price'), {
        'adgroup_id': {'a': 0, 'b': 1, 'c': 2},
        'cate_id': {'cat': 0}, 'brand': {'brand': 0},
    })
    candidates = tuple(RetrievedAd(x, 1.0, ('affinity',)) for x in ('b', 'a', 'c'))
    rows = {x: {'adgroup_id': x, 'cate_id': 'cat', 'brand': 'brand', 'price': '1'}
            for x in ('a', 'b', 'c')}
    return index, contract, candidates, rows


def test_hit_and_mrr_with_full_coverage(sample):
    index, contract, candidates, rows = sample
    case = EvaluationCase(candidates, rows, {'a': 1, 'b': 0, 'c': 0})
    result = evaluate_case(case, index, Model(), contract, top_k=2)
    assert [r.adgroup_id for r in result.ranked] == ['a', 'b']
    assert result.hit_at_k == 1 and result.reciprocal_rank_at_k == 1
    assert result.scorable_count == 3 and result.unscorable == ()
    assert min(result.assembly_ms, result.inference_ms, result.sorting_ms) >= 0


def test_missing_positive_is_not_dropped_from_metric_denominator(sample):
    index, contract, candidates, rows = sample
    del rows['a']
    case = EvaluationCase(candidates, rows, {'a': 1, 'b': 0, 'c': 0})
    result = evaluate_case(case, index, Model(), contract, top_k=2)
    assert result.scorable_count == 2 and result.unscorable == ('a',)
    assert result.hit_at_k == 0 and result.reciprocal_rank_at_k == 0


def test_positive_at_second_rank(sample):
    index, contract, candidates, rows = sample
    case = EvaluationCase(candidates, rows, {'a': 0, 'b': 1, 'c': 0})
    result = evaluate_case(case, index, Model(), contract, top_k=2)
    assert result.hit_at_k == 1 and result.reciprocal_rank_at_k == 0.5


def test_top_k_truncation_not_counted_as_unscorable(sample):
    index, contract, candidates, rows = sample
    case = EvaluationCase(candidates, rows, {'a': 0, 'b': 0, 'c': 1})
    result = evaluate_case(case, index, Model(), contract, top_k=1)
    assert result.scorable_count == 3 and result.unscorable == ()
    assert result.hit_at_k == 0


@pytest.mark.parametrize('labels', [
    {'a': 1, 'b': 0}, {'a': 1, 'b': 0, 'c': 0, 'extra': 0},
    {'a': 0, 'b': 0, 'c': 0}, {'a': True, 'b': 0, 'c': 0},
    {'a': 1, 'b': -1, 'c': 0},
])
def test_invalid_labels_fail_closed(sample, labels):
    index, contract, candidates, rows = sample
    with pytest.raises(ValueError):
        evaluate_case(EvaluationCase(candidates, rows, labels), index, Model(), contract, top_k=2)


def test_duplicate_candidates_rejected(sample):
    index, contract, candidates, rows = sample
    case = EvaluationCase(candidates[:1]*2, rows, {'b': 1})
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_case(case, index, Model(), contract, top_k=1)


def test_empty_and_invalid_k(sample):
    index, contract, candidates, rows = sample
    with pytest.raises(ValueError, match='top_k'):
        evaluate_case(EvaluationCase(candidates, rows, {'a': 1, 'b': 0, 'c': 0}),
                      index, Model(), contract, top_k=0)
    with pytest.raises(ValueError, match='at least one'):
        summarize_cases([])


def test_summary_micro_coverage_macro_quality(sample):
    index, contract, candidates, rows = sample
    case1 = EvaluationCase(candidates, rows, {'a': 1, 'b': 0, 'c': 0})
    incomplete = dict(rows)
    del incomplete['a']
    case2 = EvaluationCase(candidates, incomplete, {'a': 1, 'b': 0, 'c': 0})
    results, summary = evaluate_cases([case1, case2], index, Model(), contract, top_k=2)
    assert len(results) == 2 and summary.cases == 2
    assert summary.candidates == 6 and summary.scorable == 5 and summary.unscorable == 1
    assert summary.hit_at_k == 0.5 and summary.mrr_at_k == 0.5


def test_no_scored_rows_still_reports_zero_hit(sample):
    index, contract, candidates, _ = sample
    case = EvaluationCase(candidates, {}, {'a': 1, 'b': 0, 'c': 0})
    result = evaluate_case(case, index, Model(), contract, top_k=3)
    assert result.scorable_count == 0 and result.hit_at_k == 0
    assert result.inference_ms >= 0


def test_bad_model_probability_rejected(sample):
    index, contract, candidates, rows = sample
    class InvalidModel:
        def predict(self, matrix):
            return np.full(len(matrix), 1.5)
    with pytest.raises(ValueError, match='probabilities'):
        evaluate_case(EvaluationCase(candidates, rows, {'a': 1, 'b': 0, 'c': 0}),
                      index, InvalidModel(), contract, top_k=2)
