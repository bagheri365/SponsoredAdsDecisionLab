import numpy as np
import pytest

from sponsored_ads.behavioral_retrieval import BehavioralIndex, ObservedAd, RetrievedAd
from sponsored_ads.candidate_preranking import assemble_candidate_rows, prerank_candidates
from sponsored_ads.ctr_contract import make_contract


@pytest.fixture
def setup():
    index = BehavioralIndex([
        ObservedAd('a', 'cat', 'brand', 10),
        ObservedAd('b', 'cat', 'brand', 9),
        ObservedAd('c', 'cat', 'brand', 8),
    ])
    candidates = [RetrievedAd(x, float(i), ('affinity',)) for i, x in enumerate(('b', 'a', 'c'))]
    contract = make_contract(('userid', 'adgroup_id', 'cate_id', 'brand', 'price'), {
        'userid': {'user': 0}, 'adgroup_id': {'a': 0, 'b': 1, 'c': 2},
        'cate_id': {'cat': 0}, 'brand': {'brand': 0},
    })
    rows = {x: {'userid': 'user', 'adgroup_id': x, 'cate_id': 'cat',
                'brand': 'brand', 'price': '2.0'} for x in ('a', 'b', 'c')}
    return index, candidates, contract, rows


class StubModel:
    def predict(self, matrix):
        # All ties: ranking must use adgroup_id, not input order.
        return np.full(len(matrix), 0.3)


def test_deterministic_ties_and_attribution(setup):
    index, candidates, contract, rows = setup
    result = prerank_candidates(candidates, index, rows, StubModel(), contract, top_k=2)
    assert [r.adgroup_id for r in result.ranked] == ['a', 'b']
    assert result.ranked[0].sources == ('affinity',)
    assert result.input_candidates == 3 and result.unscorable == ()


def test_missing_row_fails_closed(setup):
    index, candidates, contract, rows = setup
    del rows['a']
    with pytest.raises(ValueError, match='missing feature row'):
        prerank_candidates(candidates, index, rows, StubModel(), contract, top_k=2)


def test_explicit_skip_reports_missing_and_incomplete(setup):
    index, candidates, contract, rows = setup
    del rows['a']
    del rows['c']['price']
    result = prerank_candidates(candidates, index, rows, StubModel(), contract,
                                top_k=10, missing_policy='skip')
    assert [r.adgroup_id for r in result.ranked] == ['b']
    assert result.unscorable == ('a', 'c')


def test_unknown_category_preserves_m05a_contract(setup):
    index, candidates, contract, rows = setup
    rows['a']['userid'] = 'unseen'
    selected, prepared, _ = assemble_candidate_rows(candidates, index, rows, contract)
    from sponsored_ads.ctr_contract import transform_rows
    encoded = transform_rows(prepared, contract)
    assert encoded[[c.adgroup_id for c in selected].index('a'), 0] == -1


def test_identity_mismatch_rejected_even_if_skipping(setup):
    index, candidates, contract, rows = setup
    rows['a']['adgroup_id'] = 'b'
    with pytest.raises(ValueError, match='ID mismatch'):
        prerank_candidates(candidates, index, rows, StubModel(), contract,
                           top_k=2, missing_policy='skip')


def test_catalog_mismatch_rejected(setup):
    index, candidates, contract, rows = setup
    rows['a']['brand'] = 'different'
    with pytest.raises(ValueError, match='catalog brand mismatch'):
        prerank_candidates(candidates, index, rows, StubModel(), contract, top_k=2)


def test_duplicate_candidate_rejected(setup):
    index, candidates, contract, rows = setup
    with pytest.raises(ValueError, match='duplicate'):
        prerank_candidates(candidates[:1] * 2, index, rows, StubModel(), contract, top_k=2)


def test_nonfinite_model_predictions_rejected(setup):
    index, candidates, contract, rows = setup
    class BadModel:
        def predict(self, matrix):
            return np.array([np.nan] * len(matrix))
    with pytest.raises(ValueError, match='nonfinite'):
        prerank_candidates(candidates, index, rows, BadModel(), contract, top_k=2)


def test_invalid_top_k_and_missing_policy(setup):
    index, candidates, contract, rows = setup
    with pytest.raises(ValueError, match='top_k'):
        prerank_candidates(candidates, index, rows, StubModel(), contract, top_k=-1)
    with pytest.raises(ValueError, match='missing_policy'):
        prerank_candidates(candidates, index, rows, StubModel(), contract,
                           top_k=1, missing_policy='fill_zero')


def test_empty_candidates(setup):
    index, _, contract, rows = setup
    result = prerank_candidates([], index, rows, StubModel(), contract, top_k=5)
    assert result.ranked == () and result.unscorable == ()
