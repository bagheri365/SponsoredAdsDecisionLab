import pytest

from sponsored_ads.auction_pricing import price_single_slot
from sponsored_ads.budget_ledger import (
    BudgetLedger, LedgerEvent, cpc_to_cents, finalize_auction_winner,
    reserve_auction_winner,
)
from sponsored_ads.models import AdCandidate


def outcome(policy='score_second_price'):
    return price_single_slot([
        AdCandidate('a', 'adv-a', .2, 2.0),
        AdCandidate('b', 'adv-b', .1, 3.0),
    ], policy=policy)


def test_initial_snapshot():
    assert BudgetLedger({'a': 100}).snapshot('a').available_cents == 100


def test_reserve_changes_available_not_spent():
    ledger = BudgetLedger({'a': 100})
    ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=80)
    assert (ledger.snapshot('a').reserved_cents, ledger.snapshot('a').spent_cents,
            ledger.snapshot('a').available_cents) == (80, 0, 20)


def test_settle_releases_unused_hold():
    ledger = BudgetLedger({'a': 100})
    ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=80)
    ledger.settle(event_id='e2', reservation_id='r', charged_cents=25)
    assert (ledger.snapshot('a').reserved_cents, ledger.snapshot('a').spent_cents,
            ledger.snapshot('a').available_cents) == (0, 25, 75)


def test_release_restores_budget():
    ledger = BudgetLedger({'a': 100})
    ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=80)
    ledger.release(event_id='e2', reservation_id='r')
    assert ledger.snapshot('a').available_cents == 100


def test_no_overspend_and_atomic_failure():
    ledger = BudgetLedger({'a': 100})
    ledger.reserve(event_id='e1', reservation_id='r1', advertiser_id='a', max_charge_cents=80)
    with pytest.raises(ValueError, match='insufficient'):
        ledger.reserve(event_id='e2', reservation_id='r2', advertiser_id='a', max_charge_cents=21)
    assert len(ledger.events) == 1
    assert ledger.snapshot('a').available_cents == 20


def test_exact_event_retry_is_idempotent():
    ledger = BudgetLedger({'a': 100})
    first = ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=50)
    assert ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=50) == first
    assert len(ledger.events) == 1
    ledger.settle(event_id='e2', reservation_id='r', charged_cents=25)
    assert ledger.settle(event_id='e2', reservation_id='r', charged_cents=25).settled_cents == 25
    assert len(ledger.events) == 2


def test_conflicting_event_id_rejected():
    ledger = BudgetLedger({'a': 100})
    ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=50)
    with pytest.raises(ValueError, match='conflicting'):
        ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=60)


def test_duplicate_reservation_id_rejected():
    ledger = BudgetLedger({'a': 100})
    ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=50)
    with pytest.raises(ValueError, match='already used'):
        ledger.reserve(event_id='e2', reservation_id='r', advertiser_id='a', max_charge_cents=50)


def test_double_finalization_rejected():
    ledger = BudgetLedger({'a': 100})
    ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=50)
    ledger.release(event_id='e2', reservation_id='r')
    with pytest.raises(ValueError, match='finalized'):
        ledger.settle(event_id='e3', reservation_id='r', charged_cents=10)


def test_settle_above_hold_rejected():
    ledger = BudgetLedger({'a': 100})
    ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=50)
    with pytest.raises(ValueError, match='exceeds'):
        ledger.settle(event_id='e2', reservation_id='r', charged_cents=51)
    assert len(ledger.events) == 1


@pytest.mark.parametrize('value', [-1, 1.0, True, '10', None])
def test_invalid_minor_units(value):
    with pytest.raises(ValueError):
        BudgetLedger({'a': value})


@pytest.mark.parametrize('value', [-1, float('inf'), float('nan'), True, '2'])
def test_invalid_cpc(value):
    with pytest.raises(ValueError):
        cpc_to_cents(value)


def test_conservative_rounding():
    assert cpc_to_cents(1.5000000000000002) == 151
    assert cpc_to_cents(1.5) == 150
    assert cpc_to_cents(0) == 0


def test_replay_reconstructs_state():
    ledger = BudgetLedger({'a': 100, 'b': 300})
    ledger.reserve(event_id='e1', reservation_id='r1', advertiser_id='a', max_charge_cents=80)
    ledger.reserve(event_id='e2', reservation_id='r2', advertiser_id='b', max_charge_cents=200)
    ledger.settle(event_id='e3', reservation_id='r1', charged_cents=40)
    ledger.release(event_id='e4', reservation_id='r2')
    replayed = BudgetLedger.replay({'a': 100, 'b': 300}, ledger.events)
    assert replayed.events == ledger.events
    assert replayed.snapshot('a') == ledger.snapshot('a')
    assert replayed.snapshot('b') == ledger.snapshot('b')


def test_tampered_release_event_rejected():
    events = (LedgerEvent('e1', 'reserve', 'r', 'a', 20),
              LedgerEvent('e2', 'release', 'r', 'a', 5))
    with pytest.raises(ValueError, match='invalid release'):
        BudgetLedger.replay({'a': 100}, events)


@pytest.mark.parametrize('policy', ['first_price', 'score_second_price'])
def test_auction_integration_click(policy):
    auction = outcome(policy)
    ledger = BudgetLedger({'adv-a': 250})
    reserve_auction_winner(ledger, auction, event_id='e1', reservation_id='r')
    assert ledger.snapshot('adv-a').reserved_cents == 200
    result = finalize_auction_winner(ledger, auction, event_id='e2', reservation_id='r', clicked=True)
    assert result.settled_cents == cpc_to_cents(auction.charged_cpc_if_clicked)
    assert ledger.snapshot('adv-a').reserved_cents == 0
    assert ledger.snapshot('adv-a').available_cents >= 0


def test_no_click_releases_reservation():
    auction = outcome()
    ledger = BudgetLedger({'adv-a': 200})
    reserve_auction_winner(ledger, auction, event_id='e1', reservation_id='r')
    finalize_auction_winner(ledger, auction, event_id='e2', reservation_id='r', clicked=False)
    assert ledger.snapshot('adv-a').available_cents == 200
    assert ledger.snapshot('adv-a').spent_cents == 0


def test_mismatched_winner_rejected():
    auction = outcome()
    ledger = BudgetLedger({'adv-a': 300, 'adv-b': 300})
    ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='adv-b', max_charge_cents=200)
    with pytest.raises(ValueError, match='advertiser'):
        finalize_auction_winner(ledger, auction, event_id='e2', reservation_id='r', clicked=True)


def test_zero_budget_zero_reservation():
    ledger = BudgetLedger({'a': 0})
    ledger.reserve(event_id='e1', reservation_id='r', advertiser_id='a', max_charge_cents=0)
    ledger.settle(event_id='e2', reservation_id='r', charged_cents=0)
    assert ledger.snapshot('a').available_cents == 0


def test_unknown_advertiser_rejected():
    with pytest.raises(ValueError, match='unknown advertiser'):
        BudgetLedger({'a': 100}).reserve(event_id='e1', reservation_id='r', advertiser_id='b', max_charge_cents=0)
