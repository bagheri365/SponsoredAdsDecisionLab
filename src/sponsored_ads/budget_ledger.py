"""M06B-1 deterministic synthetic CPC budget ledger, using integer minor units.

Not a platform billing, pacing, concurrency, or payment-processing implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Literal

from .auction_pricing import AuctionOutcome


def _cents(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer number of cents')
    return value


def cpc_to_cents(value: float) -> int:
    """Conservatively round a finite nonnegative CPC *up* to the next cent.

    Decimal(str(value)) makes the float-to-decimal boundary explicit. This is
    a simulation convention; callers must not use floats for actual billing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('CPC must be a finite nonnegative number')
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('invalid CPC') from exc
    if not decimal.is_finite() or decimal < 0:
        raise ValueError('CPC must be finite and nonnegative')
    return int((decimal * 100).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class AccountSnapshot:
    advertiser_id: str
    budget_cents: int
    reserved_cents: int
    spent_cents: int
    available_cents: int


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    advertiser_id: str
    max_charge_cents: int
    state: Literal['reserved', 'settled', 'released']
    settled_cents: int = 0


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    action: Literal['reserve', 'settle', 'release']
    reservation_id: str
    advertiser_id: str
    amount_cents: int


class BudgetLedger:
    """In-memory single-threaded event ledger with idempotent event replay.

    Event IDs are globally unique. Exact event retries return the prior result;
    reuse with a different payload fails. A reservation ID can be used once.
    """

    def __init__(self, budgets_cents: dict[str, int]):
        if not isinstance(budgets_cents, dict):
            raise TypeError('budgets_cents must be a dict')
        self._budgets: dict[str, int] = {}
        for advertiser, budget in budgets_cents.items():
            if not isinstance(advertiser, str) or not advertiser:
                raise ValueError('advertiser IDs must be nonempty strings')
            self._budgets[advertiser] = _cents(budget, 'budget')
        self._reservations: dict[str, Reservation] = {}
        self._events: dict[str, LedgerEvent] = {}
        self._history: list[LedgerEvent] = []

    def snapshot(self, advertiser_id: str) -> AccountSnapshot:
        if advertiser_id not in self._budgets:
            raise ValueError('unknown advertiser')
        reserved = sum(r.max_charge_cents for r in self._reservations.values()
                       if r.advertiser_id == advertiser_id and r.state == 'reserved')
        spent = sum(r.settled_cents for r in self._reservations.values()
                    if r.advertiser_id == advertiser_id and r.state == 'settled')
        budget = self._budgets[advertiser_id]
        available = budget - reserved - spent
        if available < 0:
            raise AssertionError('ledger overspend invariant violated')
        return AccountSnapshot(advertiser_id, budget, reserved, spent, available)

    def reservation(self, reservation_id: str) -> Reservation:
        return self._reservations[reservation_id]

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._history)

    def _retry(self, event: LedgerEvent) -> bool:
        if not isinstance(event.event_id, str) or not event.event_id:
            raise ValueError('event_id must be nonempty')
        if not isinstance(event.reservation_id, str) or not event.reservation_id:
            raise ValueError('reservation_id must be nonempty')
        prior = self._events.get(event.event_id)
        if prior is not None:
            if prior != event:
                raise ValueError('event_id reused with conflicting payload')
            return True
        return False

    def _record(self, event: LedgerEvent) -> None:
        self._events[event.event_id] = event
        self._history.append(event)

    def reserve(self, *, event_id: str, reservation_id: str,
                advertiser_id: str, max_charge_cents: int) -> Reservation:
        amount = _cents(max_charge_cents, 'max_charge_cents')
        event = LedgerEvent(event_id, 'reserve', reservation_id, advertiser_id, amount)
        if self._retry(event):
            return self._reservations[reservation_id]
        if reservation_id in self._reservations:
            raise ValueError('reservation_id already used')
        if self.snapshot(advertiser_id).available_cents < amount:
            raise ValueError('insufficient available budget')
        result = Reservation(reservation_id, advertiser_id, amount, 'reserved')
        self._reservations[reservation_id] = result
        self._record(event)
        return result

    def settle(self, *, event_id: str, reservation_id: str,
               charged_cents: int) -> Reservation:
        amount = _cents(charged_cents, 'charged_cents')
        prior = self._reservations.get(reservation_id)
        if prior is None:
            raise ValueError('unknown reservation')
        event = LedgerEvent(event_id, 'settle', reservation_id, prior.advertiser_id, amount)
        if self._retry(event):
            return self._reservations[reservation_id]
        if prior.state != 'reserved':
            raise ValueError('reservation already finalized')
        if amount > prior.max_charge_cents:
            raise ValueError('settlement exceeds reservation')
        result = Reservation(reservation_id, prior.advertiser_id,
                             prior.max_charge_cents, 'settled', amount)
        self._reservations[reservation_id] = result
        self._record(event)
        return result

    def release(self, *, event_id: str, reservation_id: str) -> Reservation:
        prior = self._reservations.get(reservation_id)
        if prior is None:
            raise ValueError('unknown reservation')
        event = LedgerEvent(event_id, 'release', reservation_id, prior.advertiser_id, 0)
        if self._retry(event):
            return self._reservations[reservation_id]
        if prior.state != 'reserved':
            raise ValueError('reservation already finalized')
        result = Reservation(reservation_id, prior.advertiser_id,
                             prior.max_charge_cents, 'released')
        self._reservations[reservation_id] = result
        self._record(event)
        return result

    @classmethod
    def replay(cls, budgets_cents: dict[str, int], events: tuple[LedgerEvent, ...]) -> BudgetLedger:
        ledger = cls(budgets_cents)
        for event in events:
            if not isinstance(event, LedgerEvent):
                raise TypeError('events must be LedgerEvent instances')
            if event.action == 'reserve':
                ledger.reserve(event_id=event.event_id, reservation_id=event.reservation_id,
                               advertiser_id=event.advertiser_id, max_charge_cents=event.amount_cents)
            elif event.action == 'settle':
                result = ledger.reservation(event.reservation_id)
                if result.advertiser_id != event.advertiser_id:
                    raise ValueError('event advertiser mismatch')
                ledger.settle(event_id=event.event_id, reservation_id=event.reservation_id,
                              charged_cents=event.amount_cents)
            elif event.action == 'release':
                result = ledger.reservation(event.reservation_id)
                if result.advertiser_id != event.advertiser_id or event.amount_cents != 0:
                    raise ValueError('invalid release event')
                ledger.release(event_id=event.event_id, reservation_id=event.reservation_id)
            else:
                raise ValueError('unsupported ledger event action')
        return ledger


def reserve_auction_winner(ledger: BudgetLedger, outcome: AuctionOutcome,
                           *, event_id: str, reservation_id: str) -> Reservation:
    """Reserve the winner's maximum CPC bid, not the conditional payment."""
    if not isinstance(outcome, AuctionOutcome):
        raise TypeError('outcome must be AuctionOutcome')
    return ledger.reserve(event_id=event_id, reservation_id=reservation_id,
                          advertiser_id=outcome.advertiser_id,
                          max_charge_cents=cpc_to_cents(outcome.bid_cpc))


def finalize_auction_winner(ledger: BudgetLedger, outcome: AuctionOutcome,
                            *, event_id: str, reservation_id: str,
                            clicked: bool) -> Reservation:
    """Only an explicitly simulated click settles conditional CPC."""
    if not isinstance(outcome, AuctionOutcome):
        raise TypeError('outcome must be AuctionOutcome')
    if type(clicked) is not bool:
        raise ValueError('clicked must be a boolean')
    reservation = ledger.reservation(reservation_id)
    if reservation.advertiser_id != outcome.advertiser_id:
        raise ValueError('reservation advertiser does not match auction winner')
    if reservation.max_charge_cents != cpc_to_cents(outcome.bid_cpc):
        raise ValueError('reservation max charge does not match winner bid')
    if clicked:
        return ledger.settle(event_id=event_id, reservation_id=reservation_id,
                             charged_cents=cpc_to_cents(outcome.charged_cpc_if_clicked))
    return ledger.release(event_id=event_id, reservation_id=reservation_id)
