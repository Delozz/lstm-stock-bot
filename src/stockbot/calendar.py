"""Alpaca market calendar wrapper. All session times are America/New_York."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetCalendarRequest

from stockbot.retry import with_retries

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Session:
    date: date
    open: datetime
    close: datetime


def _to_ny(dt: datetime) -> datetime:
    # alpaca-py returns naive datetimes already expressed in exchange time.
    return dt.replace(tzinfo=NY) if dt.tzinfo is None else dt.astimezone(NY)


def get_sessions(client, start: date, end: date) -> list[Session]:
    days = with_retries(lambda: client.get_calendar(GetCalendarRequest(start=start, end=end)))
    return [Session(d.date, _to_ny(d.open), _to_ny(d.close)) for d in days]


def session_on(sessions: list[Session], day: date) -> Session | None:
    return next((s for s in sessions if s.date == day), None)


def latest_completed(sessions: list[Session], now: datetime) -> Session | None:
    done = [s for s in sessions if s.close <= now]
    return done[-1] if done else None


def previous_session(sessions: list[Session], day: date) -> Session | None:
    before = [s for s in sessions if s.date < day]
    return before[-1] if before else None
