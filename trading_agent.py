#!/usr/bin/env python3
"""
Claude trading agent.

Claude decides what to trade. Hard-coded risk rules decide what is *allowed*.
A broker adapter executes. Claude cannot change the rules.

Safe by default: simulated broker, long-only stocks/ETFs, no margin, no options.
Not financial advice. Automated trading with real money can lose everything you deploy.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


# --------------------------------------------------------------------------
# Configuration (edit these, or override the watchlist with --watchlist)
# --------------------------------------------------------------------------
@dataclass
class Config:
    watchlist: list = field(default_factory=lambda: [
        "SPY", "QQQ", "SOXX", "NVDA", "AVGO", "TSM", "MU", "WDC", "LLY", "XOM",
    ])
    goal: str = "High risk, high return growth. Holding period of days to weeks. Long only."
    max_position_pct: float = 0.10     # max 10% of equity in any one stock
    max_exposure_pct: float = 0.60     # max 60% of equity invested; the rest stays cash
    max_orders_per_run: int = 3        # limits churn
    max_daily_loss_pct: float = 0.03   # down 3% on the day -> stop trading for the day
    max_drawdown_pct: float = 0.15     # down 15% from peak -> halt until you delete HALT
    min_stop_pct: float = 0.03         # every buy needs a stop loss between 3% and 25%
    max_stop_pct: float = 0.25
    min_price: float = 2.0             # no penny stocks
    min_confidence: float = 0.6        # buys below this confidence are rejected
    state_dir: Path = Path("agent_state")
    model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(cfg, kind, payload):
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.state_dir / "decisions.jsonl", "a") as f:
        f.write(json.dumps({"ts": now_iso(), "kind": kind, **payload}, default=str) + "\n")


# --------------------------------------------------------------------------
# Brokers
# --------------------------------------------------------------------------
class SimBroker:
    """Offline random-walk market for testing the loop. Prices are NOT real; stops are ignored."""

    def __init__(self, symbols, cash=10_000.0, seed=None):
        self.rng = random.Random(seed)
        self.cash = cash
        self.last_equity = cash
        self.pos = {}  # symbol -> [qty, avg_entry]
        self.hist = {}
        for s in symbols:
            series = [self.rng.uniform(20, 400)]
            for _ in range(29):
                series.append(series[-1] * (1 + self.rng.gauss(0.0005, 0.02)))
            self.hist[s] = series

    def price(self, s):
        return self.hist[s][-1]

    def step(self):
        for h in self.hist.values():
            h.append(h[-1] * (1 + self.rng.gauss(0.0005, 0.02)))

    def market_open(self):
        return True

    def account(self):
        eq = self.cash + sum(q * self.price(s) for s, (q, _) in self.pos.items())
        return {"equity": eq, "cash": self.cash, "last_equity": self.last_equity}

    def positions(self):
        return [
            {"symbol": s, "qty": q, "avg_entry": round(a, 2), "price": round(self.price(s), 2),
             "unrealized_pl": round((self.price(s) - a) * q, 2)}
            for s, (q, a) in self.pos.items() if q > 0
        ]

    def closes(self, symbols, days=25):
        return {s: self.hist[s][-days:] for s in symbols}

    def cancel_open_orders(self, symbol):
        pass

    def submit(self, symbol, side, qty, stop_price=None, take_profit=None):
        p = self.price(symbol)
        q, a = self.pos.get(symbol, [0, 0.0])
        if side == "buy":
            cost = qty * p
            if cost > self.cash:
                raise RuntimeError("insufficient cash")
            self.pos[symbol] = [q + qty, (q * a + cost) / (q + qty)]
            self.cash -= cost
        else:
            if qty > q:
                raise RuntimeError("cannot sell more than held")
            self.pos[symbol] = [q - qty, a]
            self.cash += qty * p
        return {"id": f"sim-{int(time.time() * 1000)}", "status": "filled", "price": p}


class AlpacaBroker:
    """Alpaca REST adapter (paper by default). Written from Alpaca's public API docs;
    test it on a paper account before trusting it."""

    def __init__(self, key, secret, paper=True):
        self.base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.data = "https://data.alpaca.markets"
        self.h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _get(self, url, **params):
        r = requests.get(url, headers=self.h, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def market_open(self):
        return self._get(f"{self.base}/v2/clock")["is_open"]

    def account(self):
        a = self._get(f"{self.base}/v2/account")
        return {"equity": float(a["equity"]), "cash": float(a["cash"]),
                "last_equity": float(a["last_equity"])}

    def positions(self):
        return [
            {"symbol": p["symbol"], "qty": float(p["qty"]),
             "avg_entry": float(p["avg_entry_price"]), "price": float(p["current_price"]),
             "unrealized_pl": float(p["unrealized_pl"])}
            for p in self._get(f"{self.base}/v2/positions")
        ]

    def fills(self, after=None):
        """New trade fills (including stop-loss / take-profit exits) since `after`."""
        params = {"direction": "asc", "page_size": 100}
        if after:
            params["after"] = after
        acts = self._get(f"{self.base}/v2/account/activities/FILL", **params)
        return [{"id": a["id"], "ts": a["transaction_time"], "symbol": a["symbol"],
                 "side": a["side"], "qty": float(a["qty"]), "price": float(a["price"])}
                for a in acts]

    def closes(self, symbols, days=25):
        start = (datetime.now(timezone.utc) - timedelta(days=days * 2)).strftime("%Y-%m-%d")
        out, token = {s: [] for s in symbols}, None
        while True:
            params = dict(symbols=",".join(symbols), timeframe="1Day", start=start,
                          limit=10000, feed="iex", adjustment="split")
            if token:
                params["page_token"] = token
            d = self._get(f"{self.data}/v2/stocks/bars", **params)
            for s, bars in (d.get("bars") or {}).items():
                out[s].extend(b["c"] for b in bars)
            token = d.get("next_page_token")
            if not token:
                break
        return {s: v[-days:] for s, v in out.items()}

    def cancel_open_orders(self, symbol):
        """Open stop/take-profit legs lock the shares, so cancel them before selling."""
        for o in self._get(f"{self.base}/v2/orders", status="open", symbols=symbol):
            requests.delete(f"{self.base}/v2/orders/{o['id']}", headers=self.h, timeout=20)

    def submit(self, symbol, side, qty, stop_price=None, take_profit=None):
        body = {"symbol": symbol, "qty": str(int(qty)), "side": side,
                "type": "market", "time_in_force": "day"}
        if side == "buy" and stop_price and take_profit:
            body.update(order_class="bracket",
                        take_profit={"limit_price": str(take_profit)},
                        stop_loss={"stop_price": str(stop_price)})
        r = requests.post(f"{self.base}/v2/orders", headers=self.h, json=body, timeout=20)
        r.raise_for_status()
        return r.json()


# --------------------------------------------------------------------------
# Market snapshot given to Claude
# --------------------------------------------------------------------------
def build_snapshot(cfg, broker, acct):
    positions = broker.positions()
    closes = broker.closes(cfg.watchlist, 25)
    watch, prices = {}, {}
    for s, c in closes.items():
        if len(c) < 6:
            continue
        rets = [c[i] / c[i - 1] - 1 for i in range(1, len(c))]
        watch[s] = {
            "last": round(c[-1], 2),
            "ret_5d_pct": round((c[-1] / c[-6] - 1) * 100, 1),
            "ret_20d_pct": round((c[-1] / c[max(0, len(c) - 21)] - 1) * 100, 1),
            "daily_vol_pct": round(statistics.pstdev(rets) * 100, 2),
        }
        prices[s] = c[-1]
    snapshot = {
        "time_utc": now_iso(),
        "account": {k: round(v, 2) for k, v in acct.items()},
        "positions": positions,
        "watchlist": watch,
    }
    return snapshot, prices


# --------------------------------------------------------------------------
# Claude as decision-maker
# --------------------------------------------------------------------------
def build_system_prompt(cfg):
    return (
        "You are the decision engine of an automated trading agent.\n"
        f"Goal: {cfg.goal}\n"
        "You may trade long-only US stocks and ETFs from the watchlist. No shorting, margin or options.\n"
        "You get a JSON snapshot: account, open positions and stats for each
