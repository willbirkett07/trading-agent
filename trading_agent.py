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
        "You get a JSON snapshot: account, open positions and stats for each watchlist symbol. "
        "You have no news or fundamentals, only those numbers. Never invent facts. "
        "Doing nothing is a valid decision; only trade when you see an edge in the data.\n"
        f"A separate risk engine clips or rejects orders that break these limits: "
        f"max {cfg.max_position_pct:.0%} of equity per position, max {cfg.max_exposure_pct:.0%} invested, "
        f"max {cfg.max_orders_per_run} orders per run, stop loss required on every buy "
        f"({cfg.min_stop_pct:.0%}-{cfg.max_stop_pct:.0%}).\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{"market_view": "one or two sentences", "orders": [{"symbol": "NVDA", "side": "buy" or "sell", '
        '"notional_usd": 900, "stop_loss_pct": 8, "take_profit_pct": 20, "confidence": 0.0-1.0, '
        '"reason": "short reason"}]}\n'
        "Percentages are plain numbers (8 means 8%). For sells, stop_loss_pct and take_profit_pct are ignored."
    )


class ClaudeDecider:
    def __init__(self, cfg):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.cfg = cfg

    def decide(self, snapshot):
        msg = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=1500,
            system=build_system_prompt(self.cfg),
            messages=[{"role": "user", "content": "Current snapshot:\n" + json.dumps(snapshot)}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def parse_decision(text):
    """Any parsing problem means 'do nothing', never a guess."""
    try:
        d = json.loads(text[text.index("{"): text.rindex("}") + 1])
        orders = d.get("orders", [])
        if not isinstance(orders, list):
            raise ValueError("orders is not a list")
        return {"market_view": str(d.get("market_view", ""))[:500],
                "orders": [o for o in orders if isinstance(o, dict)]}
    except Exception:
        return {"market_view": "unparseable model output", "orders": []}


# --------------------------------------------------------------------------
# Risk engine: the part Claude cannot override
# --------------------------------------------------------------------------
def vet_orders(cfg, proposals, acct, positions, prices):
    approved, rejected = [], []
    equity, cash = acct["equity"], acct["cash"]
    held = {p["symbol"]: p["qty"] for p in positions}
    values = {p["symbol"]: p["qty"] * p["price"] for p in positions}
    exposure = sum(values.values())

    for o in proposals:
        tag = f'{o.get("side")} {o.get("symbol")}'
        if len(approved) >= cfg.max_orders_per_run:
            rejected.append(f"{tag}: order limit reached")
            continue
        try:
            sym = str(o["symbol"]).upper()
            side = str(o["side"]).lower()
            notional = float(o["notional_usd"])
        except (KeyError, ValueError, TypeError):
            rejected.append(f"{tag}: malformed order")
            continue
        price = prices.get(sym)
        if sym not in cfg.watchlist:
            rejected.append(f"{tag}: not on watchlist")
        elif side not in ("buy", "sell"):
            rejected.append(f"{tag}: side must be buy or sell")
        elif not price or price < cfg.min_price:
            rejected.append(f"{tag}: no valid price or below minimum price")
        elif notional <= 0:
            rejected.append(f"{tag}: non-positive size")
        elif side == "sell":
            have = held.get(sym, 0)
            if have < 1:
                rejected.append(f"{tag}: no position to sell (shorting disabled)")
                continue
            qty = have if notional >= have * price else max(1, math.floor(notional / price))
            qty = int(min(qty, have))
            approved.append({"symbol": sym, "side": "sell", "qty": qty, "price": price,
                             "reason": o.get("reason", "")})
            held[sym] = have - qty
            exposure -= qty * price
            cash += qty * price
        else:  # buy
            try:
                conf = float(o.get("confidence", 0))
                stop = float(o["stop_loss_pct"]) / 100
                tp_raw = o.get("take_profit_pct")
                tp = float(tp_raw) / 100 if tp_raw is not None else None
            except (KeyError, ValueError, TypeError):
                rejected.append(f"{tag}: stop loss required and must be a number")
                continue
            if conf < cfg.min_confidence:
                rejected.append(f"{tag}: confidence {conf:.2f} below {cfg.min_confidence}")
                continue
            stop = min(max(stop, cfg.min_stop_pct), cfg.max_stop_pct)
            tp = max(tp if tp is not None else 2 * stop, stop)
            budget = min(notional,
                         cfg.max_position_pct * equity - values.get(sym, 0),
                         cfg.max_exposure_pct * equity - exposure,
                         cash)
            qty = math.floor(budget / price) if budget > 0 else 0
            if qty < 1:
                rejected.append(f"{tag}: no room under position/exposure/cash limits")
                continue
            approved.append({"symbol": sym, "side": "buy", "qty": qty, "price": price,
                             "stop_price": round(price * (1 - stop), 2),
                             "take_profit": round(price * (1 + tp), 2),
                             "reason": o.get("reason", "")})
            values[sym] = values.get(sym, 0) + qty * price
            exposure += qty * price
            cash -= qty * price
    return approved, rejected


def load_state(cfg):
    p = cfg.state_dir / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(cfg, state):
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "state.json").write_text(json.dumps(state))


def check_halts(cfg, acct, state):
    """Return a reason string if trading must not happen right now."""
    if (cfg.state_dir / "KILL").exists():
        return "KILL file present"
    if (cfg.state_dir / "HALT").exists():
        return "HALT file present (drawdown limit hit earlier); delete it to resume"
    eq = acct["equity"]
    peak = max(state.get("peak", eq), eq)
    state["peak"] = peak
    if acct["last_equity"] and eq / acct["last_equity"] - 1 <= -cfg.max_daily_loss_pct:
        return "daily loss limit hit"
    if eq / peak - 1 <= -cfg.max_drawdown_pct:
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        (cfg.state_dir / "HALT").write_text(f"drawdown limit hit at {now_iso()}")
        return "drawdown limit hit; trading halted until you delete the HALT file"
    return None


# --------------------------------------------------------------------------
# One decision cycle
# --------------------------------------------------------------------------
def run_once(cfg, broker, decider, approve=None):
    state = load_state(cfg)
    acct = broker.account()
    if hasattr(broker, "fills"):
        try:
            new_fills = broker.fills(state.get("last_fill_ts"))
            if new_fills:
                state["last_fill_ts"] = new_fills[-1]["ts"]
                log_event(cfg, "fills", {"fills": new_fills})
        except Exception as e:  # analytics must never stop trading
            log_event(cfg, "error", {"error": f"fills: {e!r}"})
    halt = check_halts(cfg, acct, state)
    save_state(cfg, state)
    if halt:
        log_event(cfg, "halt", {"reason": halt})
        print(f"HALTED: {halt}")
        return []

    snapshot, prices = build_snapshot(cfg, broker, acct)
    raw = decider.decide(snapshot)
    decision = parse_decision(raw)
    approved, rejected = vet_orders(cfg, decision["orders"], acct, snapshot["positions"], prices)

    results = []
    for o in approved:
        if approve and not approve(o):
            results.append({**o, "status": "declined by user"})
            continue
        try:
            if o["side"] == "sell":
                broker.cancel_open_orders(o["symbol"])
            res = broker.submit(o["symbol"], o["side"], o["qty"],
                                o.get("stop_price"), o.get("take_profit"))
            results.append({**o, "status": "submitted", "broker_id": res.get("id")})
        except Exception as e:  # keep going; one failed order must not stop the loop
            results.append({**o, "status": f"error: {e}"})

    log_event(cfg, "cycle", {"account": snapshot["account"], "positions": snapshot["positions"],
                             "benchmark": prices.get("SPY"), "market_view": decision["market_view"],
                             "raw_model_output": raw, "approved": approved,
                             "rejected": rejected, "results": results})
    print(f"[{now_iso()}] view: {decision['market_view']}")
    for r in results:
        print(f"  {r['side'].upper()} {r['qty']} {r['symbol']} @~{r['price']:.2f} -> {r['status']} | {r['reason']}")
    for r in rejected:
        print(f"  REJECTED {r}")
    if not results and not rejected:
        print("  no trades")
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Claude-driven trading agent (safe by default)")
    ap.add_argument("--broker", choices=["sim", "alpaca"], default="sim")
    ap.add_argument("--live", action="store_true",
                    help="REAL MONEY via Alpaca live account; also needs env I_ACCEPT_RISK=yes")
    ap.add_argument("--auto", action="store_true",
                    help="live mode: skip the per-trade y/N confirmation (fully autonomous)")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    ap.add_argument("--interval", type=int, default=60, help="minutes between cycles")
    ap.add_argument("--watchlist", help="comma-separated tickers, e.g. SPY,NVDA,MU")
    args = ap.parse_args()

    cfg = Config()
    if args.watchlist:
        cfg.watchlist = [s.strip().upper() for s in args.watchlist.split(",") if s.strip()]

    if args.live:
        if args.broker != "alpaca":
            sys.exit("--live requires --broker alpaca")
        if os.environ.get("I_ACCEPT_RISK") != "yes":
            sys.exit("Refusing live trading. Set I_ACCEPT_RISK=yes to confirm you accept the risk of losing money.")

    if args.broker == "alpaca":
        key, secret = os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")
        if not key or not secret:
            sys.exit("Set ALPACA_KEY and ALPACA_SECRET")
        broker = AlpacaBroker(key, secret, paper=not args.live)
    else:
        broker = SimBroker(cfg.watchlist, seed=int(time.time()))

    def ask_user(o):
        ans = input(f"Execute {o['side'].upper()} {o['qty']} {o['symbol']} (~${o['qty'] * o['price']:.0f})? "
                    f"Reason: {o['reason']} [y/N] ")
        return ans.strip().lower() == "y"

    approve = ask_user if (args.live and not args.auto) else None
    decider = ClaudeDecider(cfg)
    mode = "LIVE" if args.live else ("paper" if args.broker == "alpaca" else "simulated")
    print(f"Starting agent in {mode} mode. Create agent_state/KILL to stop it at any time.")

    errors = 0
    while True:
        try:
            if args.broker == "sim" or broker.market_open():
                run_once(cfg, broker, decider, approve)
            else:
                print(f"[{now_iso()}] market closed")
            errors = 0
        except Exception as e:
            errors += 1
            log_event(cfg, "error", {"error": repr(e)})
            print(f"error: {e!r}")
            if errors >= 5:
                sys.exit("Too many consecutive errors; stopping.")
        if args.once:
            break
        if args.broker == "sim":
            broker.step()
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()

