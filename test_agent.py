import json
import tempfile
import unittest
from pathlib import Path

from trading_agent import (Config, SimBroker, check_halts, parse_decision,
                           run_once, vet_orders)


def make_cfg(tmp):
    return Config(watchlist=["AAA", "BBB"], state_dir=Path(tmp))


def buy(sym="AAA", notional=5000, stop=8, conf=0.8, **kw):
    return {"symbol": sym, "side": "buy", "notional_usd": notional,
            "stop_loss_pct": stop, "take_profit_pct": 20, "confidence": conf,
            "reason": "test", **kw}


ACCT = {"equity": 10_000.0, "cash": 10_000.0, "last_equity": 10_000.0}
PRICES = {"AAA": 100.0, "BBB": 50.0}


class RiskEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = make_cfg(self.tmp.name)

    def test_position_cap(self):
        ok, _ = vet_orders(self.cfg, [buy(notional=5000)], ACCT, [], PRICES)
        self.assertEqual(ok[0]["qty"], 10)  # 10% of 10k = 1000 -> 10 shares at 100

    def test_unknown_symbol_rejected(self):
        ok, bad = vet_orders(self.cfg, [buy(sym="ZZZ")], ACCT, [], {**PRICES, "ZZZ": 10})
        self.assertEqual(ok, [])
        self.assertIn("watchlist", bad[0])

    def test_no_shorting(self):
        sell = {"symbol": "AAA", "side": "sell", "notional_usd": 500}
        ok, bad = vet_orders(self.cfg, [sell], ACCT, [], PRICES)
        self.assertEqual(ok, [])
        self.assertIn("shorting", bad[0])

    def test_sell_capped_at_holdings(self):
        pos = [{"symbol": "AAA", "qty": 3, "price": 100.0}]
        sell = {"symbol": "AAA", "side": "sell", "notional_usd": 99999}
        ok, _ = vet_orders(self.cfg, [sell], ACCT, pos, PRICES)
        self.assertEqual(ok[0]["qty"], 3)

    def test_stop_required(self):
        o = buy()
        del o["stop_loss_pct"]
        ok, bad = vet_orders(self.cfg, [o], ACCT, [], PRICES)
        self.assertEqual(ok, [])
        self.assertIn("stop loss", bad[0])

    def test_stop_clamped(self):
        tight, _ = vet_orders(self.cfg, [buy(stop=1)], ACCT, [], PRICES)
        wide, _ = vet_orders(self.cfg, [buy(stop=90)], ACCT, [], PRICES)
        self.assertEqual(tight[0]["stop_price"], 97.0)   # clamped to 3%
        self.assertEqual(wide[0]["stop_price"], 75.0)    # clamped to 25%

    def test_low_confidence_rejected(self):
        ok, bad = vet_orders(self.cfg, [buy(conf=0.3)], ACCT, [], PRICES)
        self.assertEqual(ok, [])
        self.assertIn("confidence", bad[0])

    def test_exposure_cap(self):
        pos = [{"symbol": "BBB", "qty": 116, "price": 50.0}]  # 5,800 invested, 4,200 room? cap is 6,000
        ok, _ = vet_orders(self.cfg, [buy(notional=5000)], ACCT, pos, PRICES)
        self.assertEqual(ok[0]["qty"], 2)  # only 200 of exposure room left

    def test_order_count_cap(self):
        many = [buy(sym="AAA", notional=100), buy(sym="BBB", notional=100)] * 3
        ok, bad = vet_orders(self.cfg, many, ACCT, [], PRICES)
        self.assertLessEqual(len(ok), self.cfg.max_orders_per_run)
        self.assertTrue(any("limit" in b for b in bad))

    def test_cash_limit(self):
        acct = {"equity": 10_000.0, "cash": 150.0, "last_equity": 10_000.0}
        ok, _ = vet_orders(self.cfg, [buy(notional=5000)], acct, [], PRICES)
        self.assertEqual(ok[0]["qty"], 1)


class Halts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = make_cfg(self.tmp.name)

    def test_daily_loss_halt(self):
        acct = {"equity": 9_600.0, "cash": 9_600.0, "last_equity": 10_000.0}
        self.assertIn("daily loss", check_halts(self.cfg, acct, {}))

    def test_drawdown_writes_halt_file(self):
        acct = {"equity": 8_000.0, "cash": 8_000.0, "last_equity": 8_000.0}
        msg = check_halts(self.cfg, acct, {"peak": 10_000.0})
        self.assertIn("drawdown", msg)
        self.assertTrue((self.cfg.state_dir / "HALT").exists())
        self.assertIn("HALT", check_halts(self.cfg, ACCT, {}))

    def test_kill_switch(self):
        (self.cfg.state_dir / "KILL").write_text("x")
        self.assertIn("KILL", check_halts(self.cfg, ACCT, {}))

    def test_normal_day_no_halt(self):
        self.assertIsNone(check_halts(self.cfg, ACCT, {}))


class Parsing(unittest.TestCase):
    def test_garbage_means_no_orders(self):
        self.assertEqual(parse_decision("I think you should buy everything!")["orders"], [])

    def test_json_with_chatter(self):
        text = 'Sure! {"market_view": "calm", "orders": [{"symbol": "AAA"}]} Hope that helps.'
        self.assertEqual(len(parse_decision(text)["orders"]), 1)

    def test_bad_orders_field(self):
        self.assertEqual(parse_decision('{"orders": "buy all"}')["orders"], [])


class FakeDecider:
    def __init__(self, payload):
        self.payload = payload

    def decide(self, snapshot):
        return json.dumps(self.payload)


class EndToEnd(unittest.TestCase):
    def test_oversized_order_is_clipped_and_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            broker = SimBroker(cfg.watchlist, cash=10_000.0, seed=1)
            decider = FakeDecider({"market_view": "test", "orders": [buy(notional=100_000)]})
            results = run_once(cfg, broker, decider)
            self.assertEqual(results[0]["status"], "submitted")
            invested = 10_000.0 - broker.cash
            self.assertLessEqual(invested, 0.10 * 10_000.0 + 1e-6)
            log = (Path(tmp) / "decisions.jsonl").read_text()
            self.assertIn("cycle", log)

    def test_halted_account_does_not_trade(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            (Path(tmp) / "KILL").write_text("x")
            broker = SimBroker(cfg.watchlist, seed=1)
            results = run_once(cfg, broker, FakeDecider({"orders": [buy()]}))
            self.assertEqual(results, [])
            self.assertEqual(broker.cash, 10_000.0)

    def test_declined_by_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            broker = SimBroker(cfg.watchlist, seed=1)
            results = run_once(cfg, broker, FakeDecider({"orders": [buy()]}), approve=lambda o: False)
            self.assertEqual(results[0]["status"], "declined by user")
            self.assertEqual(broker.cash, 10_000.0)


class FillsBroker(SimBroker):
    def fills(self, after=None):
        return [{"id": "1", "ts": "2026-11-06T15:00:00Z", "symbol": "AAA",
                 "side": "buy", "qty": 2.0, "price": 100.0}]


class FillsLogging(unittest.TestCase):
    def test_fills_are_logged_and_remembered(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            run_once(cfg, FillsBroker(cfg.watchlist, seed=1), FakeDecider({"orders": []}))
            kinds = [json.loads(l)["kind"] for l in (Path(tmp) / "decisions.jsonl").read_text().splitlines()]
            self.assertIn("fills", kinds)
            self.assertEqual(json.loads((Path(tmp) / "state.json").read_text())["last_fill_ts"],
                             "2026-11-06T15:00:00Z")


if __name__ == "__main__":
    unittest.main()
