import json
import tempfile
import unittest
from pathlib import Path

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest import mock

from trading_agent import (Config, ConsensusDecider, Finnhub, SimBroker, check_halts,
                           closed_trades, notify, open_theses, parse_decision, pattern_stats,
                           recent_decisions, rsi, run_once, track_record, trade_restrictions,
                           vet_orders, AlpacaBroker, live_snapshot, main, write_live)


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


def series(values):
    return [(f"2026-01-{i + 1:02d}" if i < 31 else f"2026-02-{i - 30:02d}", v) for i, v in enumerate(values)]


class Patterns(unittest.TestCase):
    def test_uptrend_stats(self):
        s = [(f"d{i:04d}", 100 + i) for i in range(300)]
        st = pattern_stats(s)
        self.assertTrue(st["above_200d_avg"])
        self.assertGreater(st["ret_12m_pct"], 0)
        self.assertEqual(st["rsi14"], 100.0)
        self.assertEqual(st["max_drawdown_1y_pct"], 0.0)

    def test_short_history_leaves_out_long_stats(self):
        st = pattern_stats([(f"d{i:03d}", 100 + i) for i in range(30)])
        self.assertNotIn("above_200d_avg", st)
        self.assertNotIn("ret_12m_pct", st)
        self.assertIn("ret_1m_pct", st)

    def test_too_little_data_returns_nothing(self):
        self.assertEqual(pattern_stats([("d1", 1.0), ("d2", 2.0)]), {})

    def test_behaviour_on_market_down_days(self):
        dates = [f"d{i:03d}" for i in range(60)]
        spy_vals, my_vals = [100.0], [100.0]
        for i in range(1, 60):
            move = -0.02 if i % 10 == 0 else 0.002
            spy_vals.append(spy_vals[-1] * (1 + move))
            my_vals.append(my_vals[-1] * (1 + 2 * move))   # falls twice as hard
        st = pattern_stats(list(zip(dates, my_vals)), list(zip(dates, spy_vals)))
        down = st["on_days_spy_fell_1pct"]
        self.assertEqual(down["days"], 5)
        self.assertLess(down["avg_this_pct"], down["avg_spy_pct"])
        self.assertGreater(st["correlation_with_spy"], 0.9)

    def test_rsi_all_down_is_zero(self):
        self.assertEqual(rsi([float(x) for x in range(30, 0, -1)]), 0.0)


class Memory(unittest.TestCase):
    EVENTS = [
        {"kind": "cycle", "ts": "2026-11-09T15:00:00+00:00", "account": {"equity": 10000.0}, "benchmark": 600.0,
         "market_view": "calm", "rejected": [],
         "results": [{"side": "buy", "qty": 10, "symbol": "AAA", "price": 100.0, "status": "submitted",
                      "reason": "momentum", "stop_price": 92.0, "take_profit": 120.0}]},
        {"kind": "fills", "ts": "x", "fills": [
            {"id": "1", "ts": "2026-11-09T15:01:00Z", "symbol": "AAA", "side": "buy", "qty": 10, "price": 100.0},
            {"id": "2", "ts": "2026-11-10T15:01:00Z", "symbol": "AAA", "side": "sell", "qty": 10, "price": 110.0}]},
        {"kind": "cycle", "ts": "2026-11-10T16:00:00+00:00", "account": {"equity": 10100.0}, "benchmark": 612.0,
         "market_view": "up", "rejected": ["x"], "results": []},
    ]

    def test_track_record_pairs_buys_and_sells(self):
        rec = track_record(self.EVENTS)
        self.assertEqual(rec["closed_trades"], 1)
        self.assertEqual(rec["win_rate_pct"], 100)
        self.assertEqual(rec["pnl_by_symbol_usd"], {"AAA": 100.0})
        self.assertEqual(rec["agent_return_pct"], 1.0)
        self.assertEqual(rec["spy_return_pct_same_period"], 2.0)

    def test_track_record_empty_log(self):
        self.assertEqual(track_record([]), {"closed_trades": 0})

    def test_open_position_thesis(self):
        th = open_theses(self.EVENTS, [{"symbol": "AAA"}, {"symbol": "BBB"}])
        self.assertEqual(th["AAA"]["why"], "momentum")
        self.assertEqual(th["AAA"]["stop_price"], 92.0)
        self.assertNotIn("BBB", th)

    def test_recent_decisions_summary(self):
        recent = recent_decisions(self.EVENTS)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[1]["blocked"], 1)
        self.assertIn("momentum", recent[0]["orders"][0])


class CapturingDecider:
    def __init__(self, payload=None):
        self.snapshot, self.payload = None, payload or {"orders": []}

    def decide(self, snapshot):
        self.snapshot = snapshot
        return json.dumps(self.payload)


class FakeFundamentals:
    def get(self, symbols):
        return {"AAA": {"pe_ttm": 25.0, "next_earnings": "2026-11-20"}}


class ExtraContext(unittest.TestCase):
    def test_snapshot_contains_patterns_fundamentals_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            dec = CapturingDecider()
            run_once(cfg, SimBroker(cfg.watchlist, seed=3), dec, fundamentals=FakeFundamentals())
            self.assertIn("AAA", dec.snapshot["patterns_1y"])
            self.assertEqual(dec.snapshot["fundamentals"]["AAA"]["pe_ttm"], 25.0)
            self.assertEqual(set(dec.snapshot["memory"]),
                             {"track_record", "open_position_theses", "recent_decisions", "your_notes_to_self",
                              "weekly_lessons"})

    def test_notes_are_saved_capped_and_fed_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            broker = SimBroker(cfg.watchlist, seed=3)
            for i in range(17):
                run_once(cfg, broker, CapturingDecider({"orders": [], "notes_to_future_self": f"note {i}"}))
            dec = CapturingDecider()
            run_once(cfg, broker, dec)
            notes = dec.snapshot["memory"]["your_notes_to_self"]
            self.assertEqual(len(notes), 15)
            self.assertEqual(notes[-1], "note 16")
            self.assertEqual(len(dec.snapshot["memory"]["recent_decisions"]), 5)

    def test_broken_extras_never_stop_trading(self):
        class Broken(SimBroker):
            def history(self, symbols, days=400):
                raise RuntimeError("history down")

        class BrokenFundamentals:
            def get(self, symbols):
                raise RuntimeError("fundamentals down")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            broker = Broken(cfg.watchlist, seed=3)
            dec = FakeDecider({"market_view": "ok", "orders": [buy(notional=500)]})
            results = run_once(cfg, broker, dec, fundamentals=BrokenFundamentals())
            self.assertEqual(results[0]["status"], "submitted")
            log = (Path(tmp) / "decisions.jsonl").read_text()
            self.assertIn("patterns_1y", log)
            self.assertIn("fundamentals", log)


class FinnhubFake(Finnhub):
    calls = 0

    def _get(self, path, **params):
        FinnhubFake.calls += 1
        if path == "/stock/metric":
            return {"metric": {"peTTM": 30.126, "revenueGrowthTTMYoy": 55.5, "junk": "x"}}
        return {"earningsCalendar": [{"date": "2026-11-20"}]}


class FundamentalsProvider(unittest.TestCase):
    def test_parses_and_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            FinnhubFake.calls = 0
            fh = FinnhubFake("token", cfg)
            first = fh.get(["NVDA"])
            self.assertEqual(first["NVDA"]["pe_ttm"], 30.13)
            self.assertEqual(first["NVDA"]["revenue_growth_yoy_pct"], 55.5)
            self.assertEqual(first["NVDA"]["next_earnings"], "2026-11-20")
            self.assertEqual(FinnhubFake.calls, 2)
            fh.get(["NVDA"])
            self.assertEqual(FinnhubFake.calls, 2)   # second call served from the cache

    def test_failure_skips_symbol(self):
        class Down(Finnhub):
            def _get(self, path, **params):
                raise RuntimeError("down")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(Down("t", make_cfg(tmp)).get(["NVDA"]), {})


NOW = datetime(2026, 11, 10, 15, 0, tzinfo=timezone.utc)


def iso(days_ago):
    return (NOW - timedelta(days=days_ago)).isoformat()


class RiskPerTrade(unittest.TestCase):
    def test_wide_stop_shrinks_the_position(self):
        cfg = make_cfg(tempfile.gettempdir())
        big_equity = {"equity": 100_000.0, "cash": 100_000.0, "last_equity": 100_000.0}
        tight, _ = vet_orders(cfg, [buy(notional=50_000, stop=5)], big_equity, [], PRICES)
        wide, _ = vet_orders(cfg, [buy(notional=50_000, stop=25)], big_equity, [], PRICES)
        # 1% of 100k = 1,000 at risk. 10% position cap is 10,000, which a 5% stop (risk 500) fits inside.
        self.assertEqual(tight[0]["qty"], 100)
        # a 25% stop would risk 2,500 on a 10,000 position, so it is cut to 4,000 notional
        self.assertEqual(wide[0]["qty"], 40)


class Restrictions(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg(tempfile.gettempdir())

    def test_earnings_blackout(self):
        snap = {"fundamentals": {"AAA": {"next_earnings": "2026-11-12"}, "BBB": {"next_earnings": "2026-12-20"}}}
        no_buy, _ = trade_restrictions(self.cfg, snap, [], now=NOW)
        self.assertIn("AAA", no_buy)
        self.assertNotIn("BBB", no_buy)

    def test_minimum_hold_blocks_selling(self):
        events = [{"kind": "cycle", "ts": iso(0.5), "account": {"equity": 1}, "results": [
            {"side": "buy", "symbol": "AAA", "status": "submitted"}]}]
        snap = {"positions": [{"symbol": "AAA", "price": 100.0, "avg_entry": 100.0}]}
        _, no_sell = trade_restrictions(self.cfg, snap, events, now=NOW)
        self.assertIn("AAA", no_sell)

    def test_minimum_hold_allows_selling_a_big_loser(self):
        events = [{"kind": "cycle", "ts": iso(0.5), "account": {"equity": 1}, "results": [
            {"side": "buy", "symbol": "AAA", "status": "submitted"}]}]
        snap = {"positions": [{"symbol": "AAA", "price": 90.0, "avg_entry": 100.0}]}   # down 10%
        _, no_sell = trade_restrictions(self.cfg, snap, events, now=NOW)
        self.assertEqual(no_sell, {})

    def test_old_buy_can_be_sold(self):
        events = [{"kind": "cycle", "ts": iso(5), "account": {"equity": 1}, "results": [
            {"side": "buy", "symbol": "AAA", "status": "submitted"}]}]
        snap = {"positions": [{"symbol": "AAA", "price": 100.0, "avg_entry": 100.0}]}
        _, no_sell = trade_restrictions(self.cfg, snap, events, now=NOW)
        self.assertEqual(no_sell, {})

    def test_rebuy_cooldown_after_a_loss_only(self):
        events = [{"kind": "fills", "fills": [
            {"ts": iso(6), "symbol": "AAA", "side": "buy", "qty": 1, "price": 100.0},
            {"ts": iso(1), "symbol": "AAA", "side": "sell", "qty": 1, "price": 90.0},
            {"ts": iso(6), "symbol": "BBB", "side": "buy", "qty": 1, "price": 100.0},
            {"ts": iso(1), "symbol": "BBB", "side": "sell", "qty": 1, "price": 110.0}]}]
        no_buy, _ = trade_restrictions(self.cfg, {}, events, now=NOW)
        self.assertIn("AAA", no_buy)       # lost money recently
        self.assertNotIn("BBB", no_buy)    # a win does not trigger the cooldown
        self.assertEqual(len(closed_trades(events)), 2)

    def test_vet_orders_applies_the_blocks(self):
        ok, bad = vet_orders(make_cfg(tempfile.gettempdir()), [buy()], ACCT, [], PRICES,
                             no_buy={"AAA": "earnings in 1 day(s)"})
        self.assertEqual(ok, [])
        self.assertIn("blocked", bad[0])
        pos = [{"symbol": "AAA", "qty": 3, "price": 100.0}]
        sell = {"symbol": "AAA", "side": "sell", "notional_usd": 300}
        ok, bad = vet_orders(make_cfg(tempfile.gettempdir()), [sell], ACCT, pos, PRICES,
                             no_sell={"AAA": "minimum hold"})
        self.assertEqual(ok, [])
        self.assertIn("blocked", bad[0])


class Consensus(unittest.TestCase):
    def run_consensus(self, trader, reviewer):
        out = ConsensusDecider(FakeDecider(trader), FakeDecider(reviewer)).decide({})
        return json.loads(out)

    def test_agreed_buy_uses_the_more_cautious_numbers(self):
        a = {"market_view": "x", "orders": [buy(notional=900, stop=8, conf=0.9)]}
        b = {"orders": [buy(notional=500, stop=5, conf=0.7)]}
        res = self.run_consensus(a, b)
        order = res["orders"][0]
        self.assertEqual(order["notional_usd"], 500.0)
        self.assertEqual(order["confidence"], 0.7)
        self.assertEqual(order["stop_loss_pct"], 5.0)
        self.assertIn("reviewer agreed", order["reason"])

    def test_buy_without_reviewer_agreement_is_dropped(self):
        res = self.run_consensus({"orders": [buy()]}, {"orders": []})
        self.assertEqual(res["orders"], [])
        self.assertIn("reviewer did not agree", res["reviewer_dropped"][0])

    def test_exit_needs_only_one_vote(self):
        sell = {"symbol": "AAA", "side": "sell", "notional_usd": 100, "reason": "risk"}
        self.assertEqual(len(self.run_consensus({"orders": [sell]}, {"orders": []})["orders"]), 1)
        self.assertEqual(len(self.run_consensus({"orders": []}, {"orders": [sell]})["orders"]), 1)

    def test_dropped_buys_reach_the_decision(self):
        res = self.run_consensus({"orders": [buy()]}, {"orders": []})
        self.assertEqual(len(parse_decision(json.dumps(res))["dropped"]), 1)


class ReflectingDecider(CapturingDecider):
    def __init__(self):
        super().__init__()
        self.reflections = 0

    def reflect(self, data):
        self.reflections += 1
        return "Lesson: keep positions small."


class Reflection(unittest.TestCase):
    def test_first_run_starts_the_clock_without_calling_claude(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            dec = ReflectingDecider()
            run_once(cfg, SimBroker(cfg.watchlist, seed=3), dec)
            self.assertEqual(dec.reflections, 0)
            self.assertIn("last_reflection", json.loads((Path(tmp) / "state.json").read_text()))

    def test_weekly_reflection_is_saved_and_fed_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            (Path(tmp) / "state.json").write_text(json.dumps(
                {"last_reflection": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()}))
            dec = ReflectingDecider()
            run_once(cfg, SimBroker(cfg.watchlist, seed=3), dec)
            self.assertEqual(dec.reflections, 1)
            self.assertEqual(dec.snapshot["memory"]["weekly_lessons"], "Lesson: keep positions small.")
            run_once(cfg, SimBroker(cfg.watchlist, seed=3), dec)   # too soon for another one
            self.assertEqual(dec.reflections, 1)

    def test_failed_reflection_does_not_stop_trading(self):
        class Failing(ReflectingDecider):
            def reflect(self, data):
                raise RuntimeError("api down")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            (Path(tmp) / "state.json").write_text(json.dumps(
                {"last_reflection": (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()}))
            dec = Failing()
            run_once(cfg, SimBroker(cfg.watchlist, seed=3), dec)
            self.assertIsNotNone(dec.snapshot)   # it still made its decision


class Alerts(unittest.TestCase):
    def test_no_topic_means_no_network_call(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch("trading_agent.requests.post") as post:
            notify("t", "m")
            post.assert_not_called()

    def test_topic_sends_a_message(self):
        with mock.patch.dict("os.environ", {"NTFY_TOPIC": "abc"}), mock.patch("trading_agent.requests.post") as post:
            notify("Title", "Body")
            self.assertEqual(post.call_args[0][0], "https://ntfy.sh/abc")
            self.assertEqual(post.call_args[1]["headers"]["Title"], "Title")

    def test_a_failing_alert_never_raises(self):
        with mock.patch.dict("os.environ", {"NTFY_TOPIC": "abc"}), \
                mock.patch("trading_agent.requests.post", side_effect=RuntimeError("down")):
            notify("t", "m")

    def test_trade_and_halt_alerts_fire_once(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"NTFY_TOPIC": "abc"}), \
                mock.patch("trading_agent.requests.post") as post:
            cfg = make_cfg(tmp)
            run_once(cfg, SimBroker(cfg.watchlist, seed=3), FakeDecider({"orders": [buy(notional=500)]}))
            self.assertEqual(post.call_count, 1)                      # one alert for the trade
            (Path(tmp) / "KILL").write_text("x")
            for _ in range(3):
                run_once(cfg, SimBroker(cfg.watchlist, seed=3), FakeDecider({"orders": []}))
            self.assertEqual(post.call_count, 2)                      # halt alert sent once, not every run


class ConfigDefaults(unittest.TestCase):
    def test_empty_model_variable_falls_back(self):
        import importlib
        import trading_agent
        with mock.patch.dict("os.environ", {"ANTHROPIC_MODEL": ""}):
            importlib.reload(trading_agent)
            self.assertEqual(trading_agent.Config().model, "claude-sonnet-5-5")
        importlib.reload(trading_agent)

    def test_defensive_etfs_are_on_the_watchlist(self):
        self.assertTrue({"XLV", "XLE", "TLT", "GLD"} <= set(Config().watchlist))


class ClosedSim(SimBroker):
    def market_open(self):
        return False


class LiveSnapshot(unittest.TestCase):
    def test_snapshot_contents(self):
        b = SimBroker(["AAA"], cash=10_000.0, seed=1)
        b.submit("AAA", "buy", 5)
        live = live_snapshot(b)
        self.assertTrue(live["market_open"])
        self.assertEqual(live["positions"][0]["symbol"], "AAA")
        self.assertAlmostEqual(live["account"]["equity"], 10_000.0, delta=0.05)
        self.assertEqual(set(live), {"updated_at", "market_open", "account", "day_pnl", "day_pnl_pct",
                                     "positions", "spy"})

    def test_day_change_uses_previous_close(self):
        b = SimBroker(["AAA"], cash=10_000.0, seed=1)
        b.last_equity = 9_000.0
        live = live_snapshot(b)
        self.assertEqual(live["day_pnl"], round(live["account"]["equity"] - 9_000.0, 2))
        self.assertEqual(live["day_pnl_pct"], round(live["day_pnl"] / 9_000.0 * 100, 2))

    def test_files_written_and_history_only_grows_on_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = SimBroker(["AAA"], cash=10_000.0, seed=1)
            write_live(b, tmp)
            write_live(b, tmp)                       # nothing changed: no new history point
            lines = (Path(tmp) / "equity.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads((Path(tmp) / "live.json").read_text())["account"]["equity"], 10_000.0)
            b.step()
            b.submit("AAA", "buy", 5)
            b.step()                                  # prices moved: equity changes
            write_live(b, tmp)
            self.assertEqual(len((Path(tmp) / "equity.jsonl").read_text().splitlines()), 2)

    def test_closed_market_and_no_change_means_no_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("trading_agent.now_iso", side_effect=["t1", "t1", "t2", "t2"]):
            b = ClosedSim(["AAA"], seed=1)
            write_live(b, tmp)
            first = (Path(tmp) / "live.json").read_text()
            write_live(b, tmp)
            self.assertEqual((Path(tmp) / "live.json").read_text(), first)   # still stamped t1

    def test_open_market_always_refreshes_the_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("trading_agent.now_iso", side_effect=["t1", "t2"]):
            b = SimBroker(["AAA"], seed=1)
            write_live(b, tmp)
            write_live(b, tmp)
            self.assertEqual(json.loads((Path(tmp) / "live.json").read_text())["updated_at"], "t2")

    def test_market_just_closed_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_live(SimBroker(["AAA"], seed=1), tmp)                      # open
            write_live(ClosedSim(["AAA"], seed=1), tmp)                      # now closed, same numbers
            self.assertFalse(json.loads((Path(tmp) / "live.json").read_text())["market_open"])

    def test_damaged_history_file_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "equity.jsonl").write_text("not json\n")
            write_live(SimBroker(["AAA"], seed=1), tmp)
            self.assertEqual(len((Path(tmp) / "equity.jsonl").read_text().splitlines()), 2)

    def test_history_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = SimBroker(["AAA"], seed=1)
            for _ in range(8):
                b.step()
                write_live(b, tmp, max_points=5)
            self.assertLessEqual(len((Path(tmp) / "equity.jsonl").read_text().splitlines()), 5)

    def test_latest_prices_parsing(self):
        class Fake(AlpacaBroker):
            def _get(self, url, **params):
                return {"SPY": {"latestTrade": {"p": 612.34}}, "ZZZ": {}}
        self.assertEqual(Fake("k", "s").latest_prices(["SPY", "ZZZ"]), {"SPY": 612.34})

    def test_benchmark_failure_does_not_break_the_snapshot(self):
        class NoPrices(SimBroker):
            def latest_prices(self, symbols):
                raise RuntimeError("data down")
        self.assertIsNone(live_snapshot(NoPrices(["AAA"], seed=1))["spy"])

    def test_cli_snapshot_needs_no_claude_and_makes_no_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with mock.patch.object(sys, "argv", ["trading_agent.py", "--snapshot", "--out-dir", "out"]), \
                        mock.patch("trading_agent.ClaudeDecider", side_effect=AssertionError("must not be created")):
                    main()
                self.assertTrue((Path(tmp) / "out" / "live.json").exists())
                self.assertFalse((Path(tmp) / "agent_state").exists())
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
