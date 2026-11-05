# Claude trading agent (cloud version)

Runs once an hour while the US market is open, on GitHub's servers, with **paper (fake) money only**. A dashboard site shows what it did. Do the setup on a computer; it's a one-time job.

**Not financial advice. Claude only sees prices, so there's no evidence it will make money. Treat it as an experiment.**

## Setup (about 15 minutes)

1. **Create the repo.** On GitHub, click New repository. Name it `trading-agent`, set it to **Public** (free dashboards need this; your secret keys stay hidden), and create it.
2. **Upload the files.** Unzip this package. In the repo click Add file, then Upload files, and drag in `trading_agent.py`, `test_agent.py`, `README.md`, and the `docs` and `agent_state` folders. Commit.
3. **Add the schedule file.** Click Add file, then Create new file. In the name box type exactly `.github/workflows/agent.yml` (the slashes make the folders). Paste in the contents of `agent.yml` from this package and commit.
4. **Add three secrets.** Settings, then Secrets and variables, then Actions, then New repository secret. Add:
   - `ANTHROPIC_API_KEY`: your key from the Anthropic console
   - `ALPACA_KEY` and `ALPACA_SECRET`: the keys from your Alpaca **paper** account (check the dashboard says "Paper" before generating them)
5. **Turn on the dashboard.** Settings, then Pages. Under Source pick "Deploy from a branch", branch `main`, folder `/docs`, then Save. Your site will be at `https://YOUR-USERNAME.github.io/trading-agent/`.
6. **Test it.** Click the Actions tab, pick "Trading agent", then Run workflow. A green tick means it worked. If the US market is closed it just says so and does nothing; the first real cycle happens on the next trading day (weekdays, roughly 2:30pm to 9pm UK time).

## Stop it

- **Pause:** Actions tab, "Trading agent", the three dots, Disable workflow.
- **Emergency brake:** add an empty file called `agent_state/KILL` in the repo (Add file, Create new file). The agent refuses to trade while it exists.

## Safety rules (Claude cannot override these)

10% max per stock, 60% max invested, 3 orders per run, a stop loss on every buy, trading stops for the day at -3%, and halts completely at -15% from peak (delete `agent_state/HALT` to resume). Stocks and ETFs on the watchlist only; no shorting, margin or options. Edit the watchlist and limits at the top of `trading_agent.py`.

## Good to know

- The workflow never passes `--live`, so it cannot touch real money. Going live is a separate, deliberate change; don't do it without weeks of paper results you trust.
- The repo is public, so anyone can see the paper account's value and trades. Your keys are not exposed.
- Each run makes one Claude call. Roughly a dollar a week or less in API cost (approximate).
- The Alpaca connection is untested against their live servers (written from their docs), so watch the first few runs in the Actions tab.
- If runs stop after a long quiet spell, GitHub may have disabled the schedule; re-enable it in the Actions tab.
