"""
Deep Hedging Under Transaction Costs: A Comparison with Black-Scholes Delta Hedging
------------------------------------------------------------------------------------
Simple, function-based Python implementation (no heavy OOP, no deep-learning
framework required - just numpy, scipy, matplotlib).

What this script does
1. Simulates stock price paths with Geometric Brownian Motion (GBM).
2. Prices a European call and computes the closed-form Black-Scholes delta.
3. Runs a Black-Scholes delta-hedging strategy with proportional transaction costs.
4. Trains a small neural network (one hidden layer, hand-rolled forward pass)
   as a "deep hedging" policy. It is trained with scipy.optimize (numerical
   gradient) to minimize hedged P&L variance net of transaction costs -
   no autograd / PyTorch / TensorFlow needed.
5. Compares both strategies: total transaction costs, P&L distribution, and
   P&L volatility, and reproduces the three figures from the paper.

Run:
    pip install numpy scipy matplotlib --break-system-packages
    python deep_hedging_vs_bs.py
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Market simulation (Geometric Brownian Motion)
# ---------------------------------------------------------------------------

def simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths):
    """Simulate GBM stock price paths. Returns array of shape (n_paths, n_steps+1)."""
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths


# ---------------------------------------------------------------------------
# 2. Black-Scholes pricing and delta (closed form)
# ---------------------------------------------------------------------------

def bs_call_price(S, K, r, sigma, tau):
    """European call price. tau = time remaining to maturity (can be 0)."""
    tau = np.maximum(tau, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_call_delta(S, K, r, sigma, tau):
    """European call delta. tau = time remaining to maturity (can be 0)."""
    tau = np.maximum(tau, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


# ---------------------------------------------------------------------------
# 3. Black-Scholes delta hedging strategy (with proportional transaction costs)
# ---------------------------------------------------------------------------

def run_bs_delta_hedge(paths, K, r, sigma, T, cost_rate):
    """
    Hedge a short call position by holding delta shares of stock, rebalancing
    at every time step, paying proportional transaction costs on trades.

    Returns:
        final_pnl      : array (n_paths,) final hedged P&L per path
        cum_costs      : array (n_paths,) cumulative transaction costs per path
    """
    n_paths, n_pts = paths.shape
    n_steps = n_pts - 1
    dt = T / n_steps
    times = np.linspace(0, T, n_pts)

    option_premium = bs_call_price(paths[:, 0], K, r, sigma, T)  # received at t=0
    cash = option_premium.copy()          # cash account (seller receives premium)
    shares_held = np.zeros(n_paths)
    cum_costs = np.zeros(n_paths)

    for t in range(n_steps):
        tau = T - times[t]
        S_t = paths[:, t]
        target_delta = bs_call_delta(S_t, K, r, sigma, tau)

        trade = target_delta - shares_held
        costs = cost_rate * np.abs(trade) * S_t
        cash -= trade * S_t + costs           # buy/sell stock + pay costs
        cash *= np.exp(r * dt)                # cash accrues risk-free interest
        shares_held = target_delta
        cum_costs += costs

    # unwind at maturity: sell remaining shares, pay the option payoff
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)
    final_cash = cash + shares_held * S_T
    final_pnl = final_cash - payoff
    return final_pnl, cum_costs


# ---------------------------------------------------------------------------
# 4. Deep hedging: small hand-rolled neural network policy
# ---------------------------------------------------------------------------

N_HIDDEN = 4          # hidden units - kept small so training is fast
N_INPUTS = 2           # inputs: normalized price, normalized time-to-maturity


def unpack_weights(theta):
    """theta -> (W1, b1, W2, b2) for a 1-hidden-layer MLP."""
    i = 0
    W1 = theta[i:i + N_INPUTS * N_HIDDEN].reshape(N_INPUTS, N_HIDDEN); i += N_INPUTS * N_HIDDEN
    b1 = theta[i:i + N_HIDDEN]; i += N_HIDDEN
    W2 = theta[i:i + N_HIDDEN].reshape(N_HIDDEN, 1); i += N_HIDDEN
    b2 = theta[i:i + 1]
    return W1, b1, W2, b2


N_PARAMS = N_INPUTS * N_HIDDEN + N_HIDDEN + N_HIDDEN + 1


def policy_forward(theta, S_norm, tau_norm):
    """Network position (in [0,1], like a call delta) given state features."""
    W1, b1, W2, b2 = unpack_weights(theta)
    tau_norm = np.full_like(S_norm, tau_norm)
    X = np.stack([S_norm, tau_norm], axis=1)          # (n_paths, 2)
    h = np.tanh(X @ W1 + b1)                            # (n_paths, N_HIDDEN)
    out = h @ W2 + b2                                    # (n_paths, 1)
    position = 1.0 / (1.0 + np.exp(-out[:, 0]))          # sigmoid -> [0,1]
    return position


def run_deep_hedge(theta, paths, K, r, sigma, T, cost_rate):
    """Same bookkeeping as run_bs_delta_hedge, but positions come from the NN."""
    n_paths, n_pts = paths.shape
    n_steps = n_pts - 1
    dt = T / n_steps
    times = np.linspace(0, T, n_pts)

    option_premium = bs_call_price(paths[:, 0], K, r, sigma, T)
    cash = option_premium.copy()
    shares_held = np.zeros(n_paths)
    cum_costs = np.zeros(n_paths)

    for t in range(n_steps):
        tau = T - times[t]
        S_t = paths[:, t]
        S_norm = S_t / K - 1.0            # moneyness-like feature
        tau_norm = tau / T

        target_pos = policy_forward(theta, S_norm, tau_norm)
        trade = target_pos - shares_held
        costs = cost_rate * np.abs(trade) * S_t
        cash -= trade * S_t + costs
        cash *= np.exp(r * dt)
        shares_held = target_pos
        cum_costs += costs

    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)
    final_cash = cash + shares_held * S_T
    final_pnl = final_cash - payoff
    return final_pnl, cum_costs


def deep_hedge_loss(theta, paths, K, r, sigma, T, cost_rate, risk_aversion=1.0):
    """
    Training objective: keep hedged P&L close to zero (variance) while the
    transaction costs the network chooses to pay are already baked into the
    P&L via run_deep_hedge, so minimizing this naturally trades off risk vs cost.
    """
    pnl, _ = run_deep_hedge(theta, paths, K, r, sigma, T, cost_rate)
    return np.mean(pnl ** 2) * risk_aversion


def train_deep_hedge(paths_train, K, r, sigma, T, cost_rate, n_epochs=40):
    """
    Optimize the network weights with scipy's L-BFGS-B (numerical gradient).
    No autograd / backprop-through-time needed - deliberately simple.
    """
    theta0 = rng.normal(scale=0.3, size=N_PARAMS)
    loss_history = []

    def objective(theta):
        loss = deep_hedge_loss(theta, paths_train, K, r, sigma, T, cost_rate)
        loss_history.append(loss)
        return loss

    result = minimize(
        objective, theta0, method="L-BFGS-B",
        options={"maxiter": n_epochs, "eps": 1e-3}
    )
    return result.x, loss_history


# ---------------------------------------------------------------------------
# 5. Run the full comparison
# ---------------------------------------------------------------------------

def main():
    # ---- market / option parameters ----
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 0.25   # 3-month ATM call
    n_steps = 20
    cost_rate = 0.01     # 1% proportional transaction cost, matches paper's stress case

    n_paths_train = 300     # paths used to train the deep hedging network
    n_paths_test = 500       # paths used for the head-to-head comparison

    print("Simulating training paths and training the deep hedging network...")
    paths_train = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths_train)
    theta, loss_history = train_deep_hedge(paths_train, K, r, sigma, T, cost_rate)

    print("Simulating test paths and running both strategies...")
    paths_test = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths_test)

    bs_pnl, bs_costs = run_bs_delta_hedge(paths_test, K, r, sigma, T, cost_rate)
    dh_pnl, dh_costs = run_deep_hedge(theta, paths_test, K, r, sigma, T, cost_rate)

    # ---- summary table ----
    print("\n===================  RESULTS  ===================")
    print(f"{'Metric':32s}{'Black-Scholes':>16s}{'Deep Hedge':>16s}")
    print(f"{'Mean cumulative cost':32s}{bs_costs.mean():16.4f}{dh_costs.mean():16.4f}")
    print(f"{'Mean final P&L':32s}{bs_pnl.mean():16.4f}{dh_pnl.mean():16.4f}")
    print(f"{'P&L std dev (volatility)':32s}{bs_pnl.std():16.4f}{dh_pnl.std():16.4f}")
    pct_cost_change = 100 * (dh_costs.mean() - bs_costs.mean()) / bs_costs.mean()
    print(f"\nDeep hedge transaction costs vs Black-Scholes: {pct_cost_change:+.1f}%")

    # ---- Figure 1: transaction cost comparison ----
    plt.figure(figsize=(6, 4))
    plt.hist(bs_costs, bins=20, alpha=0.6, label="Black-Scholes")
    plt.hist(dh_costs, bins=20, alpha=0.6, label="Deep Hedge")
    plt.title("Cumulative Transaction Costs")
    plt.xlabel("Total cost ($)"); plt.ylabel("Number of paths"); plt.legend()
    plt.tight_layout(); plt.savefig("fig1_transaction_costs.png", dpi=150)

    # ---- Figure 2: final P&L distribution ----
    plt.figure(figsize=(6, 4))
    plt.hist(bs_pnl, bins=25, alpha=0.6, label="Black-Scholes")
    plt.hist(dh_pnl, bins=25, alpha=0.6, label="Deep Hedge")
    plt.title("Hedging P&L Distribution")
    plt.xlabel("Final P&L ($)"); plt.ylabel("Number of paths"); plt.legend()
    plt.tight_layout(); plt.savefig("fig2_pnl_distribution.png", dpi=150)

    # ---- Figure 3: training loss ----
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.title("Deep Hedging Training Loss")
    plt.xlabel("Optimizer evaluation"); plt.ylabel("Loss (mean squared P&L)")
    plt.tight_layout(); plt.savefig("fig3_training_loss.png", dpi=150)

    print("\nSaved fig1_transaction_costs.png, fig2_pnl_distribution.png, "
          "fig3_training_loss.png")


def run_cost_sensitivity_sweep(S0, K, r, sigma, T, n_steps, cost_rates,
                                n_paths_train=150, n_paths_test=300, n_epochs=25):
    """
    Extension beyond the original paper: retrain the deep hedge for several
    transaction-cost levels and compare the cost/volatility tradeoff at each.
    Returns a list of dicts, one per cost rate.
    """
    rows = []
    for cr in cost_rates:
        paths_train = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths_train)
        theta, _ = train_deep_hedge(paths_train, K, r, sigma, T, cr, n_epochs=n_epochs)

        paths_test = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths_test)
        bs_pnl, bs_costs = run_bs_delta_hedge(paths_test, K, r, sigma, T, cr)
        dh_pnl, dh_costs = run_deep_hedge(theta, paths_test, K, r, sigma, T, cr)

        rows.append({
            "cost_rate": cr,
            "bs_cost_mean": bs_costs.mean(), "dh_cost_mean": dh_costs.mean(),
            "bs_pnl_std": bs_pnl.std(), "dh_pnl_std": dh_pnl.std(),
        })
        print(f"cost_rate={cr:.3%}  BS cost={bs_costs.mean():.3f}  "
              f"DH cost={dh_costs.mean():.3f}  BS std={bs_pnl.std():.3f}  "
              f"DH std={dh_pnl.std():.3f}")
    return rows


def plot_cost_sensitivity(rows):
    cost_rates = [row["cost_rate"] for row in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(cost_rates, [row["bs_cost_mean"] for row in rows], "o-", label="Black-Scholes cost")
    plt.plot(cost_rates, [row["dh_cost_mean"] for row in rows], "o-", label="Deep Hedge cost")
    plt.xlabel("Transaction cost rate"); plt.ylabel("Mean cumulative cost ($)")
    plt.title("Transaction Cost Sensitivity"); plt.legend()
    plt.tight_layout(); plt.savefig("fig4_cost_sensitivity.png", dpi=150)

    plt.figure(figsize=(6, 4))
    plt.plot(cost_rates, [row["bs_pnl_std"] for row in rows], "o-", label="Black-Scholes P&L std")
    plt.plot(cost_rates, [row["dh_pnl_std"] for row in rows], "o-", label="Deep Hedge P&L std")
    plt.xlabel("Transaction cost rate"); plt.ylabel("P&L volatility ($)")
    plt.title("Risk Sensitivity to Transaction Costs"); plt.legend()
    plt.tight_layout(); plt.savefig("fig5_risk_sensitivity.png", dpi=150)


if __name__ == "__main__":
    main()

    print("\nRunning transaction-cost sensitivity sweep (extension)...")
    sweep_rows = run_cost_sensitivity_sweep(
        S0=100.0, K=100.0, r=0.02, sigma=0.20, T=0.25, n_steps=20,
        cost_rates=[0.001, 0.005, 0.01, 0.02],
    )
    plot_cost_sensitivity(sweep_rows)
    print("Saved fig4_cost_sensitivity.png, fig5_risk_sensitivity.png")
