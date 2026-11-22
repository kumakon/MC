import math
import os
import time
from statistics import NormalDist

import numpy as np

try:
    import cupy as cp

    xp = cp
    on_gpu = True
except Exception:
    cp = None
    xp = np
    on_gpu = False

TRADING_DAYS = 252
ALPHA = 0.05  # уровень 95%


def to_cpu(x):
    return cp.asnumpy(x) if on_gpu else np.asarray(x)


def estimate_params(prices: np.ndarray):
    """Оценка годовых mu, sigma и корреляций по ценам (дни x активы)."""
    log_ret = np.diff(np.log(prices), axis=0)
    sigma_d = log_ret.std(axis=0, ddof=1)
    sigma = sigma_d * math.sqrt(TRADING_DAYS)
    # для GBM: E[log-доходность] = mu - sigma^2 / 2
    mu = log_ret.mean(axis=0) * TRADING_DAYS + 0.5 * sigma**2
    corr = np.corrcoef(log_ret, rowvar=False)
    return mu, sigma, corr


def analytic_var_cvar(initial_value, mu_p, sigma_p, horizon, alpha=ALPHA):
    """VaR и CVaR для логнормального V_T = V0 * exp(m + s * Z)."""
    m = (mu_p - 0.5 * sigma_p**2) * horizon
    s = sigma_p * math.sqrt(horizon)
    z = NormalDist().inv_cdf(alpha)
    q_value = initial_value * math.exp(m + s * z)  # alpha-квантиль стоимости
    var = initial_value - q_value
    tail_mean_value = (
        initial_value * math.exp(m + 0.5 * s**2) * NormalDist().cdf(z - s) / alpha
    )
    cvar = initial_value - tail_mean_value
    return var, cvar


def main(
    n_paths: int = 10_000_000,
    initial_value: float = 100_000.0,
    horizon_years: float = 1.0,
    seed: int = 42,
):
    # параметры портфеля (иллюстративные значения)
    weights = np.array([0.30, 0.25, 0.25, 0.20])
    mus = np.array([0.08, 0.05, 0.12, 0.03])
    vols = np.array([0.20, 0.15, 0.25, 0.10])
    corr = np.array(
        [
            [1.00, 0.40, 0.30, 0.10],
            [0.40, 1.00, 0.35, 0.15],
            [0.30, 0.35, 1.00, 0.05],
            [0.10, 0.15, 0.05, 1.00],
        ]
    )

    prices_csv = os.getenv("PRICES_CSV")
    if prices_csv:
        prices = np.genfromtxt(prices_csv, delimiter=",", skip_header=1)
        mus, vols, corr = estimate_params(prices)
        weights = np.full(len(mus), 1.0 / len(mus))  # равные доли, если не заданы

    # проверки входных данных
    if not math.isclose(weights.sum(), 1.0, abs_tol=1e-9):
        raise ValueError("Веса должны давать в сумме 1")
    if not np.allclose(corr, corr.T):
        raise ValueError("Корреляционная матрица должна быть симметричной")
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        raise ValueError("Корреляционная матрица не положительно определена")

    n_assets = len(weights)

    # аналитическое приближение: портфель как один GBM
    Sigma = np.diag(vols) @ corr @ np.diag(vols)
    mu_p = float(weights @ mus)
    sigma_p = math.sqrt(float(weights @ Sigma @ weights))
    var_an, cvar_an = analytic_var_cvar(initial_value, mu_p, sigma_p, horizon_years)

    # симуляция
    t0 = time.time()
    rng = xp.random.default_rng(seed)  # единый API для NumPy и CuPy
    Z = rng.standard_normal((n_paths, n_assets), dtype=xp.float32)
    Zc = Z @ xp.asarray(L.T, dtype=xp.float32)  # коррелированные нормали

    drift = xp.asarray((mus - 0.5 * vols**2) * horizon_years, dtype=xp.float32)
    diff = xp.asarray(vols * math.sqrt(horizon_years), dtype=xp.float32)
    alloc = xp.asarray(weights * initial_value, dtype=xp.float32)

    terminal = (alloc * xp.exp(drift + diff * Zc)).sum(axis=1)  # buy-and-hold

    if on_gpu:
        cp.cuda.Stream.null.synchronize()
    sim_time = time.time() - t0

    # метрики риска
    losses = initial_value - terminal
    var95 = float(to_cpu(xp.percentile(losses, 95)))
    cvar95 = float(to_cpu(losses[losses >= var95].mean()))
    mean_val = float(to_cpu(terminal.mean()))
    std_val = float(to_cpu(terminal.std()))
    p_loss = float(to_cpu((terminal < initial_value).mean()))
    p10, p50, p90 = (float(to_cpu(xp.percentile(terminal, q))) for q in (10, 50, 90))

    device = "GPU (CuPy/CUDA)" if on_gpu else "CPU (NumPy)"
    print("\n=== Monte Carlo Portfolio Risk (GBM по активам, buy-and-hold) ===")
    print(f"Device:              {device}")
    print(f"Scenarios (n_paths): {n_paths:,}")
    print(f"Horizon (years):     {horizon_years}")
    print(f"Initial Value:       ${initial_value:,.2f}")
    print(f"mu_p (annual):       {mu_p:.4f}")
    print(f"sigma_p (annual):    {sigma_p:.4f}")
    print("\n--- Results ---")
    print(f"Mean terminal:       ${mean_val:,.2f}")
    print(f"Std terminal:        ${std_val:,.2f}")
    print(f"10th pct:            ${p10:,.2f}")
    print(f"Median:              ${p50:,.2f}")
    print(f"90th pct:            ${p90:,.2f}")
    print(f"VaR(95%):            ${var95:,.2f}   (positive = loss)")
    print(f"CVaR(95%):           ${cvar95:,.2f}  (avg loss in worst 5%)")
    print(f"P(Loss):             {p_loss * 100:.2f}%")
    print("\n--- Check vs analytic (portfolio as single lognormal GBM) ---")
    print(f"Analytic VaR(95%):   ${var_an:,.2f}   (MC diff: {var95 - var_an:+,.2f})")
    print(f"Analytic CVaR(95%):  ${cvar_an:,.2f}  (MC diff: {cvar95 - cvar_an:+,.2f})")
    print(f"\nElapsed:             {sim_time:.2f} sec\n")


if __name__ == "__main__":
    main(
        n_paths=int(os.getenv("N_PATHS", "10000000")),
        horizon_years=float(os.getenv("HORIZON_YEARS", "1.0")),
        seed=int(os.getenv("SEED", "42")),
    )