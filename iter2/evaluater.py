import yfinance as yf
import pandas as pd
import numpy as np
import random
import importlib.util
import re
import copy
import os

from backtesting import Backtest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# =========================================================
# DATA
# =========================================================
all_data: pd.DataFrame = yf.download("SPY", period="30y", progress=False)
all_data.columns = [column[0] for column in all_data.columns]
nrows = len(all_data)
train_data = all_data[:nrows - nrows // 8]
test_data = all_data[nrows - nrows // 8:nrows]


# =========================================================
# WINDOW
# =========================================================
def random_window(rng: pd.DatetimeIndex, min_width, max_width):
    n = len(rng)
    start_idx = np.random.randint(0, n - min_width + 1)
    width = np.random.randint(min_width, np.maximum((n - start_idx), max_width) + 1)
    return rng[start_idx:start_idx + width]

def sequential_window(rng: pd.DatetimeIndex, idx, denom):
    n = len(rng)
    width = n // denom
    start_idx = width * idx
    return rng[start_idx:start_idx + width]


# =========================================================
# SCORE
# =========================================================
def safe(x):
    return 0.0 if (np.isnan(x) or np.isinf(x)) else x


def score(stats):
    # sharpe = safe(stats["Sharpe Ratio"])
    # ret = safe(stats["Return (Ann.) [%]"])
    # drawdown = abs(safe(stats["Max. Drawdown [%]"]))
    # trades = stats["# Trades"]
    # winrate = safe(stats["Win Rate [%]"])
    # pf = safe(stats["Profit Factor"])

    # if trades < 10 or drawdown > 80:
    #     return -1e9

    # return (
    #     2.0 * sharpe
    #     + 0.05 * ret
    #     - 0.03 * drawdown
    #     + 0.5 * np.log1p(pf)
    #     + 0.01 * winrate
    # )
    return stats["Return (Ann.) [%]"]


# =========================================================
# SAFE LOADER
# =========================================================
def load_strategy_checked(path, class_name="Model"):
    with open(path, "r") as f:
        source = f.read()

    code = compile(source, path, "exec")

    spec = importlib.util.spec_from_file_location("strategy_module", path)
    module = importlib.util.module_from_spec(spec)

    exec(code, module.__dict__)

    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError("Model class not found")

    return cls


# =========================================================
# COPY + PARAM INJECTION
# seed.py -> generated.py
# =========================================================
def generate_from_seed(seed_path, out_path, params):
    with open(seed_path) as f:
        text = f.read()

    def repl(match):
        key = match.group(1)
        return f"{key}{params[key.strip().split('=')[0].strip()]}"

    pattern = r"(\w+\s*=\s*)@\w+"
    text = re.sub(pattern, repl, text)

    with open(out_path, "w") as f:
        f.write(text)


# =========================================================
# EVAL
# =========================================================
def evaluate(model_path, n_runs=10, cash=10000, is_validate=False):
    Strategy = load_strategy_checked(model_path, "Model")

    scores, incomes, holds = [], [], []

    for i in range(n_runs):
        if(is_validate):
            window = sequential_window(test_data.index, i, n_runs)
            df = test_data.loc[window]
        else:
            window = sequential_window(train_data.index, i, n_runs)
            df = train_data.loc[window]

        bt = Backtest(df, Strategy, cash=cash, commission=0.002)
        stats = bt.run()

        # scores.append(score(stats))
        incomes.append(stats["Equity Final [$]"])

        hold = (df["Close"].iloc[-1] / df["Close"].iloc[0]) * cash
        holds.append(hold)
        
        scores.append(stats["Equity Final [$]"] - hold)

    return np.mean(scores), np.mean(incomes), np.mean(holds)

def clamp_params(v):
    v = np.array(v)

    v[0] = np.clip(v[0], 5, 50)    # n1
    v[1] = np.clip(v[1], 10, 100)  # n2
    v[2] = np.clip(v[2], 50, 300)  # trend_n

    if v[0] >= v[1]:
        v[0] = v[1] - 1

    return v


def params_to_vec(p):
    return np.array([p["n1"], p["n2"], p["trend_n"]], dtype=float)


def vec_to_params(v):
    v = clamp_params(v)
    return {
        "n1": int(v[0]),
        "n2": int(v[1]),
        "trend_n": int(v[2]),
    }

import numpy as np
import copy
def evaluate_stable(path):
    s, _, _ = evaluate(path, 10)
    return s

def optimize(seed_path, gen_path, n_iter=50, pop_size=30, elite_k=5):

    bounds = {
        "n1": (5, 500),
        "n2": (10, 1000),
        "trend_n": (10, 3000),
    }

    keys = list(bounds.keys())

    def sample_random():
        return {
            "n1": np.random.randint(*bounds["n1"]),
            "n2": np.random.randint(*bounds["n2"]),
            "trend_n": np.random.randint(*bounds["trend_n"]),
        }

    def clip(p):
        return {
            k: int(np.clip(p[k], *bounds[k]))
            for k in keys
        }

    def mutate(p, sigma):
        out = {}
        for k in keys:
            out[k] = p[k] + int(np.random.normal(0, sigma[k]))
        return clip(out)

    # ----------------------------
    # INIT POPULATION (pure random)
    # ----------------------------
    population = []
    for _ in range(pop_size):
        p = sample_random()
        generate_from_seed(seed_path, gen_path, p)
        score = evaluate_stable(gen_path)
        population.append((score, p))

    population.sort(reverse=True, key=lambda x: x[0])

    sigma = {"n1": 5, "n2": 10, "trend_n": 30}

    best_score, best_params = population[0]

    # ----------------------------
    # EVOLUTION LOOP
    # ----------------------------
    for i in range(n_iter):

        elites = population[:elite_k]

        new_pop = []

        # exploit elites
        for score, p in elites:
            for _ in range(pop_size // elite_k):
                cand = mutate(p, sigma)

                if cand["n1"] >= cand["n2"]:
                    continue

                generate_from_seed(seed_path, gen_path, cand)
                s = evaluate_stable(gen_path)
                new_pop.append((s, cand))

        # explore fresh random points (important!)
        for _ in range(pop_size // 2):
            cand = sample_random()
            generate_from_seed(seed_path, gen_path, cand)
            s = evaluate_stable(gen_path)
            new_pop.append((s, cand))

        new_pop.sort(reverse=True, key=lambda x: x[0])
        population = new_pop[:pop_size]

        if population[0][0] > best_score:
            best_score, best_params = population[0]
            # tighten search slightly around winners
            sigma = {k: max(1, int(v * 0.9)) for k, v in sigma.items()}
        else:
            # increase exploration if stuck
            sigma = {k: min(bounds[k][1]-bounds[k][0], int(v * 1.1 + 1)) for k, v in sigma.items()}

        print(f"iter {i}: best={best_score:.4f}, params={best_params}, sigma={sigma}")

    generate_from_seed(seed_path, gen_path, best_params)
    final_train_score, _, _ = evaluate(gen_path, is_validate=False)
    final_score, _, _ = evaluate(gen_path, is_validate=True)

    print("\nFINAL:", best_params)
    print(f"Train: {final_train_score}")
    print(f"Validate: {final_score}")

    return best_params

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    best = optimize(
        seed_path="iter2/out/seed.py",
        gen_path="iter2/out/generated.py",
        n_iter=100
    )

    print("DONE:", best)