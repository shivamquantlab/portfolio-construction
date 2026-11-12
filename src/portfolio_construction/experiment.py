"""Train/test minimum-variance demonstration on seeded synthetic returns."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .optimizer import minimum_variance_weights


def run(output_dir: Path, seed: int = 61) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    vol = np.array([0.010, 0.015, 0.022])
    corr = np.array([[1.0, 0.35, 0.15], [0.35, 1.0, 0.25], [0.15, 0.25, 1.0]])
    covariance = np.outer(vol, vol) * corr
    training = rng.multivariate_normal(np.zeros(3), covariance, size=504)
    testing = rng.multivariate_normal(np.zeros(3), covariance, size=252)
    weights = minimum_variance_weights(np.cov(training, rowvar=False))
    portfolio_returns = testing @ weights
    equal_returns = testing @ np.full(3, 1.0 / 3)
    wealth_min = np.cumprod(1.0 + portfolio_returns)
    wealth_equal = np.cumprod(1.0 + equal_returns)
    summary: dict[str, object] = {
        "seed": seed,
        "training_observations": int(training.shape[0]),
        "testing_observations": int(testing.shape[0]),
        "weights": weights.tolist(),
        "test_annualized_volatility": float(portfolio_returns.std(ddof=1) * np.sqrt(252)),
        "equal_weight_test_annualized_volatility": float(equal_returns.std(ddof=1) * np.sqrt(252)),
        "synthetic_test_terminal_wealth": float(wealth_min[-1]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(wealth_min, label="Minimum variance", color="#1d4ed8")
    ax.plot(wealth_equal, label="Equal weight", color="#64748b", alpha=0.85)
    ax.set(title="Synthetic held-out wealth paths", xlabel="Test period", ylabel="Growth of 1 unit")
    ax.legend(frameon=False)
    ax.text(0.01, 0.98, "SYNTHETIC • illustrative only • not investment performance",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_holdout.png", dpi=150)
    plt.close(fig)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()
    run(args.output_dir, seed=args.seed)
    print(f"Wrote synthetic portfolio report to {args.output_dir}")


if __name__ == "__main__":
    main()

