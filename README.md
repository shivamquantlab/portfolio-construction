# Portfolio Construction

## Research question
How do long-only minimum-variance weights behave when the covariance estimate is learned on a training sample and evaluated on separate synthetic observations?

## Methodology and formulas
The optimizer minimizes w'Σw subject to w_i >= 0 and sum(w_i)=1 using projected gradient descent on the probability simplex. The experiment estimates Σ from 504 synthetic training returns and holds the resulting weights fixed for 252 independent synthetic test returns.

## Assumptions and limitations
- The three assets are synthetic multivariate-normal returns with a fixed covariance; seed 61 controls both samples.
- This is a covariance-only optimizer, not a return forecast. It excludes expected-return uncertainty, weight limits beyond long-only/full-investment, rebalancing, taxes, and cash.
- No trading costs are included; estimated weights can be unstable under covariance error.
- The held-out chart is a simulation diagnostic, not real-world investment performance. No real securities or market history are used.

## Reproducibility
With Python 3.11+, run python -m pip install -e ".[test]", then python -m pytest and portfolio-demo --seed 61. The command regenerates reports/synthetic_holdout.png and reports/summary.json. Training and test samples are generated independently from a fixed synthetic covariance.

