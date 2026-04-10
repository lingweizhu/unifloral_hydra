<h1 align="center">🌹 Unifloral: Unified Offline Reinforcement Learning</h1>

<p align="center">
    <a href= "https://arxiv.org/abs/2504.11453">
        <img src="https://img.shields.io/badge/arXiv-2504.11453-b31b1b.svg" /></a>
</p>

Unified implementations and rigorous evaluation for offline reinforcement learning - built by [Matthew Jackson](https://github.com/EmptyJackson), [Uljad Berdica](https://github.com/uljad), and [Jarek Liesen](https://github.com/keraJLi).

## 💡 Code Philosophy

- ⚛️ **Single-file**: We implement algorithms as standalone Python files.
- 🤏 **Minimal**: We only edit what is necessary between algorithms, making comparisons straightforward.
- ⚡️ **GPU-accelerated**: We use JAX and end-to-end compile all training code, enabling lightning-fast training.

Inspired by [CORL](https://github.com/tinkoff-ai/CORL) and [CleanRL](https://github.com/vwxyzjn/cleanrl) - check them out!

## 🤖 Algorithms

We provide two types of algorithm implementation:

1. **Standalone**: Each algorithm is implemented as a [single file](algorithms) with minimal dependencies, making it easy to understand and modify.
2. **Unified**: Most algorithms are available as configs for our unified implementation [`unifloral.py`](algorithms/unifloral.py).

After training, final evaluation results are saved to `.npz` files in [`final_returns/`](final_returns) for analysis using our evaluation protocol.

The repo now supports two workflows:

1. Direct single-file execution with Tyro, e.g. `python algorithms/unifloral.py --help`
2. A Hydra launcher via `python main.py`, with multirun sweeps and optional MySQL logging

### Model-free

| Algorithm | Standalone | Unified | Extras |
| --- | --- | --- | --- |
| BC | [`bc.py`](algorithms/bc.py) | [`unifloral/bc.yaml`](configs/unifloral/bc.yaml) | - |
| SAC-N | [`sac_n.py`](algorithms/sac_n.py) | [`unifloral/sac_n.yaml`](configs/unifloral/sac_n.yaml) | [[ArXiv]](https://arxiv.org/abs/2110.01548) |
| EDAC | [`edac.py`](algorithms/edac.py) | [`unifloral/edac.yaml`](configs/unifloral/edac.yaml) | [[ArXiv]](https://arxiv.org/abs/2110.01548) |
| CQL | [`cql.py`](algorithms/cql.py) | - | [[ArXiv]](https://arxiv.org/abs/2006.04779) |
| IQL | [`iql.py`](algorithms/iql.py) | [`unifloral/iql.yaml`](configs/unifloral/iql.yaml) | [[ArXiv]](https://arxiv.org/abs/2110.06169) |
| TD3-BC | [`td3_bc.py`](algorithms/td3_bc.py) | [`unifloral/td3_bc.yaml`](configs/unifloral/td3_bc.yaml) | [[ArXiv]](https://arxiv.org/abs/2106.06860) |
| ReBRAC | [`rebrac.py`](algorithms/rebrac.py) | [`unifloral/rebrac.yaml`](configs/unifloral/rebrac.yaml) | [[ArXiv]](https://arxiv.org/abs/2305.09836) |
| TD3-AWR | - | [`unifloral/td3_awr.yaml`](configs/unifloral/td3_awr.yaml) | [[ArXiv]](https://arxiv.org/abs/2504.11453) |

### Model-based

We implement a single script for dynamics model training: [`dynamics.py`](algorithms/dynamics.py), with config [`dynamics.yaml`](configs/dynamics.yaml).

| Algorithm | Standalone | Unified | Extras |
| --- | --- | --- | --- |
| MOPO | [`mopo.py`](algorithms/mopo.py) | - | [[ArXiv]](https://arxiv.org/abs/2005.13239) |
| MOReL | [`morel.py`](algorithms/morel.py) | - | [[ArXiv]](https://arxiv.org/abs/2005.05951) |
| COMBO | [`combo.py`](algorithms/combo.py) | - | [[ArXiv]](https://arxiv.org/abs/2102.08363) |
| MoBRAC | - | [`unifloral/mobrac.yaml`](configs/unifloral/mobrac.yaml) | [[ArXiv]](https://arxiv.org/abs/2504.11453) |

New ones coming soon 👀

## 📊 Evaluation

Our evaluation script ([`evaluation.py`](evaluation.py)) implements the protocol described in our paper, analysing the performance of a UCB bandit over a range of policy evaluations.

```python
from evaluation import load_results_dataframe, bootstrap_bandit_trials
import jax.numpy as jnp

# Load all results from the final_returns directory
df = load_results_dataframe("final_returns")

# Run bandit trials with bootstrapped confidence intervals
results = bootstrap_bandit_trials(
    returns_array=jnp.array(policy_returns),  # Shape: (num_policies, num_rollouts)
    num_subsample=8,     # Number of policies to subsample
    num_repeats=1000,    # Number of bandit trials
    max_pulls=200,       # Maximum pulls per trial
    ucb_alpha=2.0,       # UCB exploration coefficient
    n_bootstraps=1000,   # Bootstrap samples for confidence intervals
    confidence=0.95      # Confidence level
)

# Access results
pulls = results["pulls"]                      # Number of pulls at each step
means = results["estimated_bests_mean"]       # Mean score of estimated best policy
ci_low = results["estimated_bests_ci_low"]    # Lower confidence bound
ci_high = results["estimated_bests_ci_high"]  # Upper confidence bound
```

## ⚙️ Launcher

The root [`main.py`](main.py) is now a Hydra entrypoint around the existing single-file [`algorithms/unifloral.py`](algorithms/unifloral.py) implementation.

It keeps the fast training path in `unifloral.py`, while adding:

- Hydra multirun sweeps
- per-agent config groups under [`configs/agent/`](configs/agent)
- a simple top-level `dataset` field, so you can sweep environments directly from [`configs/config.yaml`](configs/config.yaml)
- optional MySQL logging with the same `runs` / `returns` / `summary` pattern used in `d4rl_jax`

The default experiment definition lives in [`configs/config.yaml`](configs/config.yaml). A typical setup looks like:

```yaml
defaults:
  - db: credentials
  - schema: policy-schema
  - agent: iql
  - override hydra/launcher: joblib
  - _self_

seed: 0
dataset: halfcheetah-medium-v2
log: false

hydra:
  mode: MULTIRUN
  launcher:
    n_jobs: ${oc.env:NUM_JOBS,1}
  sweeper:
    params:
      seed: range(0,5)
      dataset: halfcheetah-medium-v2,hopper-medium-v2,walker2d-medium-v2
      agent: iql,rebrac,td3_bc
```

On a single GPU machine, a good default is:

```bash
NUM_JOBS=1 python main.py
```

`log: false` disables WandB by default so local Hydra/MySQL sweeps can run without a WandB account. You can still enable it with `log=true`.

You can still override anything from the CLI when needed:

```bash
NUM_JOBS=1 python main.py hydra.sweeper.params.agent=iql,rebrac hydra.sweeper.params.dataset=halfcheetah-medium-v2,hopper-medium-v2
```

## 🗄️ MySQL

Database logging is disabled by default in [`configs/db/credentials.yaml`](configs/db/credentials.yaml).

To enable it:

```bash
cp configs/db/credentials-local.example.yaml configs/db/credentials-local.yaml
```

Then fill in your MySQL credentials and run:

```bash
NUM_JOBS=1 python main.py db=credentials-local
```

The default schema lives in [`configs/schema/policy-schema.yaml`](configs/schema/policy-schema.yaml).

## 📝 Cite us!
```bibtex
@misc{jackson2025clean,
      title={A Clean Slate for Offline Reinforcement Learning},
      author={Matthew Thomas Jackson and Uljad Berdica and Jarek Liesen and Shimon Whiteson and Jakob Nicolaus Foerster},
      year={2025},
      eprint={2504.11453},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.11453},
}
```
