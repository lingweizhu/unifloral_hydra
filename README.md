# unifloral_hydra

`unifloral_hydra` is a Hydra-based experiment wrapper around the original single-file `unifloral` implementation.

The goal of this fork is simple:

- keep the fast JAX training path in [`algorithms/unifloral.py`](/Users/lingweizhu/Desktop/workspace/unifloral/algorithms/unifloral.py)
- add config-driven sweeps over agents, datasets, and seeds
- store results to MySQL in the same spirit as `d4rl_jax`
- keep the repo simple instead of introducing a large framework rewrite

## What Changed

Compared with the upstream `unifloral` repo, this fork adds:

- a Hydra launcher in [`main.py`](/Users/lingweizhu/Desktop/workspace/unifloral/main.py)
- Hydra agent config groups in [`configs/agent/`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/agent)
- top-level sweep configuration in [`configs/config.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/config.yaml)
- optional MySQL logging in [`experiment.py`](/Users/lingweizhu/Desktop/workspace/unifloral/experiment.py)
- database schema config in [`configs/schema/policy-schema.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/schema/policy-schema.yaml)
- database credentials config in [`configs/db/`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/db)

The training logic itself is still executed by [`algorithms/unifloral.py`](/Users/lingweizhu/Desktop/workspace/unifloral/algorithms/unifloral.py). The main change there is that the script now exposes `run(args)` so the Hydra launcher can call it and log results afterward.

## Repo Layout

- [`main.py`](/Users/lingweizhu/Desktop/workspace/unifloral/main.py): Hydra entrypoint
- [`algorithms/unifloral.py`](/Users/lingweizhu/Desktop/workspace/unifloral/algorithms/unifloral.py): original unified training implementation
- [`experiment.py`](/Users/lingweizhu/Desktop/workspace/unifloral/experiment.py): MySQL logging helpers
- [`configs/config.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/config.yaml): default run and sweep definition
- [`configs/agent/`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/agent): per-agent defaults

Available Hydra agent configs:

- `bc`
- `bpr`
- `edac`
- `iql`
- `mobrac`
- `rebrac`
- `sac_n`
- `td3_awr`
- `td3_bc`

## Installation

Use the `unifloral` conda environment:

```bash
conda activate unifloral
pip install -r requirements.txt
```

This repo still depends on D4RL and `mujoco-py`, so full environment runtime is easiest on Linux. On macOS Apple Silicon, import and config validation work, but MuJoCo-backed D4RL environments are still hard to build reliably.

## Running

The default launcher is Hydra:

```bash
python main.py
```

By default, Hydra reads [`configs/config.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/config.yaml), composes the selected `agent` config, and executes the sweep declared in `hydra.sweeper.params` sequentially. This repo intentionally uses Hydra's default launcher instead of the joblib launcher, because joblib can try to pickle D4RL/Gym/JAX objects and fail with errors such as `cannot pickle 'mmap.mmap' object`.

If you want a single run instead of a sweep:

```bash
python main.py hydra.mode=RUN agent=iql dataset=halfcheetah-medium-v2 seed=0
```

You can still run the original single-file script directly:

```bash
python algorithms/unifloral.py --help
```

## Sweep Configuration

The intended workflow is to define sweeps in [`configs/config.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/config.yaml), not by typing long command lines.

Example:

```yaml
defaults:
  - db: credentials
  - schema: policy-schema
  - agent: iql
  - _self_

run: 0
db_prefix: lingwei
db_name: unifloral
seed: 0
dataset: halfcheetah-medium-v2
log: false

hydra:
  mode: MULTIRUN
  job:
    chdir: false
  sweeper:
    params:
      seed: range(0,5)
      dataset: halfcheetah-medium-v2,hopper-medium-v2,walker2d-medium-v2
      agent: iql,rebrac,td3_bc
```

This setup is designed for exactly the workflow of:

- one or more agents
- multiple D4RL datasets
- multiple random seeds
- sequential execution on a single GPU

Hydra handles the Cartesian product over the sweep parameters, while the actual training is still done inside [`algorithms/unifloral.py`](/Users/lingweizhu/Desktop/workspace/unifloral/algorithms/unifloral.py).

## Agent Defaults

Each agent has its own Hydra config file under [`configs/agent/`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/agent).

For example:

- [`configs/agent/iql.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/agent/iql.yaml)
- [`configs/agent/rebrac.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/agent/rebrac.yaml)
- [`configs/agent/td3_bc.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/agent/td3_bc.yaml)

These files contain the per-agent hyperparameter defaults that used to live in the old sweep presets. There is no environment-specific config group in this fork; datasets are swept directly through the top-level `dataset` field.

Top-level values from [`configs/config.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/config.yaml) override agent defaults when both define the same key.

## MySQL Logging

Database logging is disabled by default in [`configs/db/credentials.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/db/credentials.yaml):

```yaml
disable: true
```

To enable it:

```bash
cp configs/db/credentials-local.example.yaml configs/db/credentials-local.yaml
```

Fill in your credentials in [`configs/db/credentials-local.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/db/credentials-local.yaml), then run:

```bash
python main.py db=credentials-local
```

The default schema is defined in [`configs/schema/policy-schema.yaml`](/Users/lingweizhu/Desktop/workspace/unifloral/configs/schema/policy-schema.yaml) and writes to:

- `runs`
- `returns`
- `summary`

The database name is composed as:

```text
{db_prefix}_{db_name}
```

## What Stayed Unchanged

The performance-sensitive training path was intentionally left alone as much as possible.

- the core update logic in `unifloral.py` was not rewritten into Hydra
- the launcher builds an `Args` object and calls `run(args)`
- Hydra and MySQL logic live outside the training step implementation
- direct `tyro` usage still works through `python algorithms/unifloral.py ...`

This means the fork changes experiment orchestration and result persistence, not the core learning algorithm implementation.

## Verification Status

What has been verified in the `unifloral` conda environment:

- Hydra config composition works
- `main.py` imports and resolves agent configs correctly
- the launcher can build `Args` from Hydra config
- MySQL logging code is wired into the launcher path
- edited Python files pass syntax checks

What has not been fully verified on this macOS Apple Silicon machine:

- full D4RL MuJoCo runtime
- end-to-end training execution with actual MuJoCo environments

That limitation comes from the D4RL and `mujoco-py` stack on Apple Silicon, not from the Hydra launcher itself.

## Upstream

This fork is based on the original `unifloral` repository by Matthew Jackson, Uljad Berdica, and Jarek Liesen:

- upstream repo: [EmptyJackson/unifloral](https://github.com/EmptyJackson/unifloral)
- paper: [A Clean Slate for Offline Reinforcement Learning](https://arxiv.org/abs/2504.11453)
