import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from experiment import ExperimentManager, Metric, prep_cfg_for_db


log = logging.getLogger(__name__)
DB_CONFIG_KEYS = {"db", "schema", "hydra"}


def get_algorithm_module(agent_name):
    if agent_name == "bpr":
        from algorithms import bpr

        return bpr

    from algorithms import unifloral

    return unifloral


def build_args(cfg_dict, args_cls):
    from dataclasses import asdict, fields

    arg_defaults = asdict(args_cls())
    arg_field_names = {field.name for field in fields(args_cls)}
    arg_defaults.update({key: cfg_dict[key] for key in arg_field_names if key in cfg_dict})
    return args_cls(**arg_defaults)


def get_explicit_top_level_overrides():
    explicit_keys = set()
    for override in HydraConfig.get().overrides.task:
        key = override.split("=", 1)[0].lstrip("+~")
        key = key.split("@", 1)[0]
        if "." not in key:
            explicit_keys.add(key)
    return explicit_keys


def flatten_agent_config(cfg_dict, explicit_top_level_overrides=None):
    explicit_top_level_overrides = explicit_top_level_overrides or set()
    agent_cfg = cfg_dict.pop("agent", {})
    if isinstance(agent_cfg, dict):
        for key, value in agent_cfg.items():
            if key not in explicit_top_level_overrides:
                cfg_dict[key] = value
    return cfg_dict


def make_metrics(cfg_dict):
    flattened_cfg = prep_cfg_for_db(cfg_dict, DB_CONFIG_KEYS)
    exp = ExperimentManager(
        cfg_dict["db_name"],
        flattened_cfg,
        cfg_dict["db_prefix"],
        cfg_dict["db"],
    )
    metrics = {}
    for table_name, spec in cfg_dict["schema"].items():
        metrics[table_name] = Metric(
            table_name,
            spec["columns"],
            spec["primary_keys"],
            exp,
        )
    return metrics


def log_results(metrics, run_id, result):
    for entry in result.eval_history:
        metrics["returns"].add_data(
            [
                run_id,
                entry["step"],
                entry["episode"],
                entry["return"],
                entry["score"],
            ]
        )
    metrics["returns"].commit_to_database()

    summary = result.summary
    metrics["summary"].add_data(
        [
            run_id,
            summary["step"],
            summary["episode"],
            summary["auc100"],
            summary["norm_auc100"],
            summary["auc50"],
            summary["norm_auc50"],
            summary["auc10"],
            summary["norm_auc10"],
            summary["last_score"],
            summary["time_taken"],
        ]
    )
    metrics["summary"].commit_to_database()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if HydraConfig.get().mode.value == 2:
        with open_dict(cfg):
            cfg.run = int(cfg.run) + int(HydraConfig.get().job.num)

    with open_dict(cfg):
        cfg.agent_name = HydraConfig.get().runtime.choices.agent

    log.info(
        "Output directory: %s",
        HydraConfig.get().runtime.output_dir,
    )
    cfg_dict = flatten_agent_config(
        OmegaConf.to_container(cfg, resolve=True),
        get_explicit_top_level_overrides(),
    )
    metrics = make_metrics(cfg_dict)
    algorithm = get_algorithm_module(cfg_dict["agent_name"])

    result = algorithm.run(build_args(cfg_dict, algorithm.Args))
    log_results(metrics, cfg_dict["run"], result)


if __name__ == "__main__":
    main()
