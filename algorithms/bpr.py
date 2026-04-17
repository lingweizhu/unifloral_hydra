from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
import os
from timeit import default_timer as timer
import warnings

import distrax
import d4rl
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from flax.training.train_state import TrainState
import gym
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tyro
import wandb

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset: str = "halfcheetah-medium-v2"
    algorithm: str = "bpr"
    num_updates: int = 1_000_000
    eval_interval: int = 2500
    eval_workers: int = 8
    eval_final_episodes: int = 1000
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    # --- Generic optimization ---
    lr: float = 1e-3
    actor_lr: float = 1e-3
    lr_schedule: str = "constant"
    batch_size: int = 1024
    gamma: float = 0.99
    polyak_step_size: float = 0.005
    norm_obs: bool = True
    # --- Actor architecture ---
    actor_num_layers: int = 3
    actor_layer_width: int = 256
    actor_ln: bool = True
    deterministic: bool = False
    deterministic_eval: bool = True
    use_tanh_mean: bool = True
    use_log_std_param: bool = False
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    # --- Critic architecture ---
    num_critics: int = 2
    critic_num_layers: int = 3
    critic_layer_width: int = 256
    critic_ln: bool = True
    aggregate_q: str = "min"
    # --- Critic loss ---
    num_critic_updates_per_step: int = 2
    policy_noise: float = 0.0
    noise_clip: float = 0.0
    use_target_actor: bool = True
    # --- Entropy loss ---
    use_entropy_loss: bool = True
    ent_coef_init: float = 1.0
    actor_entropy_coef: float = 0.0
    critic_entropy_coef: float = 1.0
    # --- BPR ---
    bpr_lambda: float = 1.0
    bpr_actor_coef: float = 1.0
    behavior_energy_lr: float = 1e-3
    behavior_energy_num_layers: int = 3
    behavior_energy_layer_width: int = 256
    behavior_energy_num_negatives: int = 16
    behavior_energy_temperature: float = 1.0
    behavior_energy_pretrain_steps: int = 100_000


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

AgentTrainState = namedtuple(
    "AgentTrainState", "actor actor_target dual_q dual_q_target alpha behavior_energy"
)
Transition = namedtuple("Transition", "obs action reward next_obs done")
RunResult = namedtuple("RunResult", "eval_history summary final_info")


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


class SoftQNetwork(nn.Module):
    args: Args
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        if self.args.norm_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(self.args.critic_num_layers):
            x = nn.Dense(self.args.critic_layer_width, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.args.critic_ln else x
        q = nn.Dense(1, bias_init=sym(3e-3), kernel_init=sym(3e-3))(x)
        return q.squeeze(-1)


class VectorQ(nn.Module):
    args: Args
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=-1,
            axis_size=self.args.num_critics,
        )
        q_values = vmap_critic(self.args, self.obs_mean, self.obs_std)(obs, action)
        return q_values


class BehaviorEnergy(nn.Module):
    args: Args
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        if self.args.norm_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(self.args.behavior_energy_num_layers):
            x = nn.Dense(
                self.args.behavior_energy_layer_width,
                bias_init=constant(0.1),
            )(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.args.critic_ln else x
        energy = nn.Dense(1, bias_init=sym(3e-3), kernel_init=sym(3e-3))(x)
        return energy.squeeze(-1)


class Actor(nn.Module):
    args: Args
    obs_mean: jax.Array
    obs_std: jax.Array
    num_actions: int

    @nn.compact
    def __call__(self, x, eval=False):
        if self.args.norm_obs:
            x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(self.args.actor_num_layers):
            x = nn.Dense(self.args.actor_layer_width, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.args.actor_ln else x

        init_fn = sym(1e-3)
        mean = nn.Dense(self.num_actions, bias_init=init_fn, kernel_init=init_fn)(x)
        if self.args.use_tanh_mean:
            mean = jnp.tanh(mean)
        if self.args.deterministic or (self.args.deterministic_eval and eval):
            assert self.args.use_tanh_mean, "Deterministic actor requires clipped mean"
            return distrax.Deterministic(mean)

        if self.args.use_log_std_param:
            log_std = self.param(
                "log_std",
                init_fn=lambda key: jnp.zeros(self.num_actions, dtype=jnp.float32),
            )
        else:
            std_fn = nn.Dense(self.num_actions, bias_init=init_fn, kernel_init=init_fn)
            log_std = std_fn(x)
        std = jnp.exp(jnp.clip(log_std, self.args.log_std_min, self.args.log_std_max))
        pi = distrax.Normal(mean, std)
        if not self.args.use_tanh_mean:
            pi = distrax.Transformed(pi, distrax.Tanh())
        return pi


class EntropyCoef(nn.Module):
    args: Args

    @nn.compact
    def __call__(self):
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.args.ent_coef_init)),
        )
        return log_ent_coef


def create_train_state(args, rng, network, dummy_input, lr, steps=None):
    if args.lr_schedule == "cosine":
        lr = optax.cosine_decay_schedule(lr, steps or args.num_updates)
    elif args.lr_schedule != "constant":
        raise ValueError(f"Invalid learning rate schedule: {args.lr_schedule}")
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr, eps=1e-5),
    )


def eval_agent(args, rng, env, agent_state):
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)
    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, args.eval_workers)
    obs = env.reset()

    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs, eval=True)
        action = pi.sample(seed=rng)
        return jnp.nan_to_num(action)

    max_episode_steps = env.env_fns[0]().spec.max_episode_steps
    while step < max_episode_steps and not returned.all():
        step += 1
        rng, rng_step = jax.random.split(rng)
        rng_step = jax.random.split(rng_step, args.eval_workers)
        action = _policy_step(rng_step, jnp.array(obs))
        obs, reward, done, info = env.step(onp.array(action))
        cum_reward += reward * ~returned
        returned |= done

    if step >= max_episode_steps and not returned.all():
        warnings.warn("Maximum steps reached before all episodes terminated")
    return cum_reward


def sample_from_buffer(buffer, batch_size, rng):
    idxs = jax.random.randint(rng, (batch_size,), 0, len(buffer.obs))
    return jax.tree_util.tree_map(lambda x: x[idxs], buffer)


def aggregate_q_values(args, q_values):
    if args.aggregate_q == "min":
        return jnp.min(q_values)
    if args.aggregate_q == "mean":
        return jnp.mean(q_values)
    if args.aggregate_q == "first":
        return q_values[0]
    raise ValueError(f"Unknown Q aggregation: {args.aggregate_q}")


def make_behavior_energy_train_step(args, energy_apply_fn, dataset, num_actions):
    """Train an EBM to assign lower energy to dataset actions than random actions."""

    def _train_step(runner_state, _):
        rng, behavior_energy = runner_state

        rng, rng_batch, rng_neg = jax.random.split(rng, 3)
        batch = sample_from_buffer(dataset, args.batch_size, rng_batch)
        neg_actions = jax.random.uniform(
            rng_neg,
            (args.batch_size, args.behavior_energy_num_negatives, num_actions),
            minval=-1.0,
            maxval=1.0,
        )
        actions = jnp.concatenate([batch.action[:, None, :], neg_actions], axis=1)
        obs = jnp.repeat(batch.obs[:, None, :], actions.shape[1], axis=1)

        @jax.value_and_grad
        def _energy_loss_fn(params):
            energy_fn = lambda obs, action: energy_apply_fn(params, obs, action)
            energies = jax.vmap(jax.vmap(energy_fn))(obs, actions)
            logits = -energies / args.behavior_energy_temperature
            return -jax.nn.log_softmax(logits, axis=1)[:, 0].mean()

        energy_loss, energy_grad = _energy_loss_fn(behavior_energy.params)
        behavior_energy = behavior_energy.apply_gradients(grads=energy_grad)
        return (rng, behavior_energy), {"behavior_energy_loss": energy_loss}

    return _train_step


r"""
          __/)
       .-(__(=:
    |\ |    \)
    \ ||
     \||
      \|
    ___|_____
    \       /
     \     /
      \___/     Agent
"""


def make_train_step(args, actor_apply_fn, q_apply_fn, alpha_apply_fn, energy_apply_fn, dataset):
    """Make JIT-compatible BPR train step."""

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        rng, rng_batch = jax.random.split(rng)
        batch = sample_from_buffer(dataset, args.batch_size, rng_batch)
        losses = {}

        if args.use_entropy_loss:
            pi = jax.vmap(lambda obs: actor_apply_fn(agent_state.actor.params, obs))
            pi_rng, rng = jax.random.split(rng)
            _, log_pi = pi(batch.obs).sample_and_log_prob(seed=pi_rng)
            entropy = -log_pi.sum(-1).mean()
            target_entropy = -batch.action.shape[-1]
            ent_diff = entropy - target_entropy
            alpha_loss_fn = jax.value_and_grad(lambda p: alpha_apply_fn(p) * ent_diff)
            alpha_loss, alpha_grad = alpha_loss_fn(agent_state.alpha.params)
            updated_alpha = agent_state.alpha.apply_gradients(grads=alpha_grad)
            agent_state = agent_state._replace(alpha=updated_alpha)
            alpha = jnp.exp(alpha_apply_fn(agent_state.alpha.params))
            losses.update({"alpha_loss": alpha_loss, "alpha": alpha})
        else:
            alpha = 0.0

        @partial(jax.value_and_grad, has_aux=True)
        def _actor_loss_function(params, rng):
            q_params = agent_state.dual_q.params

            def _compute_loss(rng, transition):
                pi = actor_apply_fn(params, transition.obs)
                rng_a1, rng_a2 = jax.random.split(rng)
                a1 = jax.lax.stop_gradient(pi.sample(seed=rng_a1))
                a2 = jax.lax.stop_gradient(pi.sample(seed=rng_a2))
                logp1 = pi.log_prob(a1).sum()
                logp2 = pi.log_prob(a2).sum()
                q1 = aggregate_q_values(args, q_apply_fn(q_params, transition.obs, a1))
                q2 = aggregate_q_values(args, q_apply_fn(q_params, transition.obs, a2))
                e1 = energy_apply_fn(
                    agent_state.behavior_energy.params, transition.obs, a1
                )
                e2 = energy_apply_fn(
                    agent_state.behavior_energy.params, transition.obs, a2
                )
                behavior_pref = jax.lax.stop_gradient(e2 - e1)
                policy_pref = (logp1 - jax.lax.stop_gradient(q1)) - (
                    logp2 - jax.lax.stop_gradient(q2)
                )
                bpr_loss = jnp.square(behavior_pref - args.bpr_lambda * policy_pref)
                return {
                    "bpr_loss": bpr_loss,
                    "bpr_behavior_pref": behavior_pref,
                    "bpr_policy_pref": policy_pref,
                    "bpr_q1": q1,
                    "bpr_q2": q2,
                    "actor_loss": args.bpr_actor_coef * bpr_loss,
                }

            rng = jax.random.split(rng, args.batch_size)
            actor_losses = jax.vmap(_compute_loss)(rng, batch)
            if args.use_entropy_loss:
                pi = jax.vmap(lambda obs: actor_apply_fn(params, obs))
                pi_rng, _ = jax.random.split(rng[0])
                _, log_pi = pi(batch.obs).sample_and_log_prob(seed=pi_rng)
                entropy_loss = log_pi.sum(-1)
                actor_losses["entropy_loss"] = entropy_loss
                actor_losses["actor_loss"] += (
                    args.actor_entropy_coef * alpha * entropy_loss
                )
            actor_losses = jax.tree_util.tree_map(jnp.mean, actor_losses)
            return actor_losses["actor_loss"], actor_losses

        rng, rng_actor = jax.random.split(rng)
        (_, actor_losses), actor_grad = _actor_loss_function(
            agent_state.actor.params, rng_actor
        )
        agent_state = agent_state._replace(
            actor=agent_state.actor.apply_gradients(grads=actor_grad)
        )
        losses.update(actor_losses)

        def _update_critics(runner_state, _):
            rng, agent_state = runner_state

            def _compute_target(rng, transition):
                next_obs = transition.next_obs
                if args.use_target_actor:
                    next_pi = actor_apply_fn(agent_state.actor_target.params, next_obs)
                else:
                    next_pi = actor_apply_fn(agent_state.actor.params, next_obs)
                rng, rng_action, rng_noise = jax.random.split(rng, 3)
                next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng_action)
                noise = jax.random.normal(rng_noise, shape=next_action.shape)
                noise *= args.policy_noise
                noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
                next_action = jnp.clip(next_action + noise, -1, 1)

                next_q = q_apply_fn(agent_state.dual_q_target.params, next_obs, next_action)
                next_v = jnp.min(next_q)
                target_losses = {"critic_next_v": next_v}
                if args.use_entropy_loss:
                    critic_entropy_loss = log_next_pi.sum()
                    next_v += args.critic_entropy_coef * alpha * critic_entropy_loss
                    target_losses["critic_entropy_loss"] = critic_entropy_loss
                next_v *= (1.0 - transition.done) * args.gamma
                return transition.reward + next_v, target_losses

            rng, rng_targets = jax.random.split(rng)
            rng_targets = jax.random.split(rng_targets, args.batch_size)
            targets, target_losses = jax.vmap(_compute_target)(rng_targets, batch)

            @jax.value_and_grad
            def _q_loss_fn(params):
                q_pred = q_apply_fn(params, batch.obs, batch.action)
                q_diff = q_pred - targets.reshape(args.batch_size, 1)
                return jnp.square(q_diff).sum(-1).mean()

            q_loss, q_grad = _q_loss_fn(agent_state.dual_q.params)
            agent_state = agent_state._replace(
                dual_q=agent_state.dual_q.apply_gradients(grads=q_grad)
            )
            critic_losses = jax.tree_util.tree_map(jnp.mean, target_losses)
            critic_losses["critic_loss"] = q_loss
            return (rng, agent_state), critic_losses

        (rng, agent_state), critic_losses = jax.lax.scan(
            _update_critics,
            (rng, agent_state),
            None,
            length=args.num_critic_updates_per_step,
        )
        losses.update(jax.tree_util.tree_map(jnp.mean, critic_losses))

        def _update_target(state, target_state):
            new_params = optax.incremental_update(
                state.params, target_state.params, args.polyak_step_size
            )
            return target_state.replace(step=target_state.step + 1, params=new_params)

        agent_state = agent_state._replace(
            dual_q_target=_update_target(agent_state.dual_q, agent_state.dual_q_target)
        )
        if args.use_target_actor:
            agent_state = agent_state._replace(
                actor_target=_update_target(agent_state.actor, agent_state.actor_target)
            )

        return (rng, agent_state), jax.tree_util.tree_map(jnp.mean, losses)

    return _train_step


def run(args: Args) -> RunResult:
    if args.deterministic:
        raise ValueError("BPR requires a stochastic actor so log probabilities exist")

    start = timer()
    rng = jax.random.PRNGKey(args.seed)
    eval_history = []
    final_info = {}

    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type="train_agent",
        )

    env = gym.vector.make(args.dataset, num_envs=args.eval_workers)
    dataset = d4rl.qlearning_dataset(gym.make(args.dataset))
    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )

    num_actions = env.single_action_space.shape[0]
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros(env.single_observation_space.shape)
    dummy_action = jnp.zeros(num_actions)

    actor_net = Actor(args, obs_mean, obs_std, num_actions)
    q_net = VectorQ(args, obs_mean, obs_std)
    energy_net = BehaviorEnergy(args, obs_mean, obs_std)
    alpha_net = EntropyCoef(args) if args.use_entropy_loss else None

    rng, rng_actor, rng_q, rng_energy, rng_alpha = jax.random.split(rng, 5)
    actor = create_train_state(args, rng_actor, actor_net, [dummy_obs], args.actor_lr)
    actor_target = (
        create_train_state(args, rng_actor, actor_net, [dummy_obs], args.actor_lr)
        if args.use_target_actor
        else None
    )
    nq = args.num_updates * args.num_critic_updates_per_step
    dual_q = create_train_state(
        args, rng_q, q_net, [dummy_obs, dummy_action], args.lr, steps=nq
    )
    behavior_energy = create_train_state(
        args,
        rng_energy,
        energy_net,
        [dummy_obs, dummy_action],
        args.behavior_energy_lr,
        steps=max(args.behavior_energy_pretrain_steps, 1),
    )
    alpha = (
        create_train_state(args, rng_alpha, alpha_net, [], args.lr)
        if args.use_entropy_loss
        else None
    )

    if args.behavior_energy_pretrain_steps > 0:
        _energy_train_step_fn = make_behavior_energy_train_step(
            args, energy_net.apply, dataset, num_actions
        )
        (rng, behavior_energy), energy_loss = jax.lax.scan(
            _energy_train_step_fn,
            (rng, behavior_energy),
            None,
            args.behavior_energy_pretrain_steps,
        )
        print(
            "Behavior energy pretrain loss:",
            float(energy_loss["behavior_energy_loss"][-1]),
        )

    agent_state = AgentTrainState(
        actor=actor,
        actor_target=actor_target,
        dual_q=dual_q,
        dual_q_target=create_train_state(
            args, rng_q, q_net, [dummy_obs, dummy_action], args.lr
        ),
        alpha=alpha,
        behavior_energy=behavior_energy,
    )

    _agent_train_step_fn = make_train_step(
        args,
        actor_net.apply,
        q_net.apply,
        alpha_net.apply if args.use_entropy_loss else None,
        energy_net.apply,
        dataset,
    )

    num_evals = args.num_updates // args.eval_interval
    for eval_idx in range(num_evals):
        (rng, agent_state), loss = jax.lax.scan(
            _agent_train_step_fn,
            (rng, agent_state),
            None,
            args.eval_interval,
        )

        rng, rng_eval = jax.random.split(rng)
        returns = eval_agent(args, rng_eval, env, agent_state)
        scores = d4rl.get_normalized_score(args.dataset, returns) * 100.0

        step = (eval_idx + 1) * args.eval_interval
        print("Step:", step, f"\t Score: {scores.mean():.2f}")
        log_dict = {
            "return": returns.mean(),
            "score": scores.mean(),
            "score_std": scores.std(),
            "num_updates": step,
            **{k: loss[k][-1] for k in loss},
        }
        eval_history.append(
            {
                "step": int(step),
                "episode": 0,
                "return": float(log_dict["return"]),
                "score": float(log_dict["score"]),
                "score_std": float(log_dict["score_std"]),
                "losses": {k: float(log_dict[k]) for k in loss},
            }
        )
        if args.log:
            wandb.log(log_dict)

    if args.eval_final_episodes > 0:
        final_iters = int(onp.ceil(args.eval_final_episodes / args.eval_workers))
        print(f"Evaluating final agent for {final_iters} iterations...")
        _rng = jax.random.split(rng, final_iters)
        rets = onp.array([eval_agent(args, _rng, env, agent_state) for _rng in _rng])
        scores = d4rl.get_normalized_score(args.dataset, rets) * 100.0
        agg_fn = lambda x, k: {k: x, f"{k}_mean": x.mean(), f"{k}_std": x.std()}
        info = agg_fn(rets, "final_returns") | agg_fn(scores, "final_scores")

        os.makedirs("final_returns", exist_ok=True)
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{args.algorithm}_{args.dataset}_{time_str}.npz"
        output_path = os.path.join("final_returns", filename)
        with open(output_path, "wb") as f:
            onp.savez_compressed(f, **info, args=asdict(args))
        final_info = {
            "path": output_path,
            "final_returns_mean": float(info["final_returns_mean"]),
            "final_returns_std": float(info["final_returns_std"]),
            "final_scores_mean": float(info["final_scores_mean"]),
            "final_scores_std": float(info["final_scores_std"]),
        }
        if args.log:
            wandb.save(output_path)

    env.close()
    if args.log:
        wandb.finish()

    reward_history = [entry["return"] for entry in eval_history]
    score_history = [entry["score"] for entry in eval_history]
    total_time = (timer() - start) / 60.0
    summary = {
        "step": int(eval_history[-1]["step"]) if eval_history else 0,
        "episode": 0,
        "auc100": float(onp.mean(reward_history)) if reward_history else 0.0,
        "norm_auc100": float(onp.mean(score_history)) if score_history else 0.0,
        "auc50": float(onp.mean(reward_history[-max(1, len(reward_history) // 2) :]))
        if reward_history
        else 0.0,
        "norm_auc50": float(
            onp.mean(score_history[-max(1, len(score_history) // 2) :])
        )
        if score_history
        else 0.0,
        "auc10": float(onp.mean(reward_history[-max(1, len(reward_history) // 10) :]))
        if reward_history
        else 0.0,
        "norm_auc10": float(
            onp.mean(score_history[-max(1, len(score_history) // 10) :])
        )
        if score_history
        else 0.0,
        "last_score": float(score_history[-1]) if score_history else 0.0,
        "time_taken": float(total_time),
    }
    return RunResult(eval_history=eval_history, summary=summary, final_info=final_info)


if __name__ == "__main__":
    run(tyro.cli(Args))
