"""
PPO with an attached autoencoder for safe action projection.

Adapted from Safe-Policy-Optimization (https://github.com/PKU-Alignment/Safe-Policy-Optimization).
Requires the SafePO package (safepo) to be installed.

The autoencoder can be attached in three modes:
  - posthoc: pretrained AE is frozen. Actions are projected through it at rollout
            time, but the projection penalty during policy updates is computed
            with the AE detached—the actor simply gets a "move toward the
            projected point" gradient without seeing the projection geometry.
  - e2e: pretrained AE is frozen, but the projection penalty is computed
            through the AE's forward pass (no torch.no_grad). Gradients flow
            back through the frozen AE into the actor, so the actor learns a
            projection-aware policy.
  - none: no AE; standard PPO baseline.

Running the script:
    python safe_rl/ppo_ae.py --task <ENV_ID> --ae_mode <posthoc|e2e|none> \
        [--autoencoder_path PATH] [--ae_latent_dim INT] [--ae_hidden_dim INT] \
        [--ae_num_decoders INT] [--seed INT] [--device cpu|cuda] [--use_eval]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import deque

import numpy as np
try:
    from isaacgym import gymutil
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env, make_sa_isaac_env
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from autoencoder import ConditionalConstraintAwareAutoencoder


default_cfg = {
    'total_steps': 3000000,
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.02,
    'batch_size': 64,
    'learning_iters': 40,
    'max_grad_norm': 40.0,
    'proj_action_penalty_coef': 0.05,
}

isaac_gym_specific_cfg = {
    'total_steps': 3,
    'steps_per_epoch': 32768,
    'hidden_sizes': [1024, 1024, 512],
    'gamma': 0.96,
    'target_kl': 0.016,
    'num_mini_batch': 4,
    'use_value_coefficient': True,
    'learning_iters': 8,
    'max_grad_norm': 1.0,
    'use_critic_norm': False,
    'proj_action_penalty_coef': 0.05,
}


def _load_autoencoder(path, act_dim, obs_dim, latent_dim, hidden_dim,
                       num_decoders, device):
    """Instantiate and load a pretrained ConditionalConstraintAwareAutoencoder."""
    ae = ConditionalConstraintAwareAutoencoder(
        action_dim=act_dim,
        state_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_decoders=num_decoders,
    ).to(device)
    ae.load_state_dict(torch.load(path, map_location=device))
    return ae


def main(args, cfg_env=None):
    # ---- seed & device -------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f'{args.device}:{args.device_id}')
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
        num_cuda = torch.cuda.device_count()
        assert 0 <= args.device_id < num_cuda, (
            f"Invalid --device-id {args.device_id}; available: 0..{num_cuda-1}"
        )
        torch.cuda.set_device(args.device_id)

    # ---- environment ---------------------------------------------------
    if args.task not in isaac_gym_map.keys():
        env, obs_space, act_space = make_sa_mujoco_env(
            num_envs=args.num_envs, env_id=args.task, seed=args.seed,
        )
        eval_env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=args.task, seed=None)
        config = default_cfg
    else:
        sim_params = parse_sim_params(args, cfg_env, None)
        env = make_sa_isaac_env(args=args, cfg=cfg_env, sim_params=sim_params)
        eval_env = env
        obs_space = env.observation_space
        act_space = env.action_space
        args.num_envs = env.num_envs
        config = isaac_gym_specific_cfg

    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]

    # ---- training schedule ---------------------------------------------
    steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
    total_steps = config.get("total_steps", args.total_steps)
    local_steps_per_epoch = steps_per_epoch // args.num_envs
    epochs = total_steps // steps_per_epoch

    # ---- policy --------------------------------------------------------
    policy = ActorVCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=config["hidden_sizes"],
    ).to(device)

    # ---- autoencoder ---------------------------------------------------
    ae_mode = args.ae_mode
    autoencoder = None

    if ae_mode != "none":
        if not args.autoencoder_path:
            raise ValueError("--autoencoder_path is required when --ae_mode != none")
        if not os.path.exists(args.autoencoder_path):
            raise FileNotFoundError(
                f"Autoencoder checkpoint not found: {args.autoencoder_path}"
            )
        latent_dim = args.ae_latent_dim if args.ae_latent_dim is not None else act_dim
        autoencoder = _load_autoencoder(
            path=args.autoencoder_path,
            act_dim=act_dim,
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=args.ae_hidden_dim,
            num_decoders=args.ae_num_decoders,
            device=device,
        )
        autoencoder.eval()
        for p in autoencoder.parameters():
            p.requires_grad_(False)

    # ---- optimizers ----------------------------------------------------
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    actor_scheduler = LinearLR(
        actor_optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs,
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=3e-4,
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=3e-4,
    )
    # ---- buffer --------------------------------------------------------
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=args.num_envs,
        gamma=config["gamma"],
    )

    # ---- logger --------------------------------------------------------
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    rew_deque = deque(maxlen=50)
    cost_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    logger.log("Start with training.")

    # ---- rollout init --------------------------------------------------
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret = np.zeros(args.num_envs)
    ep_cost = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs)

    is_isaac = args.task in isaac_gym_map.keys()

    #  TRAINING LOOP
    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        rollout_start_time = time.time()

        if (epoch + 1) % 15 == 0:
            torch.save(
                policy.actor,
                os.path.join(args.log_dir, f"ppo_actor_epoch{epoch}.pt"),
            )

        # ---- rollout ---------------------------------------------------
        for step in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(
                    obs, deterministic=False,
                )
                if autoencoder is not None:
                    projected_act = autoencoder.project_action(act, obs)
                else:
                    projected_act = act

            action_for_env = (
                projected_act.detach().squeeze()
                if is_isaac
                else projected_act.detach().squeeze().cpu().numpy()
            )
            next_obs, reward, cost, terminated, truncated, info = env.step(
                action_for_env,
            )

            ep_ret += reward.cpu().numpy() if is_isaac else reward
            ep_cost += cost.cpu().numpy() if is_isaac else cost
            ep_len += 1

            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array([
                    arr if arr is not None else np.zeros(obs.shape[-1])
                    for arr in info["final_observation"]
                ])
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"], dtype=torch.float32, device=device,
                )

            buffer.store(
                obs=obs, act=act, reward=reward, cost=cost,
                value_r=value_r, value_c=value_c, log_prob=log_prob,
            )

            obs = next_obs
            epoch_end = step >= local_steps_per_epoch - 1

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False,
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx],
                                    deterministic=False,
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        rew_deque.append(ep_ret[idx])
                        cost_deque.append(ep_cost[idx])
                        len_deque.append(ep_len[idx])
                        logger.store(**{
                            "Metrics/EpRet": np.mean(rew_deque),
                            "Metrics/EpCost": np.mean(cost_deque),
                            "Metrics/EpLen": np.mean(len_deque),
                        })
                        ep_ret[idx] = 0.0
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        logger.logged = False

                    buffer.finish_path(
                        last_value_r=last_value_r,
                        last_value_c=last_value_c,
                        idx=idx,
                    )

        rollout_end_time = time.time()

        # ---- evaluation ------------------------------------------------
        eval_start_time = time.time()
        eval_episodes = 1 if epoch < epochs - 1 else 10

        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.as_tensor(
                    eval_obs, dtype=torch.float32, device=device,
                )
                eval_rew, eval_cost_ep, eval_len_ep = 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        act_eval, _, _, _ = policy.step(
                            eval_obs, deterministic=True,
                        )
                        if autoencoder is not None:
                            act_eval = autoencoder.project_action(
                                act_eval, eval_obs,
                            )
                    next_obs_eval, r, c, term, trunc, _ = eval_env.step(
                        act_eval.detach().squeeze().cpu().numpy(),
                    )
                    eval_obs = torch.as_tensor(
                        next_obs_eval, dtype=torch.float32, device=device,
                    )
                    eval_rew += r
                    eval_cost_ep += c
                    eval_len_ep += 1
                    eval_done = term[0] or trunc[0]
                eval_rew_deque.append(eval_rew)
                eval_cost_deque.append(eval_cost_ep)
                eval_len_deque.append(eval_len_ep)

            logger.store(**{
                "Metrics/EvalEpRet": np.mean(eval_rew_deque),
                "Metrics/EvalEpCost": np.mean(eval_cost_deque),
                "Metrics/EvalEpLen": np.mean(eval_len_deque),
            })

        eval_end_time = time.time()

        # ---- policy update ---------------------------------------------
        data = buffer.get()
        old_distribution = policy.actor(data["obs"])
        advantage = data["adv_r"]

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"], data["act"], data["log_prob"],
                data["target_value_r"], data["target_value_c"],
                advantage,
            ),
            batch_size=config.get(
                "batch_size",
                args.steps_per_epoch // config.get("num_mini_batch", 1),
            ),
            shuffle=True,
        )

        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        proj_penalty_coef = config.get("proj_action_penalty_coef", 0.0)

        for _ in range(config["learning_iters"]):
            for (obs_b, act_b, log_prob_b, target_value_r_b,
                 target_value_c_b, adv_b) in dataloader:

                # -- critic losses --
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(
                    policy.reward_critic(obs_b), target_value_r_b,
                )
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(
                    policy.cost_critic(obs_b), target_value_c_b,
                )
                if config.get("use_critic_norm", True):
                    for p in policy.reward_critic.parameters():
                        loss_r += p.pow(2).sum() * 0.001
                    for p in policy.cost_critic.parameters():
                        loss_c += p.pow(2).sum() * 0.001

                # -- actor loss (clipped surrogate) --
                distribution = policy.actor(obs_b)
                log_prob_new = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob_new - log_prob_b)
                ratio_clipped = torch.clamp(ratio, 0.8, 1.2)
                loss_pi = -torch.min(ratio * adv_b, ratio_clipped * adv_b).mean()

                # -- projection penalty --
                proj_penalty = torch.tensor(0.0, device=obs_b.device)
                if autoencoder is not None and proj_penalty_coef > 0.0:
                    if ae_mode == "posthoc":
                        with torch.no_grad():
                            projected = autoencoder.project_action(
                                distribution.loc, obs_b,
                            )
                        proj_penalty = nn.functional.mse_loss(
                            distribution.loc, projected,
                        )
                    elif ae_mode == "e2e":
                        projected = autoencoder.project_action(
                            distribution.loc, obs_b,
                        )
                        proj_penalty = nn.functional.mse_loss(
                            distribution.loc, projected,
                        )

                # -- total loss & step --
                total_loss = (
                    loss_pi + (2 * loss_r + loss_c
                               if config.get("use_value_coefficient", False)
                               else loss_r + loss_c)
                    + proj_penalty_coef * proj_penalty
                )

                actor_optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])

                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()

                logger.store(**{
                    "Loss/Loss_reward_critic": loss_r.mean().item(),
                    "Loss/Loss_cost_critic": loss_c.mean().item(),
                    "Loss/Loss_actor": loss_pi.mean().item(),
                    "Loss/Loss_proj_penalty": proj_penalty.item(),
                })

            new_distribution = policy.actor(data["obs"])
            kl = (
                torch.distributions.kl.kl_divergence(
                    old_distribution, new_distribution,
                )
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break

        update_end_time = time.time()
        actor_scheduler.step()

        # ---- checkpointing ---------------------------------------------
        if (epoch + 1) % 15 == 0:
            logger.torch_save(itr=epoch)
            if not is_isaac:
                logger.save_state(
                    state_dict={"Normalizer": env.obs_rms}, itr=epoch,
                )

        # ---- logging ---------------------------------------------------
        if not logger.logged:
            logger.log_tabular("Metrics/EpRet")
            logger.log_tabular("Metrics/EpCost")
            logger.log_tabular("Metrics/EpLen")
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/TotalSteps", (epoch + 1) * steps_per_epoch)
            logger.log_tabular("Train/StopIter", update_counts)
            logger.log_tabular("Train/KL", final_kl)
            logger.log_tabular("Train/LR", actor_scheduler.get_last_lr()[0])
            logger.log_tabular("Loss/Loss_reward_critic")
            logger.log_tabular("Loss/Loss_cost_critic")
            logger.log_tabular("Loss/Loss_actor")
            logger.log_tabular("Loss/Loss_proj_penalty")
            logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular("Time/Update", update_end_time - eval_end_time)
            logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
            logger.log_tabular("Value/RewardAdv", data["adv_r"].mean().item())
            logger.log_tabular("Value/CostAdv", data["adv_c"].mean().item())
            logger.dump_tabular()

    logger.close()


if __name__ == "__main__":
    # Parse AE-specific args first (unknown args forwarded to SafePO)
    ae_parser = argparse.ArgumentParser(add_help=False)
    ae_parser.add_argument(
        "--ae_mode", choices=["posthoc", "e2e", "none"], default="posthoc",
        help="Autoencoder attachment mode",
    )
    ae_parser.add_argument(
        "--autoencoder_path", type=str, default=None,
        help="Path to pretrained autoencoder .pt file",
    )
    ae_parser.add_argument("--ae_latent_dim", type=int, default=None,
                           help="Latent dim (default: action_dim)")
    ae_parser.add_argument("--ae_hidden_dim", type=int, default=64)
    ae_parser.add_argument("--ae_num_decoders", type=int, default=1)
    ae_args, remaining_argv = ae_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_argv
    args, cfg_env = single_agent_args()

    for k, v in vars(ae_args).items():
        setattr(args, k, v)

    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = f"seed-{str(args.seed).zfill(3)}"
    relpath = f"{subfolder}-{relpath}"
    algo = os.path.basename(__file__).split(".")[0]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    args.log_dir = os.path.join(project_root, "runs", args.experiment, args.task, algo, relpath)

    main(args, cfg_env)
