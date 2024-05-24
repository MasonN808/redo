# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb

# Need to do a pip install -e redo on new machines to fix imports
from agent import linear_schedule
from buffer import ReplayBuffer, PrioritizedReplayBuffer
from config import ConfigLunarKAN_custom3
from redo import run_redo
from utils import lecun_normal_initializer, make_env, set_cuda_configuration

# Enables WandB cloud syncing
os.environ['WANDB_DISABLED'] = 'False'
os.environ["WANDB_API_KEY"] = '9762ecfe45a25eda27bb421e664afe503bb42297'

def main(cfg: ConfigLunarKAN_custom3) -> None:
    def dqn_loss(
        q_network: cfg.QNetwork,
        target_network: cfg.QNetwork,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        device: str,
        weights: torch.Tensor=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the double DQN loss."""
        with torch.no_grad():
            # Get value estimates from the target network
            target_vals = target_network.forward(next_obs)
            # Select actions through the policy network
            policy_actions = q_network(next_obs).argmax(dim=1)
            target_max = target_vals[range(len(target_vals)), policy_actions]
            # Calculate Q-target
            td_target = rewards.flatten() + gamma * target_max * (1 - dones.flatten())


        old_val = q_network(obs).gather(1, actions).squeeze()
        # For prioritized experience replay buffer
        td_error = torch.abs(old_val - td_target).detach()
        # Don't know where else to put this
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        td_error = td_error.to(device)

        if weights != None:
            # Weights are used when using PER for correcting bias (see https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py)
            mse_loss = torch.mean((old_val - td_target)**2 * weights)
        else:
            mse_loss = F.mse_loss(td_target, old_val)
        return mse_loss, old_val, td_error

    """Main training method for ReDO DQN."""
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        config=vars(cfg),
        name=run_name,
        notes=cfg.wandb_notes,
        monitor_gym=True,
        save_code=True,
        mode="online" if cfg.track else "disabled",
    )

    if cfg.save_model:
        evaluation_episode = 0
        wandb.define_metric("evaluation_episode")
        wandb.define_metric("eval/episodic_return", step_metric="evaluation_episode")

    # To get deterministic pytorch to work
    if cfg.torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    device = set_cuda_configuration(cfg.gpu)
    wrapped_envs = [make_env(cfg.env_id, cfg.seed + i, i, cfg.capture_video, run_name) for i in range(cfg.num_envs)]

    # env setup
    envs = gym.vector.SyncVectorEnv(wrapped_envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = cfg.QNetwork(envs).to(device)
    if cfg.use_lecun_init:
        # Use the same initialization scheme as jax/flax
        q_network.apply(lecun_normal_initializer)

    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
    target_network = cfg.QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    if cfg.use_per:
        rb = PrioritizedReplayBuffer(
            cfg.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
    else:
        rb = ReplayBuffer(
            cfg.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=cfg.seed)
    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                epi_return = info["episode"]["r"].item()
                # print(f"global_step={global_step}, episodic_return={epi_return}")
                wandb.log(
                    {
                        "charts/episodic_return": epi_return,
                        "charts/episodic_length": info["episode"]["l"].item(),
                        "charts/epsilon": epsilon,
                    },
                    step=global_step,
                )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            # Flag for logging
            done_update = False
            if done_update := global_step % cfg.train_frequency == 0:
                if isinstance(rb, ReplayBuffer):
                    data = rb.sample(cfg.batch_size)
                    loss, old_val, td_error = dqn_loss(
                        q_network=q_network,
                        target_network=target_network,
                        obs=data.observations,
                        next_obs=data.next_observations,
                        actions=data.actions,
                        rewards=data.rewards,
                        dones=data.dones,
                        gamma=cfg.gamma,
                        device=device
                    )
                elif isinstance(rb, PrioritizedReplayBuffer):
                    data, weights, tree_idxs = rb.sample(cfg.batch_size)
                    weights = weights.to(device)
                    loss, old_val, td_error = dqn_loss(
                        q_network=q_network,
                        target_network=target_network,
                        obs=data.observations,
                        next_obs=data.next_observations,
                        actions=data.actions,
                        rewards=data.rewards,
                        dones=data.dones,
                        gamma=cfg.gamma,
                        device=device,
                        weights=weights,
                    )
                else:
                    raise RuntimeError("Unknown buffer")

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if isinstance(rb, PrioritizedReplayBuffer):
                    # Move td_error to CPU before converting to NumPy array if using gpu
                    rb.update_priorities(tree_idxs, td_error.cpu().numpy())

                logs = {
                    "losses/td_loss": loss,
                    "losses/q_values": old_val.mean().item(),
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                }

            if global_step % cfg.redo_check_interval == 0:

                if isinstance(rb, ReplayBuffer):
                    redo_samples = rb.sample(cfg.redo_bs)
                elif isinstance(rb, PrioritizedReplayBuffer):
                    redo_samples, _, _ = rb.sample(cfg.redo_bs)
                else:
                    raise RuntimeError("Unknown buffer")

                redo_out = run_redo(
                    redo_samples,
                    model=q_network,
                    optimizer=optimizer,
                    tau=cfg.redo_tau,
                    re_initialize=cfg.enable_redo,
                    use_lecun_init=cfg.use_lecun_init,
                    use_or=cfg.use_or,
                )

                q_network = redo_out["model"]
                optimizer = redo_out["optimizer"]

                logs |= {
                    f"regularization/dormant_t={cfg.redo_tau}_fraction": redo_out["dormant_fraction"],
                    f"regularization/dormant_t={cfg.redo_tau}_count": redo_out["dormant_count"],
                    "regularization/dormant_t=0.0_fraction": redo_out["zero_fraction"],
                    "regularization/dormant_t=0.0_count": redo_out["zero_count"],
                }

            if global_step % 100 == 0 and done_update:
                # print("SPS:", int(global_step / (time.time() - start_time)))
                wandb.log(
                    logs,
                    step=global_step,
                )

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        cfg.tau * q_network_param.data + (1.0 - cfg.tau) * target_network_param.data
                    )

    if cfg.save_model:
        model_path = Path(f"runs/{run_name}/{cfg.exp_name}")
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(q_network.state_dict(), model_path / ".cleanrl_model")
        # print(f"model saved to {model_path}")
        from src.evaluate import evaluate

        episodic_returns = evaluate(
            model_path=model_path,
            make_env=make_env,
            env_id=cfg.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=cfg.QNetwork,
            device=device,
            epsilon=0.05,
            capture_video=False,
        )
        for episodic_return in episodic_returns:
            wandb.log({"evaluation_episode": evaluation_episode, "eval/episodic_return": episodic_return})
            evaluation_episode += 1

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    # Be sure to set custom q network in bash file
    cfg = tyro.cli(ConfigLunarKAN_custom3)
    main(cfg)

