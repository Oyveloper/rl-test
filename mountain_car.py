import argparse
from distutils.util import strtobool
import time
import numpy as np
import random
import torch
import gymnasium as gym
from tqdm import tqdm
import wandb
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.PPO import PPO
import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

seed = 2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Use GAE for advantage computation",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=32, help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.0, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )
    parser.add_argument(
        "--anneal-lr",
        type=bool,
        default=True,
        help="the target KL divergence threshold",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=5,
        help="the number of parallel game environments",
    )
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(seed, idx, capture_video, run_name):
    def thunk():
        env = gym.envs.make("HalfCheetah-v4", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


run_name = "ppo-mountain-car-continuous"

total_num_episodes = int(6e3)


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


args = parse_args()
wandb.init(
    project="master",
    monitor_gym=True,
    sync_tensorboard=True,
)

writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(seed + i, i, False, run_name) for i in range(args.num_envs)]
)

obs_space_dims = np.prod(envs.single_observation_space.shape)
action_space_dims = np.prod(envs.single_action_space.shape)


agent = PPO(obs_space_dims, action_space_dims).to(device)
lr = args.learning_rate

optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)


obs = torch.zeros(
    (args.num_steps, args.num_envs) + envs.single_observation_space.shape
).to(device)
actions = torch.zeros(
    (args.num_steps, args.num_envs) + envs.single_action_space.shape
).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

global_step = 0
start_time = time.time()
next_obs = torch.Tensor(envs.reset()[0]).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

for update in range(1, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    # ROLLOUT STAGE
    # No updates yet, we collect into batches and train later
    for step in tqdm(range(0, args.num_steps)):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.sample_action(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # Apply the action and get the next_obs and reward
        next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(
            device
        )

        # for item in info:
        #     if "episode" in item.keys():
        #         writer.add_scalar(
        #             "charts/episodic_return", item["episode"]["r"], global_step
        #         )
        #         writer.add_scalar(
        #             "charts/episodic_length", item["episode"]["l"], global_step
        #         )
        #         break

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimization time
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # Loop through each minibatch
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Do a forward, but still use the previous sampled action
                _, newlogprob, entropy, newvalue = agent.sample_action(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                # calc prob ratio
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # early stopping
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    agent.save("models/ppo-mountain-car-continuous")
    envs.close()
    writer.close()
