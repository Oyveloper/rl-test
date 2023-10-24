import torch
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt

# import seaborn as sns
from models.reinforce import REINFORCE

env = gym.make("CartPole-v1")  # , render_mode="human")

wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(6e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
print(env.action_space)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.n
rewards_over_seeds = []


for seed in [2]:  # [1, 2, 3, 5, 8]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    agent = REINFORCE(obs_space_dims, action_space_dims)

    reward_over_episodes = []

    for episode in range(total_num_episodes):
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)

agent.save("model.pt")

# obs, info = wrapped_env.reset(seed=seed)
# done = False
# while not done:
#     env.render()
#     action = agent.sample_action(obs)

#     obs, reward, terminated, truncated, info = wrapped_env.step(action)
#     agent.rewards.append(reward)

#     done = terminated or truncated

rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
x = range(len(rewards_to_plot[0]))
plt.plot(x, rewards_to_plot[0])
# df1 = pd.DataFrame(rewards_to_plot).melt()
# df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
# sns.set(style="darkgrid", context="talk", palette="rainbow")
# sns.lineplot(x="episodes", y="reward", data=df1).set(
#     title="REINFORCE for InvertedPendulum-v4"
# )
plt.show()
