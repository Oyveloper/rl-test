import gymnasium as gym
from models.reinforce import REINFORCE

env = gym.make("CartPole-v1", render_mode="human")


wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.n
rewards_over_seeds = []


agent = REINFORCE(obs_space_dims, action_space_dims)

agent.load("model.pt")


obs, info = wrapped_env.reset(seed=2)

done = False
while not done:
    action = agent.sample_action(obs)

    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    agent.rewards.append(reward)

    done = terminated or truncated
