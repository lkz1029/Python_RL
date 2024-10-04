import gym

#깃허브 확인용 문구

# Create the LunarLander environment
env = gym.make("LunarLander-v2", render_mode="human")

# Set the seed for reproducibility
env.action_space.seed(42)

# Reset the environment and get the initial observation
observation, info = env.reset(seed=42)

# Run for 1000 steps (or fewer if terminated/truncated)
for _ in range(1000):
    # Take a random action
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    # If the episode is terminated or truncated, reset the environment
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment
env.close()
