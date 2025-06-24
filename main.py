import imageio
from tqdm import tqdm

from src.configs.base_config  import get_base_config
from src.environment.crafter_env import CrafterEnvWrapper

def run_random_demo():
    """Runs a simple demonstration with a random agent in the Crafter environment."""
    print("Starting a random agent demonstration...")

    # 1. Load Configuration
    config = get_base_config()
    print(f"Using configuration: {config}")

    # 2. Initialize Environment
    env = CrafterEnvWrapper(config.env)
    print(f"Environment '{config.env.env_name}' initialized.")

    # 3. Run the random agent loop
    num_episodes = 3
    for i_ in range(num_episodes):
        print(f"\n---Starting Episode {i_ + 1} / {num_episodes} ---")

        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        # Store frames for GIF
        frames = []

        # Create progress bar
        pbar = tqdm(desc=f"Episode {i_ + 1} progress")
        while not terminated and not truncated:
            # Render the environment and store the frame
            # Crafter's render returns an image array, which we need
            frame = env.render()
            frames.append(frame)

            # Have agent take a random action
            action = env.action_space.sample()

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1

            pbar.update()

        pbar.close()
        print(f"Episode finished after {step_count} steps.")
        print(f"Total reward: {total_reward:.2f}")

        # Save collected frames as a gif
        gif_path = f"{config.experiment_dir}/random_agent_episode_{i_+1}.gif"
        print(f"Saving episode GIF to {gif_path}")
        imageio.mimsave(gif_path, frames, fps=15) # Slow down to see actions
        print("GIF saved.")

    # Clean up
    env.close()
    print("\nDemonstration finished.")

if __name__ == "__main__":
    import os
    config = get_base_config()
    os.makedirs(config.experiment_dir, exist_ok=True)

    run_random_demo()