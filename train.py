import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from sailing_env import MultiAgentSailingZoo
from config import DEFAULT_MODEL_PATH

class ProgressCallback(BaseCallback):
    """Reports training progress to an external state dict (used by the web app)."""

    def __init__(self, total_timesteps: int, state_dict: dict | None = None):
        super().__init__()
        self.total = total_timesteps
        self.state = state_dict

    def _on_step(self) -> bool:
        if self.state is not None:
            if self.state.get("cancel_requested"):
                return False
            pct = min(int(self.num_timesteps / self.total * 100), 99)
            self.state["progress"] = pct
            self.state["message"] = (
                f"Training... {self.num_timesteps:,}/{self.total:,} steps"
            )
        return True

def build_vec_env(field_size: int = 400, max_steps: int = 250, num_vec_envs: int = 8):
    """Create a vectorized PettingZoo environment wrapped for Stable-Baselines3."""
    env = MultiAgentSailingZoo(field_size=field_size, max_steps=max_steps)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3"
    )
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return env

def train(
    field_size: int = 400,
    max_steps: int = 250,
    total_timesteps: int = 500_000,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 128,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    verbose: int = 1,
    save_path: str = DEFAULT_MODEL_PATH,
    state_dict: dict | None = None,
) -> str:
    """Train a PPO model on the multi-agent sailing environment.

    Args:
        field_size: Size of the sailing field.
        max_steps: Maximum steps per episode.
        total_timesteps: Total training timesteps.
        save_path: Where to save the trained model (without .zip).
        state_dict: Optional dict for live progress reporting (web app).

    Returns:
        The path the model was saved to.
    """
    env = build_vec_env(field_size=field_size, max_steps=max_steps)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=verbose,
    )

    print("--- Training started (Sailing MARL) ---")
    callback = ProgressCallback(total_timesteps, state_dict)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"--- Model saved to '{save_path}.zip' ---")
    return save_path