import numpy as np
from stable_baselines3 import PPO
from config import DEFAULT_MODEL_PATH
from sailing_env import MultiAgentSailingZoo

def validate(
    num_episodes: int = 100,
    model_path: str = DEFAULT_MODEL_PATH,
    field_size: int = 400,
    max_steps: int = 250,
    state_dict: dict | None = None,
) -> dict:
    """Run *num_episodes* and return aggregated statistics.

    Args:
        num_episodes: How many episodes to run.
        model_path: Path to the saved PPO model (without .zip).
        field_size: Sailing field size.
        max_steps: Maximum steps per episode.
        state_dict: Optional dict for live progress reporting (web app).

    Returns:
        dict with counts, position_stats, metrics, and success_rate.
    """
    model = PPO.load(model_path)
    print(f"Model '{model_path}' loaded.")

    env = MultiAgentSailingZoo(field_size=field_size, max_steps=max_steps)

    counts = {
        "wins_boat_0": 0,
        "wins_boat_1": 0,
        "collisions": 0,
        "out_of_bounds": 0,
        "timeouts": 0,
    }

    position_stats = {
        "boat_0": {"inside": 0, "outside": 0, "win_inside": 0, "win_outside": 0},
        "boat_1": {"inside": 0, "outside": 0, "win_inside": 0, "win_outside": 0},
    }

    metrics = {
        "boat_0": {"vmg": [], "triple": []},
        "boat_1": {"vmg": [], "triple": []},
    }

    print(f"Running {num_episodes} validation episodes...")

    for i in range(num_episodes):
        observations, infos = env.reset()

        # Track which agent started on the inside
        current_inside_agent = None
        for a in env.possible_agents:
            if env.boat_states[a]["is_inside"]:
                position_stats[a]["inside"] += 1
                current_inside_agent = a
            else:
                position_stats[a]["outside"] += 1

        terminated = False
        truncated = False
        last_infos = infos

        while not (terminated or truncated):
            actions = {}
            for agent_id in env.agents:
                obs = observations[agent_id]
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = action.item()
                actions[agent_id] = action

            observations, rewards, terminations, truncations, infos = env.step(actions)
            if infos:
                last_infos = infos
            terminated = all(terminations.values())
            truncated = all(truncations.values())

        # --- Outcome ---
        if env.winner:
            if env.winner == "boat_0":
                counts["wins_boat_0"] += 1
            else:
                counts["wins_boat_1"] += 1

            if env.winner == current_inside_agent:
                position_stats[env.winner]["win_inside"] += 1
            else:
                position_stats[env.winner]["win_outside"] += 1
        else:
            if any(truncations.values()):
                counts["timeouts"] += 1
            else:
                p0 = np.array([env.boat_states["boat_0"]["x"], env.boat_states["boat_0"]["y"]])
                p1 = np.array([env.boat_states["boat_1"]["x"], env.boat_states["boat_1"]["y"]])
                if np.linalg.norm(p0 - p1) < (env.boat_radius * 2.1):
                    counts["collisions"] += 1
                else:
                    counts["out_of_bounds"] += 1

        # --- Collect metrics ---
        for a in env.possible_agents:
            ai = last_infos.get(a, {})
            metrics[a]["vmg"].append(ai.get("avg_vmg", 0))
            metrics[a]["triple"].append(ai.get("triple_turns", 0))

        # Progress reporting
        pct = int((i + 1) / num_episodes * 100)
        if state_dict is not None:
            state_dict["progress"] = pct
            state_dict["message"] = f"Episode {i + 1}/{num_episodes}"

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_episodes} episodes...")

    env.close()

    success_rate = round(
        (counts["wins_boat_0"] + counts["wins_boat_1"]) / max(1, num_episodes) * 100, 1
    )

    agents_stats = {}
    for a in ["boat_0", "boat_1"]:
        ps = position_stats[a]
        agents_stats[a] = {
            "avg_vmg": round(float(np.mean(metrics[a]["vmg"])), 2),
            "avg_triple_turns": round(float(np.mean(metrics[a]["triple"])), 2),
            "start_inside": ps["inside"],
            "start_outside": ps["outside"],
            "win_inside": ps["win_inside"],
            "win_outside": ps["win_outside"],
        }

    results = {
        "num_episodes": num_episodes,
        "counts": counts,
        "success_rate": success_rate,
        "agents": agents_stats,
    }

    _print_report(results)
    return results

def _print_report(results: dict) -> None:
    """Pretty-print validation results to stdout."""
    counts = results["counts"]
    agents = results["agents"]

    print("\n" + "=" * 80)
    print(f"{'PPO VALIDATION REPORT':^80}")
    print("=" * 80)

    print(f"{'Outcome':<25} | {'Count':<10}")
    print("-" * 80)
    for key, val in counts.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<23} | {val:<10}")

    print("-" * 80)
    print(
        f"{'Agent':<10} | {'Start IN':<10} | {'Win (IN)':<18} "
        f"| {'Start OUT':<10} | {'Win (OUT)':<18}"
    )
    print("-" * 80)
    for a, s in agents.items():
        pct_in = (s["win_inside"] / s["start_inside"] * 100) if s["start_inside"] > 0 else 0
        pct_out = (s["win_outside"] / s["start_outside"] * 100) if s["start_outside"] > 0 else 0
        print(
            f"{a:<10} | {s['start_inside']:<10} | {s['win_inside']} ({pct_in:.0f}%){'':>10} "
            f"| {s['start_outside']:<10} | {s['win_outside']} ({pct_out:.0f}%)"
        )

    print("-" * 80)
    print(f"{'Agent':<10} | {'Avg VMG':<12} | {'Avg Triple Turns':<18}")
    print("-" * 80)
    for a, s in agents.items():
        print(f"{a:<10} | {s['avg_vmg']:<12.2f} | {s['avg_triple_turns']:<18.2f}")

    print("=" * 80)
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    validate()