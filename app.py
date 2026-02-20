import os
import threading
import uuid
import numpy as np
import imageio
from flask import Flask, render_template, request, jsonify, send_from_directory

from config import DEFAULT_MODEL_PATH
from train import train as run_training
from simulate import load_model, run_episode
from evaluate import validate

app = Flask(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

training_state = {
    "running": False,
    "progress": 0,           # 0‑100
    "message": "",
    "done": False,
    "model_path": None,
    "error": None,
    "cancel_requested": False,
    # store params so simulate uses the same ones
    "params": {}
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    """Start training in a background thread."""
    if training_state["running"]:
        return jsonify({"error": "Training already in progress"}), 409

    data = request.json
    field_size  = int(data.get("field_size", 400))
    max_steps   = int(data.get("max_steps", 250))
    total_timesteps = int(data.get("total_timesteps", 500_000))

    training_state.update({
        "running": True,
        "progress": 0,
        "message": "Initializing...",
        "done": False,
        "model_path": None,
        "error": None,
        "cancel_requested": False,
        "params": {
            "field_size": field_size,
            "max_steps": max_steps,
        }
    })

    thread = threading.Thread(
        target=_train_worker,
        args=(field_size, max_steps, total_timesteps),
        daemon=True
    )
    thread.start()
    return jsonify({"status": "started"})

@app.route("/train/status")
def train_status():
    return jsonify({
        "running":  training_state["running"],
        "progress": training_state["progress"],
        "message":  training_state["message"],
        "done":     training_state["done"],
        "error":    training_state["error"],
    })

@app.route("/train/cancel", methods=["POST"])
def cancel_training():
    if not training_state["running"]:
        return jsonify({"error": "No training in progress"}), 400
    training_state["cancel_requested"] = True
    return jsonify({"status": "cancel_requested"})

@app.route("/simulate", methods=["POST"])
def simulate():
    """Run one episode, produce an mp4 and stats."""
    model_path = training_state.get("model_path") or DEFAULT_MODEL_PATH
    if not os.path.exists(model_path + ".zip"):
        return jsonify({"error": "No trained model found. Train first."}), 400

    params = training_state["params"]
    field_size = params.get("field_size", 400)
    max_steps = params.get("max_steps", 250)

    model = load_model(model_path)
    result = run_episode(model, field_size=field_size, max_steps=max_steps, render=True)

    # Pad end if there was a winner (already handled in run_episode)
    frames = result["frames"]

    # Save video
    video_name = f"simulation_{uuid.uuid4().hex[:8]}.mp4"
    video_path = os.path.join(STATIC_DIR, video_name)
    imageio.mimsave(video_path, frames, fps=15)

    env = result["env"]
    last_infos = result["infos"]
    step = result["steps"]

    # Build stats
    winner = result["winner"]
    collision = False
    if not winner:
        p0 = np.array([env.boat_states["boat_0"]['x'], env.boat_states["boat_0"]['y']])
        p1 = np.array([env.boat_states["boat_1"]['x'], env.boat_states["boat_1"]['y']])
        if np.linalg.norm(p0 - p1) < (env.boat_radius * 2.1):
            collision = True

    if winner:
        outcome = f"Target reached by {winner}"
    elif collision:
        outcome = "Collision"
    elif step >= env.max_steps:
        outcome = "Timeout"
    else:
        outcome = "Out of bounds"

    stats = {
        "outcome": outcome,
        "winner": winner,
        "steps": step,
        "agents": {}
    }
    for a in env.possible_agents:
        ai = last_infos.get(a, {})
        stats["agents"][a] = {
            "avg_vmg":      round(ai.get("avg_vmg", 0), 2),
            "max_speed":    round(ai.get("max_speed", 0), 1),
            "is_winner":    ai.get("is_winner", False),
            "triple_turns": ai.get("triple_turns", 0),
        }

    return jsonify({"video_url": f"/static/{video_name}", "stats": stats})

test_state = {
    "running": False,
    "progress": 0,
    "message": "",
    "done": False,
    "error": None,
    "results": None,
}

@app.route("/test", methods=["POST"])
def run_test():
    """Run N test episodes in a background thread."""
    if test_state["running"]:
        return jsonify({"error": "Test already in progress"}), 409

    model_path = training_state.get("model_path") or DEFAULT_MODEL_PATH
    if not os.path.exists(model_path + ".zip"):
        return jsonify({"error": "No trained model found. Train first."}), 400

    data = request.json
    num_episodes = int(data.get("num_episodes", 100))

    test_state.update({
        "running": True,
        "progress": 0,
        "message": "Starting tests...",
        "done": False,
        "error": None,
        "results": None,
    })

    params = training_state["params"]
    thread = threading.Thread(
        target=_test_worker,
        args=(model_path, num_episodes, params),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "started"})

@app.route("/test/status")
def test_status():
    return jsonify({
        "running":  test_state["running"],
        "progress": test_state["progress"],
        "message":  test_state["message"],
        "done":     test_state["done"],
        "error":    test_state["error"],
        "results":  test_state["results"],
    })

def _test_worker(model_path, num_episodes, params):
    try:
        field_size = params.get("field_size", 400)
        max_steps = params.get("max_steps", 250)

        results = validate(
            num_episodes=num_episodes,
            model_path=model_path,
            field_size=field_size,
            max_steps=max_steps,
            state_dict=test_state,
        )

        test_state.update({
            "running": False,
            "progress": 100,
            "message": "Tests complete!",
            "done": True,
            "results": results,
        })

    except Exception as e:
        test_state.update({
            "running": False,
            "progress": 0,
            "message": f"Error: {e}",
            "done": False,
            "error": str(e),
        })

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

def _train_worker(field_size, max_steps, total_timesteps):
    try:
        training_state["message"] = "Building environments..."
        training_state["progress"] = 0

        save_path = run_training(
            field_size=field_size,
            max_steps=max_steps,
            total_timesteps=total_timesteps,
            verbose=0,
            save_path=DEFAULT_MODEL_PATH,
            state_dict=training_state,
        )

        if training_state.get("cancel_requested"):
            training_state.update({
                "running": False,
                "progress": 0,
                "message": "Training cancelled.",
                "done": False,
                "error": "cancelled",
                "cancel_requested": False,
            })
            return

        training_state.update({
            "running": False,
            "progress": 100,
            "message": "Training complete!",
            "done": True,
            "model_path": save_path,
        })

    except Exception as e:
        training_state.update({
            "running": False,
            "progress": 0,
            "message": f"Error: {e}",
            "done": False,
            "error": str(e),
        })

if __name__ == "__main__":
    app.run(debug=False, port=5000)