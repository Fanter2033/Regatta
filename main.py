import argparse
from config import DEFAULT_MODEL_PATH as DEFAULT_MODEL

def cli_train(args):
    from train import train
    train(
        field_size=args.field_size,
        max_steps=args.max_steps,
        total_timesteps=args.timesteps,
        save_path=args.model,
    )

def cli_simulate(args):
    from simulate import generate_videos
    videos = generate_videos(
        num_episodes=args.episodes,
        model_path=args.model,
        output_dir=args.output_dir,
        field_size=args.field_size,
        max_steps=args.max_steps,
    )
    print(f"\nGenerated {len(videos)} video(s).")

def cli_evaluate(args):
    from evaluate import validate
    validate(
        num_episodes=args.episodes,
        model_path=args.model,
        field_size=args.field_size,
        max_steps=args.max_steps,
    )

def cli_app(args):
    from app import app
    app.run(debug=False, port=args.port)

def main():
    parser = argparse.ArgumentParser(
        description="SailingGym"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Train a PPO model")
    p_train.add_argument("--timesteps", type=int, default=500_000,
                         help="Total training timesteps (default: 500000)")
    p_train.add_argument("--field-size", type=int, default=400)
    p_train.add_argument("--max-steps", type=int, default=250)
    p_train.add_argument("--model", default=DEFAULT_MODEL,
                         help="Save path for the model (without .zip)")
    p_train.set_defaults(func=cli_train)

    # simulate
    p_sim = subparsers.add_parser("simulate", help="Run simulation episodes and generate videos")
    p_sim.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to record (default: 10)")
    p_sim.add_argument("--model", default=DEFAULT_MODEL)
    p_sim.add_argument("--output-dir", default=".",
                       help="Directory to save videos (default: .)")
    p_sim.add_argument("--field-size", type=int, default=400)
    p_sim.add_argument("--max-steps", type=int, default=250)
    p_sim.set_defaults(func=cli_simulate)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run validation episodes and print stats")
    p_eval.add_argument("--episodes", type=int, default=100,
                        help="Number of validation episodes (default: 100)")
    p_eval.add_argument("--model", default=DEFAULT_MODEL)
    p_eval.add_argument("--field-size", type=int, default=400)
    p_eval.add_argument("--max-steps", type=int, default=250)
    p_eval.set_defaults(func=cli_evaluate)

    # app
    p_app = subparsers.add_parser("app", help="Start the Flask web application")
    p_app.add_argument("--port", type=int, default=5000)
    p_app.set_defaults(func=cli_app)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()