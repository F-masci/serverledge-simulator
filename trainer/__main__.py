"""Punto di ingresso CLI per l'addestramento dello scaler PPO con Ray RLlib."""

import argparse
from pathlib import Path

from .ppo import train_rllib_ppo


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Addestra lo scaler regionale PPO tramite Gymnasium e Ray RLlib',
    )
    parser.add_argument('scenario', type=Path, help='File di scenario YAML, TOML o JSON')
    parser.add_argument('--region', type=str, default=None, help='Regione target (default: primary_region)')
    parser.add_argument('--iterations', type=int, default=10, help='Numero di iterazioni RLlib (default: 10)')
    parser.add_argument('--max-steps', type=int, default=50, help='Passi Delta massimi per episodio (default: 50)')
    parser.add_argument('--max-arrivals', type=int, default=None, help='Limite arrivi esterni per episodio')
    parser.add_argument('--max-time', type=float, default=None, help='Orizzonte temporale episodio in secondi')
    parser.add_argument('--workers', type=int, default=0, help='Numero di EnvRunner paralleli Ray (default: 0)')
    parser.add_argument('--gpus', type=float, default=0.0, help='GPU per learner (es. 1.0 per CUDA, default: 0.0)')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Directory in cui salvare il checkpoint RLlib')
    parser.add_argument('--seed', type=int, default=None, help='Seed opzionale (default: seed dello scenario)')
    args = parser.parse_args()

    result = train_rllib_ppo(
        args.scenario,
        num_iterations=args.iterations,
        region=args.region,
        max_arrivals=args.max_arrivals,
        max_time_s=args.max_time,
        max_steps=args.max_steps,
        num_env_runners=args.workers,
        num_gpus_per_learner=args.gpus,
        checkpoint_dir=args.checkpoint,
        seed=args.seed,
    )
    for row in result['history']:
        print(
            f"Iter {row['training_iteration']:>3} | "
            f"return_mean={row['episode_return_mean']} | "
            f"len_mean={row['episode_len_mean']} | "
            f"steps={row['num_env_steps_sampled_lifetime']}"
        )
    if result['checkpoint_path'] is not None:
        print(f"Checkpoint salvato in: {result['checkpoint_path']}")


if __name__ == '__main__':
    main()
