import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ai_agent.behavioral_cloning import (
    load_expert_data,
    train_behavioral_cloning,
)
from env.balatro_env import ACTION_DIM, OBS_DIM, BalatroEnv
from env.feature_encoder import OBS_DIM_V2


def write_dataset(path: Path, count: int = 8, observation_dim: int = OBS_DIM) -> None:
    with path.open("w", encoding="utf-8") as data_file:
        for index in range(count):
            observation = np.zeros(observation_dim, dtype=np.float32)
            observation[index % observation_dim] = 1.0
            action = np.zeros(ACTION_DIM, dtype=np.int64)
            action[0] = index % 2
            action[1 + (index % 4)] = 1
            data_file.write(
                json.dumps(
                    {
                        "schema_version": 2,
                        "obs": observation.tolist(),
                        "action": action.tolist(),
                        "raw_state": {"state_version": 2, "seq": index},
                    }
                )
                + "\n"
            )


class BehavioralCloningTests(unittest.TestCase):
    def test_load_expert_data_validates_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "invalid.jsonl"
            data_path.write_text(
                json.dumps({"obs": [0.0], "action": [0]}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "obs must contain"):
                load_expert_data(data_path)

    def test_train_save_and_reload_sb3_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = temp_path / "expert.jsonl"
            model_base = temp_path / "balatro_bc"
            model_path = temp_path / "balatro_bc.zip"
            write_dataset(data_path)

            summary = train_behavioral_cloning(
                data_path=data_path,
                model_path=model_base,
                epochs=1,
                batch_size=4,
                validation_fraction=0.25,
                seed=7,
                device="cpu",
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(model_path.with_suffix(".metrics.json").exists())
            self.assertEqual(summary["records"], 8)
            live_env = BalatroEnv(bridge_dir=temp_path, step_delay=0)
            model = PPO.load(model_path, env=live_env, device="cpu")
            action, _ = model.predict(
                np.zeros(OBS_DIM, dtype=np.float32),
                deterministic=True,
            )
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isin(action, [0, 1]).all())
            model.get_env().close()

    def test_loader_accepts_v2_and_rejects_mixed_versions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "expert_v2.jsonl"
            write_dataset(data_path, count=2, observation_dim=OBS_DIM_V2)

            dataset = load_expert_data(data_path)
            self.assertEqual(dataset.observation_dim, OBS_DIM_V2)

            with data_path.open("a", encoding="utf-8") as data_file:
                data_file.write(
                    json.dumps(
                        {
                            "obs": np.zeros(OBS_DIM).tolist(),
                            "action": [0, 1, 0, 0, 0, 0, 0, 0, 0],
                        }
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(ValueError, "mixes observation versions"):
                load_expert_data(data_path)


if __name__ == "__main__":
    unittest.main()
