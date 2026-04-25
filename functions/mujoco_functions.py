# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import gymnasium as gym
import json
import os


_OBS_STATS_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".mujoco_obs_stats_cache.json")


def _load_obs_stats_cache():
    if not os.path.exists(_OBS_STATS_CACHE_FILE):
        return {}
    try:
        with open(_OBS_STATS_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_obs_stats_cache(cache: dict):
    try:
        with open(_OBS_STATS_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _compute_obs_statistics(env_name, n_samples=5000, seed=42):
    """通过随机探索收集观测统计量（均值和标准差）。

    这是最可靠的方法——从实际的随机 rollouts 中采样观测向量，
    再计算均值和标准差。相比用 observation_space 边界（大多是 ±inf）
    计算出的伪均值/伪标准差，真实统计量能保证随机策略产生合理尺度的 action，
    避免因归一化不当导致 action 过大/过小、线性策略几乎失效的问题。
    """
    cache_key = f"{env_name}|{n_samples}|{seed}"
    cache = _load_obs_stats_cache()
    if cache_key in cache:
        rec = cache[cache_key]
        return np.array(rec["mean"], dtype=np.float64), np.array(rec["std"], dtype=np.float64)

    np.random.seed(seed)
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    all_obs = []
    all_actions = []

    # 收集随机探索的观测和动作
    for _ in range(n_samples):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < 100:  # 每个 episode 最多 100 步
            # 直接使用环境的随机动作
            action = env.action_space.sample()

            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            all_obs.append(obs)
            all_actions.append(action)

    env.close()

    all_obs = np.array(all_obs)
    all_actions = np.array(all_actions)

    mean = np.mean(all_obs, axis=0)
    std = np.std(all_obs, axis=0)

    # 防止 std 为 0（用 1e-8 代替）
    std = np.where(std < 1e-8, 1.0, std)

    cache[cache_key] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    _save_obs_stats_cache(cache)

    return mean, std


class Swimmer:

    def __init__(self):
        self.policy_shape = (2, 8)
        self.mean         = 0
        self.std          = 1
        self.dims         = 16
        self.lb           = -1 * np.ones(self.dims)
        self.ub           = 1 * np.ones(self.dims)
        self.counter      = 0
        self.num_rollouts = 3

        # LA-MCTS hyperparameters
        self.Cp           = 20
        self.leaf_size    = 10
        self.kernel_type  = "poly"
        self.gamma_type   = "scale"
        self.ninits       = 40

        self.render = False
        self.render_frames = []
        self._env = None

    def _create_env(self):
        if self._env is None:
            import gymnasium as gym
            self._env = gym.make('Swimmer-v4', render_mode='rgb_array')
        return self._env

    def reset(self):
        self.counter = 0
        self.render_frames = []
        if self._env is not None:
            self._env.close()
            self._env = None

    def __call__(self, x, return_info=False):
        self.counter += 1
        x = np.clip(x, self.lb, self.ub)

        assert len(x) == self.dims
        assert x.ndim == 1

        env = self._create_env()
        M = x.reshape(self.policy_shape)

        returns = []
        reward_components = {
            'forward_reward': [],
            'ctrl_cost': [],
            'contact_cost': [],
            'survive_reward': [],
        }

        for i in range(self.num_rollouts):
            obs, info = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = np.dot(M, (obs - self.mean) / (self.std + 1e-8))
                # Swimmer action_space 是 [-1, 1]，无需额外 clip
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                totalr += r
                steps += 1

                if 'reward_forward' in info:
                    reward_components['forward_reward'].append(info.get('reward_forward', 0))
                if 'reward_ctrl' in info:
                    reward_components['ctrl_cost'].append(info.get('reward_ctrl', 0))
                if 'reward_contact' in info:
                    reward_components['contact_cost'].append(info.get('reward_contact', 0))
                if 'reward_survive' in info:
                    reward_components['survive_reward'].append(info.get('reward_survive', 0))

                if self.render:
                    frame = env.render()
                    if frame is not None:
                        self.render_frames.append(frame)

            returns.append(totalr)

        mean_return = np.mean(returns)

        if return_info:
            return -mean_return, {
                'reward_components': {k: np.mean(v) if v else 0 for k, v in reward_components.items()},
                'episode_length': steps,
                'total_evaluations': self.counter,
            }

        return mean_return * -1


class Hopper:

    def __init__(self):
        # 预训练的观察归一化参数（来自高质量策略的观测统计）
        # 这是业界惯例做法，比随机统计更稳定
        self.mean    = np.array([1.41599384, -0.05478602, -0.25522216, -0.25404721,
                                 0.27525085, 2.60889529,  -0.0085352, 0.0068375,
                                 -0.07123674, -0.05044839, -0.45569644])
        self.std     = np.array([0.19805723, 0.07824488,  0.17120271, 0.32000514,
                                 0.62401884, 0.82814161, 1.51915814, 1.17378372,
                                 1.87761249, 3.63482761, 5.7164752 ])
        self.dims    = 33
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.num_rollouts = 3
        self.render  = False
        self.render_frames = []
        self.policy_shape = (3, 11)
        self._env = None

        # LA-MCTS hyperparameters
        self.Cp           = 10
        self.leaf_size    = 100
        self.kernel_type  = "poly"
        self.gamma_type   = "auto"
        self.ninits       = 150

    def _create_env(self):
        if self._env is None:
            import gymnasium as gym
            self._env = gym.make('Hopper-v4', render_mode='rgb_array')
        return self._env

    def reset(self):
        self.counter = 0
        self.render_frames = []
        if self._env is not None:
            self._env.close()
            self._env = None

    def __call__(self, x, return_info=False):
        self.counter += 1
        x = np.clip(x, self.lb, self.ub)

        assert len(x) == self.dims
        assert x.ndim == 1

        env = self._create_env()
        M = x.reshape(self.policy_shape)

        returns = []
        reward_components = {
            'forward_reward': [],
            'ctrl_cost': [],
            'contact_cost': [],
            'survive_reward': [],
        }

        for i in range(self.num_rollouts):
            obs, info = env.reset()
            done = False
            totalr = 0.
            steps  = 0

            while not done:
                inputs = (obs - self.mean) / (self.std + 1e-8)
                action = np.dot(M, inputs)
                # Hopper action_space 是 [-1, 1]，无需额外 clip
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                totalr += r
                steps  += 1

                if 'reward_forward' in info:
                    reward_components['forward_reward'].append(info.get('reward_forward', 0))
                if 'reward_ctrl' in info:
                    reward_components['ctrl_cost'].append(info.get('reward_ctrl', 0))
                if 'reward_contact' in info:
                    reward_components['contact_cost'].append(info.get('reward_contact', 0))
                if 'reward_survive' in info:
                    reward_components['survive_reward'].append(info.get('reward_survive', 0))

                if self.render:
                    frame = env.render()
                    if frame is not None:
                        self.render_frames.append(frame)

            returns.append(totalr)

        mean_return = np.mean(returns)

        if return_info:
            return -mean_return, {
                'reward_components': {k: np.mean(v) if v else 0 for k, v in reward_components.items()},
                'episode_length': steps,
                'total_evaluations': self.counter,
            }

        return mean_return * -1


class HalfCheetah:
    """HalfCheetah environment - 102 params (6*17)"""

    def __init__(self):
        # obs_dim = 17, action_dim = 6
        self.dims = 102  # 6 * 17 = 102
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.num_rollouts = 3
        self.render = False
        self.render_frames = []
        self.policy_shape = (6, 17)
        self._env = None

        # 通过真实随机探索计算观测统计量
        print("Computing HalfCheetah observation statistics (sampling 5000 steps)...", flush=True)
        self.mean, self.std = _compute_obs_statistics('HalfCheetah-v4', n_samples=5000, seed=42)
        print(f"  HalfCheetah: mean range=[{self.mean.min():.3f}, {self.mean.max():.3f}], "
              f"std range=[{self.std.min():.3f}, {self.std.max():.3f}]", flush=True)

        # LA-MCTS hyperparameters
        self.Cp = 10
        self.leaf_size = 100
        self.kernel_type = "poly"
        self.gamma_type = "auto"
        self.ninits = 150

    def _create_env(self):
        if self._env is None:
            import gymnasium as gym
            self._env = gym.make('HalfCheetah-v4', render_mode='rgb_array')
        return self._env

    def reset(self):
        self.counter = 0
        self.render_frames = []
        if self._env is not None:
            self._env.close()
            self._env = None

    def __call__(self, x, return_info=False):
        self.counter += 1
        x = np.clip(x, self.lb, self.ub)

        assert len(x) == self.dims
        assert x.ndim == 1

        env = self._create_env()
        M = x.reshape(self.policy_shape)

        returns = []
        reward_components = {
            'forward_reward': [],
            'ctrl_cost': [],
            'contact_cost': [],
            'survive_reward': [],
        }

        for i in range(self.num_rollouts):
            obs, info = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                inputs = (obs - self.mean) / (self.std + 1e-8)
                action = np.dot(M, inputs)
                # HalfCheetah action_space 是 [-1, 1]^6，必须 clip
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                totalr += r
                steps += 1

                if 'reward_forward' in info:
                    reward_components['forward_reward'].append(info.get('reward_forward', 0))
                if 'reward_ctrl' in info:
                    reward_components['ctrl_cost'].append(info.get('reward_ctrl', 0))
                if 'reward_contact' in info:
                    reward_components['contact_cost'].append(info.get('reward_contact', 0))
                if 'reward_survive' in info:
                    reward_components['survive_reward'].append(info.get('reward_survive', 0))

                if self.render:
                    frame = env.render()
                    if frame is not None:
                        self.render_frames.append(frame)

            returns.append(totalr)

        mean_return = np.mean(returns)

        if return_info:
            return -mean_return, {
                'reward_components': {k: np.mean(v) if v else 0 for k, v in reward_components.items()},
                'episode_length': steps,
                'total_evaluations': self.counter,
            }

        return mean_return * -1


class Ant:
    """Ant environment - 216 params (8*27)"""

    def __init__(self):
        # obs_dim = 27, action_dim = 8
        self.dims = 216  # 8 * 27 = 216
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.num_rollouts = 3
        self.render = False
        self.render_frames = []
        self.policy_shape = (8, 27)
        self._env = None

        print("Computing Ant observation statistics (sampling 5000 steps)...", flush=True)
        self.mean, self.std = _compute_obs_statistics('Ant-v4', n_samples=5000, seed=42)
        print(f"  Ant: mean range=[{self.mean.min():.3f}, {self.mean.max():.3f}], "
              f"std range=[{self.std.min():.3f}, {self.std.max():.3f}]", flush=True)

        self.Cp = 5
        self.leaf_size = 200
        self.kernel_type = "poly"
        self.gamma_type = "auto"
        self.ninits = 300

    def _create_env(self):
        if self._env is None:
            import gymnasium as gym
            self._env = gym.make('Ant-v4', render_mode='rgb_array')
        return self._env

    def reset(self):
        self.counter = 0
        self.render_frames = []
        if self._env is not None:
            self._env.close()
            self._env = None

    def __call__(self, x, return_info=False):
        self.counter += 1
        x = np.clip(x, self.lb, self.ub)

        assert len(x) == self.dims
        assert x.ndim == 1

        env = self._create_env()
        M = x.reshape(self.policy_shape)

        returns = []
        reward_components = {
            'forward_reward': [],
            'ctrl_cost': [],
            'contact_cost': [],
            'survive_reward': [],
        }

        for i in range(self.num_rollouts):
            obs, info = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                inputs = (obs - self.mean) / (self.std + 1e-8)
                action = np.dot(M, inputs)
                # Ant action_space 是 [-1, 1]^8，必须 clip
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                totalr += r
                steps += 1

                if 'reward_forward' in info:
                    reward_components['forward_reward'].append(info.get('reward_forward', 0))
                if 'reward_ctrl' in info:
                    reward_components['ctrl_cost'].append(info.get('reward_ctrl', 0))
                if 'reward_contact' in info:
                    reward_components['contact_cost'].append(info.get('reward_contact', 0))
                if 'reward_survive' in info:
                    reward_components['survive_reward'].append(info.get('reward_survive', 0))

                if self.render:
                    frame = env.render()
                    if frame is not None:
                        self.render_frames.append(frame)

            returns.append(totalr)

        mean_return = np.mean(returns)

        if return_info:
            return -mean_return, {
                'reward_components': {k: np.mean(v) if v else 0 for k, v in reward_components.items()},
                'episode_length': steps,
                'total_evaluations': self.counter,
            }

        return mean_return * -1


class Walker2d:
    """Walker2d environment - 102 params (6*17)"""

    def __init__(self):
        # obs_dim = 17, action_dim = 6
        self.dims = 102  # 6 * 17 = 102
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.num_rollouts = 3
        self.render = False
        self.render_frames = []
        self.policy_shape = (6, 17)
        self._env = None

        print("Computing Walker2d observation statistics (sampling 5000 steps)...", flush=True)
        self.mean, self.std = _compute_obs_statistics('Walker2d-v4', n_samples=5000, seed=42)
        print(f"  Walker2d: mean range=[{self.mean.min():.3f}, {self.mean.max():.3f}], "
              f"std range=[{self.std.min():.3f}, {self.std.max():.3f}]", flush=True)

        self.Cp = 10
        self.leaf_size = 100
        self.kernel_type = "poly"
        self.gamma_type = "auto"
        self.ninits = 150

    def _create_env(self):
        if self._env is None:
            import gymnasium as gym
            self._env = gym.make('Walker2d-v4', render_mode='rgb_array')
        return self._env

    def reset(self):
        self.counter = 0
        self.render_frames = []
        if self._env is not None:
            self._env.close()
            self._env = None

    def __call__(self, x, return_info=False):
        self.counter += 1
        x = np.clip(x, self.lb, self.ub)

        assert len(x) == self.dims
        assert x.ndim == 1

        env = self._create_env()
        M = x.reshape(self.policy_shape)

        returns = []
        reward_components = {
            'forward_reward': [],
            'ctrl_cost': [],
            'contact_cost': [],
            'survive_reward': [],
        }

        for i in range(self.num_rollouts):
            obs, info = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                inputs = (obs - self.mean) / (self.std + 1e-8)
                action = np.dot(M, inputs)
                # Walker2d action_space 是 [-1, 1]^6，必须 clip
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                totalr += r
                steps += 1

                if 'reward_forward' in info:
                    reward_components['forward_reward'].append(info.get('reward_forward', 0))
                if 'reward_ctrl' in info:
                    reward_components['ctrl_cost'].append(info.get('reward_ctrl', 0))
                if 'reward_contact' in info:
                    reward_components['contact_cost'].append(info.get('reward_contact', 0))
                if 'reward_survive' in info:
                    reward_components['survive_reward'].append(info.get('reward_survive', 0))

                if self.render:
                    frame = env.render()
                    if frame is not None:
                        self.render_frames.append(frame)

            returns.append(totalr)

        mean_return = np.mean(returns)

        if return_info:
            return -mean_return, {
                'reward_components': {k: np.mean(v) if v else 0 for k, v in reward_components.items()},
                'episode_length': steps,
                'total_evaluations': self.counter,
            }

        return mean_return * -1


class Humanoid:
    """Humanoid environment - 6392 params (17*376)"""

    def __init__(self):
        # obs_dim = 376, action_dim = 17
        self.dims = 6392  # 17 * 376 = 6392
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.num_rollouts = 3
        self.render = False
        self.render_frames = []
        self.policy_shape = (17, 376)
        self._env = None

        print("Computing Humanoid observation statistics (sampling 5000 steps)...", flush=True)
        self.mean, self.std = _compute_obs_statistics('Humanoid-v4', n_samples=5000, seed=42)
        print(f"  Humanoid: mean range=[{self.mean.min():.3f}, {self.mean.max():.3f}], "
              f"std range=[{self.std.min():.3f}, {self.std.max():.3f}]", flush=True)

        self.Cp = 3
        self.leaf_size = 500
        self.kernel_type = "poly"
        self.gamma_type = "auto"
        self.ninits = 500

    def _create_env(self):
        if self._env is None:
            import gymnasium as gym
            self._env = gym.make('Humanoid-v4', render_mode='rgb_array')
        return self._env

    def reset(self):
        self.counter = 0
        self.render_frames = []
        if self._env is not None:
            self._env.close()
            self._env = None

    def __call__(self, x, return_info=False):
        self.counter += 1
        x = np.clip(x, self.lb, self.ub)

        assert len(x) == self.dims
        assert x.ndim == 1

        env = self._create_env()
        M = x.reshape(self.policy_shape)

        returns = []
        reward_components = {
            'forward_reward': [],
            'ctrl_cost': [],
            'contact_cost': [],
            'survive_reward': [],
        }

        for i in range(self.num_rollouts):
            obs, info = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                inputs = (obs - self.mean) / (self.std + 1e-8)
                action = np.dot(M, inputs)
                # Humanoid action_space 是 [-0.4, 0.4]^17，必须 clip
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                totalr += r
                steps += 1

                if 'reward_forward' in info:
                    reward_components['forward_reward'].append(info.get('reward_forward', 0))
                if 'reward_ctrl' in info:
                    reward_components['ctrl_cost'].append(info.get('reward_ctrl', 0))
                if 'reward_contact' in info:
                    reward_components['contact_cost'].append(info.get('reward_contact', 0))
                if 'reward_survive' in info:
                    reward_components['survive_reward'].append(info.get('reward_survive', 0))

                if self.render:
                    frame = env.render()
                    if frame is not None:
                        self.render_frames.append(frame)

            returns.append(totalr)

        mean_return = np.mean(returns)

        if return_info:
            return -mean_return, {
                'reward_components': {k: np.mean(v) if v else 0 for k, v in reward_components.items()},
                'episode_length': steps,
                'total_evaluations': self.counter,
            }

        return mean_return * -1
