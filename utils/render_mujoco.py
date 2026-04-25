#!/usr/bin/env python3
"""
MuJoCo Environment Renderer
渲染最佳策略并生成动画，同时输出多维度 reward 分解
支持生成 HTML 动画（嵌入帧序列）

用法:
    # 方式1: 作为模块导入使用
    from utils.render_mujoco import render_policy, render_all_results
    render_all_results('mujoco/results')
    
    # 方式2: 命令行使用
    conda activate BBO_Task
    python -m utils.render_mujoco --env swimmer --algorithm cmaes
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from functions.mujoco_functions import Swimmer, Hopper, HalfCheetah, Ant, Walker2d, Humanoid
import gymnasium as gym


def load_best_policy(env_name: str, algorithm: str, result_dir: str = 'mujoco/results'):
    """从结果文件加载最佳策略"""
    result_path = Path(result_dir) / env_name / algorithm / 'final_result.json'
    
    if not result_path.exists():
        print(f"结果文件不存在: {result_path}")
        return None
    
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    return np.array(data['best_x']), data['best_fx']


def render_policy(env_name: str, algorithm: str, result_dir: str = 'mujoco/results', 
                  num_frames: int = 200, output_format: str = 'html'):
    """
    渲染策略并生成动画
    
    Args:
        env_name: 环境名称 (swimmer, hopper)
        algorithm: 算法名称
        result_dir: 结果目录
        num_frames: 捕获的帧数
        output_format: 输出格式 ('html', 'gif')
    """
    
    # 加载最佳策略
    best_x, best_fx = load_best_policy(env_name, algorithm, result_dir)
    if best_x is None:
        return None
    
    print(f"加载策略: {env_name}/{algorithm}")
    print(f"  Reward: {-best_fx:.4f}")
    print(f"  Policy shape: {best_x.shape}")
    
    # 创建环境
    if env_name == 'swimmer':
        env_class = Swimmer
    elif env_name == 'hopper':
        env_class = Hopper
    elif env_name == 'halfcheetah':
        env_class = HalfCheetah
    elif env_name == 'ant':
        env_class = Ant
    elif env_name == 'walker2d':
        env_class = Walker2d
    elif env_name == 'humanoid':
        env_class = Humanoid
    else:
        raise ValueError(f"未知环境: {env_name}")
    
    mujoco_env = env_class()
    mujoco_env.render = True
    mujoco_env.render_frames = []
    
    # 创建环境实例用于渲染
    env = gym.make(f'{env_name.capitalize()}-v4', render_mode='rgb_array')
    
    M = best_x.reshape(mujoco_env.policy_shape)
    
    # 运行一个 episode 并捕获帧
    obs, info = env.reset()
    frames = []
    total_reward = 0
    
    # reward 分解收集
    reward_components = {
        'forward_reward': [],
        'ctrl_cost': [],
        'contact_cost': [],
        'survive_reward': [],
    }
    
    steps = 0
    max_steps = 1000  # 最多1000步
    
    while steps < max_steps:
        # 计算动作
        if env_name == 'swimmer':
            action = np.dot(M, (obs - mujoco_env.mean) / mujoco_env.std)
        else:
            inputs = (obs - mujoco_env.mean) / mujoco_env.std
            action = np.dot(M, inputs)
        
        # 执行动作
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 捕获帧
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # 收集 reward 组件
        if 'reward_forward' in info:
            reward_components['forward_reward'].append(info.get('reward_forward', 0))
        if 'reward_ctrl' in info:
            reward_components['ctrl_cost'].append(info.get('reward_ctrl', 0))
        if 'reward_contact' in info:
            reward_components['contact_cost'].append(info.get('reward_contact', 0))
        if 'reward_survive' in info:
            reward_components['survive_reward'].append(info.get('reward_survive', 0))
        
        total_reward += r
        steps += 1
        
        if done or len(frames) >= num_frames:
            break
    
    env.close()
    
    print(f"  Episode length: {steps}")
    print(f"  Total reward: {total_reward:.4f}")
    
    # 计算 reward 分解
    reward_summary = {}
    for key, values in reward_components.items():
        if values:
            reward_summary[key] = {
                'mean': np.mean(values),
                'sum': np.sum(values),
                'std': np.std(values),
            }
    
    print(f"\n  Reward 分解:")
    for key, stats in reward_summary.items():
        print(f"    {key}: mean={stats['mean']:.4f}, sum={stats['sum']:.4f}")
    
    # 创建输出目录
    output_dir = Path(result_dir) / env_name / algorithm
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存帧为 HTML 动画
    if output_format == 'html' and frames:
        html_path = output_dir / 'animation.html'
        create_html_animation(frames, html_path, env_name, algorithm, reward_summary, -best_fx)
        print(f"\n  HTML 动画已保存: {html_path}")
    
    # 保存 reward 分解
    reward_info = {
        'env_name': env_name,
        'algorithm': algorithm,
        'total_reward': float(total_reward),
        'episode_length': steps,
        'reward_components': reward_summary,
    }
    
    reward_path = output_dir / 'reward_breakdown.json'
    with open(reward_path, 'w') as f:
        json.dump(reward_info, f, indent=2)
    print(f"  Reward 分解已保存: {reward_path}")
    
    return reward_info, frames


def create_html_animation(frames, output_path, env_name, algorithm, reward_summary, total_reward):
    """创建 HTML 动画文件"""
    import base64
    
    # 将帧转换为 base64
    frame_b64_list = []
    for frame in frames:
        # frame 是 RGB numpy 数组
        import io
        from PIL import Image
        
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        frame_b64_list.append(img_str)
    
    # 创建 HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{env_name.capitalize()} - {algorithm.upper()} Animation</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00d4ff;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }}
        .reward-summary {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
        }}
        .reward-summary h2 {{
            color: #00d4ff;
            margin-top: 0;
        }}
        .reward-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .reward-item {{
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00d4ff;
        }}
        .reward-item.total {{
            border-left-color: #00ff88;
            grid-column: 1 / -1;
        }}
        .reward-label {{
            color: #aaa;
            font-size: 0.9em;
        }}
        .reward-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #fff;
        }}
        .canvas-container {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }}
        canvas {{
            border: 2px solid #00d4ff;
            border-radius: 10px;
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        button {{
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            border: none;
            color: white;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 25px;
            cursor: pointer;
            margin: 0 10px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.5);
        }}
        .progress-bar {{
            width: 100%;
            height: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            margin-top: 15px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            transition: width 0.1s;
        }}
        .frame-counter {{
            text-align: center;
            color: #888;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌀 {env_name.capitalize()} - {algorithm.upper()} Policy Animation</h1>
        
        <div class="reward-summary">
            <h2>📊 Reward 分解 (Multi-Dimensional Reward Breakdown)</h2>
            <div class="reward-grid">
                <div class="reward-item total">
                    <div class="reward-label">Total Reward</div>
                    <div class="reward-value">{total_reward:.2f}</div>
                </div>
"""
    
    # 添加各个 reward 组件
    reward_colors = {
        'forward_reward': '#00ff88',
        'ctrl_cost': '#ff6b6b',
        'contact_cost': '#ffd93d',
        'survive_reward': '#6bcb77',
    }
    
    for key, stats in reward_summary.items():
        color = reward_colors.get(key, '#00d4ff')
        label = key.replace('_', ' ').title()
        html_content += f"""
                <div class="reward-item" style="border-left-color: {color};">
                    <div class="reward-label">{label}</div>
                    <div class="reward-value">Sum: {stats['sum']:.2f}</div>
                    <div class="reward-label">Mean: {stats['mean']:.4f}</div>
                </div>
"""
    
    html_content += f"""
            </div>
        </div>
        
        <div class="canvas-container">
            <canvas id="animation" width="{frames[0].shape[1]}" height="{frames[0].shape[0]}"></canvas>
        </div>
        
        <div class="controls">
            <button onclick="toggleAnimation()">⏯️ Play/Pause</button>
            <button onclick="resetAnimation()">🔄 Reset</button>
            <button onclick="setSpeed(0.5)">🐢 Slow</button>
            <button onclick="setSpeed(1)">🐇 Normal</button>
            <button onclick="setSpeed(2)">⚡ Fast</button>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress"></div>
        </div>
        <div class="frame-counter">Frame: <span id="frameNum">0</span> / {len(frames)}</div>
    </div>
    
    <script>
        const frames = [
"""
    
    # 添加帧数据
    for i, img_str in enumerate(frame_b64_list):
        html_content += f'            "{img_str}"'
        if i < len(frame_b64_list) - 1:
            html_content += ',\n'
    
    html_content += """
        ];
        
        const canvas = document.getElementById('animation');
        const ctx = canvas.getContext('2d');
        const progress = document.getElementById('progress');
        const frameNum = document.getElementById('frameNum');
        
        let currentFrame = 0;
        let isPlaying = true;
        let speed = 1;
        let lastTime = 0;
        const frameTime = 1000 / 30; // 30 FPS
        
        function loadFrame(index) {
            const img = new Image();
            img.onload = function() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
            };
            img.src = 'data:image/png;base64,' + frames[index];
            progress.style.width = ((index + 1) / frames.length * 100) + '%';
            frameNum.textContent = index + 1;
        }
        
        function animate(timestamp) {
            if (!isPlaying) return;
            
            if (timestamp - lastTime >= frameTime / speed) {
                loadFrame(currentFrame);
                currentFrame = (currentFrame + 1) % frames.length;
                lastTime = timestamp;
            }
            
            requestAnimationFrame(animate);
        }
        
        function toggleAnimation() {
            isPlaying = !isPlaying;
            if (isPlaying) {
                lastTime = performance.now();
                requestAnimationFrame(animate);
            }
        }
        
        function resetAnimation() {
            currentFrame = 0;
            loadFrame(0);
        }
        
        function setSpeed(s) {
            speed = s;
        }
        
        // 启动动画
        loadFrame(0);
        lastTime = performance.now();
        requestAnimationFrame(animate);
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def render_all_results(result_dir: str = 'mujoco/results'):
    """渲染所有已存在的结果，使用 tqdm 显示进度"""
    result_path = Path(result_dir)
    
    if not result_path.exists():
        print(f"结果目录不存在: {result_dir}")
        return
    
    # 收集所有需要渲染的任务
    tasks = []
    for env_dir in result_path.iterdir():
        if not env_dir.is_dir():
            continue
        
        env_name = env_dir.name
        
        for algo_dir in env_dir.iterdir():
            if not algo_dir.is_dir():
                continue
            
            algorithm = algo_dir.name
            
            final_result = algo_dir / 'final_result.json'
            if not final_result.exists():
                continue
            
            tasks.append((env_name, algorithm))
    
    if not tasks:
        print("没有找到需要渲染的结果")
        return
    
    print(f"\n开始渲染 {len(tasks)} 个任务...")
    
    # 使用 tqdm 显示进度
    with tqdm(total=len(tasks), desc="Rendering", unit="task") as pbar:
        for env_name, algorithm in tasks:
            pbar.set_description(f"{env_name}/{algorithm}")
            
            print(f"\n{'='*50}")
            print(f"渲染: {env_name} / {algorithm}")
            print(f"{'='*50}")
            
            try:
                render_policy(env_name, algorithm, result_dir)
            except Exception as e:
                print(f"  渲染失败: {e}")
                import traceback
                traceback.print_exc()
            
            pbar.update(1)
    
    print("\n所有渲染完成!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MuJoCo 渲染器')
    parser.add_argument('--env', '-e', type=str, default='swimmer',
                        help='环境名称 (swimmer, hopper)')
    parser.add_argument('--algorithm', '-a', type=str, default='cmaes',
                        help='算法名称')
    parser.add_argument('--result-dir', '-r', type=str, default='mujoco/results',
                        help='结果目录')
    parser.add_argument('--frames', '-f', type=int, default=200,
                        help='帧数')
    parser.add_argument('--all', '-all', action='store_true',
                        help='渲染所有结果')
    
    args = parser.parse_args()
    
    if args.all:
        render_all_results(args.result_dir)
    else:
        render_policy(args.env, args.algorithm, args.result_dir, args.frames)
