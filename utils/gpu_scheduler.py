"""
GPU调度器 - 自动管理多GPU分配
使用内存占用情况来判断GPU是否可用
"""
import fcntl
import json
import os
import subprocess
import time
import threading
from typing import List, Optional


class GPUScheduler:
    """GPU调度器 - 自动选择可用GPU"""
    
    def __init__(self, state_file: str = "/tmp/bbo_gpu_reservations.json"):
        self._lock = threading.Lock()
        self.state_file = state_file

    def _is_pid_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _with_state_lock(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        fd = os.open(self.state_file, os.O_CREAT | os.O_RDWR, 0o644)
        f = os.fdopen(fd, "r+")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        return f

    def _load_state(self, f) -> dict:
        f.seek(0)
        raw = f.read().strip()
        if not raw:
            return {"reservations": {}}
        try:
            state = json.loads(raw)
            if "reservations" not in state:
                state = {"reservations": {}}
            return state
        except json.JSONDecodeError:
            return {"reservations": {}}

    def _save_state(self, f, state: dict):
        f.seek(0)
        f.truncate(0)
        f.write(json.dumps(state, indent=2))
        f.flush()
        os.fsync(f.fileno())

    def _prune_stale_reservations(self, state: dict) -> dict:
        reservations = state.get("reservations", {})
        alive = {}
        for key, rec in reservations.items():
            pid = int(rec.get("pid", -1))
            if self._is_pid_alive(pid):
                alive[key] = rec
        state["reservations"] = alive
        return state
    
    def get_gpu_info(self) -> List[dict]:
        """获取所有GPU信息"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            return []
        if result.returncode != 0:
            return []
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_used': int(parts[2]),
                        'memory_total': int(parts[3]),
                        'memory_free': int(parts[4]),
                    })
        
        return gpu_info
    
    def get_available_gpu(self, min_memory_mb: int = 3000,
                         allow_occupied: bool = False) -> Optional[int]:
        """
        获取一个可用的GPU（基于内存空闲情况）

        Args:
            min_memory_mb: 最小可用内存（MB）
            allow_occupied: False（默认）：只选完全空闲（memory_used == 0）的 GPU；
                            True：允许选择任意有足够空闲显存的 GPU。

        Returns:
            GPU索引，如果没有可用GPU返回None
        """
        with self._lock:
            gpu_info = self.get_gpu_info()

            if allow_occupied:
                available = [g for g in gpu_info if g['memory_free'] >= min_memory_mb]
            else:
                available = [
                    g for g in gpu_info
                    if g['memory_used'] == 0 and g['memory_free'] >= min_memory_mb
                ]

            if available:
                available.sort(key=lambda x: x['memory_free'], reverse=True)
                return available[0]['index']

            return None

    def acquire_gpu(
        self,
        job_key: str,
        min_memory_mb: int = 3000,
        allow_reuse_reserved: bool = False,
        allow_occupied: bool = False,
    ) -> Optional[int]:
        """为给定任务获取并保留GPU。相同job_key会稳定返回同一张卡。

        Args:
            job_key: 任务唯一标识
            min_memory_mb: 最小可用内存（MB）
            allow_reuse_reserved: 是否允许复用同一进程内其他 job_key 保留的 GPU
            allow_occupied: 是否允许选择已有其他进程占用（但显存有空闲）的 GPU。
                            False（默认）：只选完全空闲（memory_used == 0）的 GPU，
                                         找不到直接返回 None；
                            True：允许选择任意有足够空闲显存的 GPU（包括已有进程的）。

        Returns:
            GPU索引，获取失败返回 None
        """
        with self._lock:
            f = self._with_state_lock()
            try:
                state = self._load_state(f)
                state = self._prune_stale_reservations(state)
                reservations = state.setdefault("reservations", {})

                # 同一进程切换任务时，清理该 PID 的其他旧保留，避免串行任务互相影响
                current_pid = os.getpid()
                stale_same_pid_keys = [
                    k for k, v in reservations.items()
                    if int(v.get("pid", -1)) == current_pid and k != job_key
                ]
                for k in stale_same_pid_keys:
                    del reservations[k]

                # 已保留则复用
                if job_key in reservations:
                    return int(reservations[job_key]["gpu"])

                gpu_info = self.get_gpu_info()
                if not gpu_info:
                    return None

                used_by_others = {int(v["gpu"]) for k, v in reservations.items() if k != job_key}

                # 默认模式：只选完全空闲的 GPU（memory_used == 0）
                candidates = [
                    g for g in gpu_info
                    if g["memory_free"] >= min_memory_mb
                    and g["index"] not in used_by_others
                    and g["memory_used"] == 0
                ]

                # allow_occupied=True 时，允许选择已有其他进程占用的 GPU
                if not candidates and allow_occupied:
                    candidates = [
                        g for g in gpu_info
                        if g["memory_free"] >= min_memory_mb
                        and g["index"] not in used_by_others
                    ]

                # 可选：允许复用已保留GPU（默认关闭，避免多任务撞卡）
                if not candidates and allow_reuse_reserved:
                    candidates = [g for g in gpu_info if g["memory_free"] >= min_memory_mb]

                if not candidates:
                    return None

                candidates.sort(key=lambda x: x["memory_free"], reverse=True)
                chosen = int(candidates[0]["index"])
                reservations[job_key] = {
                    "gpu": chosen,
                    "pid": os.getpid(),
                    "updated_at": time.time(),
                }
                self._save_state(f, state)
                return chosen
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                f.close()

    def release_gpu(self, job_key: str):
        """释放任务保留的GPU。"""
        with self._lock:
            f = self._with_state_lock()
            try:
                state = self._load_state(f)
                reservations = state.setdefault("reservations", {})
                if job_key in reservations:
                    del reservations[job_key]
                    self._save_state(f, state)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                f.close()
    
    def wait_for_gpu(self, min_memory_mb: int = 1000,
                     timeout: Optional[float] = None,
                     check_interval: float = 30.0,
                     allow_occupied: bool = False) -> Optional[int]:
        """
        等待直到有可用GPU

        Args:
            min_memory_mb: 最小可用内存
            timeout: 超时时间（秒），None表示无限等待
            check_interval: 检查间隔（秒）
            allow_occupied: False（默认）：只等完全空闲的 GPU；
                            True：允许等待有空闲显存但已有进程的 GPU。

        Returns:
            GPU索引，如果没有可用GPU返回None
        """
        start_time = time.time()

        while True:
            gpu_idx = self.get_available_gpu(min_memory_mb, allow_occupied=allow_occupied)
            if gpu_idx is not None:
                return gpu_idx

            if timeout is not None and (time.time() - start_time) >= timeout:
                return None

            print(f"No GPU with {min_memory_mb}MB free (allow_occupied={allow_occupied}), "
                  f"waiting... (checked every {check_interval}s)")
            time.sleep(check_interval)
    
    def print_status(self):
        """打印GPU状态"""
        gpu_info = self.get_gpu_info()
        
        print("\n" + "="*70)
        print("GPU Status:")
        print("-"*70)
        print(f"{'GPU':<6} {'Name':<25} {'Used':<10} {'Free':<10} {'Status':<15}")
        print("-"*70)
        for gpu in gpu_info:
            status = "In Use" if gpu['memory_used'] > 500 else "Available"
            print(f"GPU {gpu['index']:<3} {gpu['name']:<25} {gpu['memory_used']:>6} MB {gpu['memory_free']:>6} MB {status:<15}")
        print("="*70)


# 全局调度器实例
_gpu_scheduler = None
_scheduler_lock = threading.Lock()

def get_gpu_scheduler() -> GPUScheduler:
    """获取GPU调度器单例"""
    global _gpu_scheduler
    if _gpu_scheduler is None:
        with _scheduler_lock:
            if _gpu_scheduler is None:
                _gpu_scheduler = GPUScheduler()
    return _gpu_scheduler


if __name__ == "__main__":
    # 测试
    scheduler = get_gpu_scheduler()
    scheduler.print_status()
