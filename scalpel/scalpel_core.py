"""
Scalpel Core — 重写的 Scalpel MCTS 优化器
完全照抄 original/main.py 的逻辑，只修改接口
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from collections import namedtuple, defaultdict
from abc import ABC, abstractmethod
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy import stats

# GPU 优化
torch.backends.cudnn.benchmark = True

__all__ = ['ScalpelCore', 'OptTransformer', 'PyTorchModelWrapper', 'ModelTrainer']


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network components (from original/main.py)
# ─────────────────────────────────────────────────────────────────────────────

class GlobalConvolution(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_heads, self.head_dim, self.head_dim))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_heads, self.head_dim))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_heads, self.head_dim, self.head_dim))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_heads, self.head_dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_heads, self.head_dim)
        o1_real = F.gelu(torch.einsum('...bi,bio->...bo', x.real, self.w1[0]) - torch.einsum('...bi,bio->...bo', x.imag, self.w1[1]) + self.b1[0])
        o1_imag = F.gelu(torch.einsum('...bi,bio->...bo', x.imag, self.w1[0]) + torch.einsum('...bi,bio->...bo', x.real, self.w1[1]) + self.b1[1])
        o2_real = (torch.einsum('...bi,bio->...bo', o1_real, self.w2[0]) - torch.einsum('...bi,bio->...bo', o1_imag, self.w2[1]) + self.b2[0])
        o2_imag = (torch.einsum('...bi,bio->...bo', o1_imag, self.w2[0]) + torch.einsum('...bi,bio->...bo', o1_real, self.w2[1]) + self.b2[1])
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = x.type(dtype)
        return x


class FourierNeuralFilter(nn.Module):
    def __init__(self, d_model: int = 128, num_heads: int = 16, expand_ratio: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_model * expand_ratio
        self.norm = nn.LayerNorm(d_model)
        self.in_proj1 = nn.Linear(self.d_model, self.d_ff)
        self.in_proj2 = nn.Linear(self.d_model, self.d_ff)
        self.dropout = nn.Dropout(dropout)
        self.conv = GlobalConvolution(d_model=self.d_ff, num_heads=num_heads)
        self.out_proj = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        y = self.in_proj1(x)
        z = self.in_proj2(x)
        y = self.conv(y)
        y = self.dropout(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y + identity


class OptTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, input_scale=1.0):
        super().__init__()
        self.input_scale = input_scale
        self.embedding = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            FourierNeuralFilter(d_model=d_model, num_heads=nhead, expand_ratio=dim_feedforward//d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.regressor = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x / self.input_scale
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.regressor(x)
        return x


class PyTorchModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)

    def predict(self, X):
        self.model.eval()
        if isinstance(X, list):
            X = np.array(X)
        if len(X.shape) == 1:
            t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
        elif len(X.shape) == 3:
            t = torch.tensor(X, dtype=torch.float32).squeeze(-1).to(self.device)
        else:
            t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(t).cpu().numpy()


class ModelTrainer:
    """从 original/main.py 复制，只修改了 device 参数处理"""
    
    _INPUT_SCALE_MAP = {
        'ackley': 5.0, 'rastrigin': 5.0, 'rosenbrock': 5.0,
        'griewank': 600.0, 'levy': 10.0, 'schwefel': 1000.0, 'michalewicz': np.pi,
    }

    def __init__(self, f_name, dims, device='cuda', lr=0.001):
        self.f_name = f_name
        self.dims = dims
        self.lr = lr
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.input_scale = self._INPUT_SCALE_MAP.get(f_name, 1.0)

    def train(self, X, y, verbose=False):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_tr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_te = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_te = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(self.device)
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

        model = OptTransformer(d_model=64, nhead=4, num_layers=2, input_scale=self.input_scale)
        criterion = nn.MSELoss()
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=1e-7)

        if verbose:
            print(f"[Model] input_scale={self.input_scale:.2f}", flush=True)

        best_loss = float('inf')
        patience, counter = 30, 0
        best_state = None

        for epoch in range(500):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_te), y_te).item()
            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}: Val Loss {val_loss:.4f}", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}", flush=True)
                    break
        if best_state:
            model.load_state_dict(best_state)
        if verbose:
            model.eval()
            with torch.no_grad():
                pred = model(X_te).cpu().numpy().reshape(-1)
            mape = metrics.mean_absolute_percentage_error(y_test.reshape(-1), pred)
            r2 = stats.pearsonr(pred, y_test.reshape(-1))[0]**2
            print(f"MAPE: {mape:.5f}, R2: {r2:.4f}", flush=True)
        return PyTorchModelWrapper(model, self.device)


# ─────────────────────────────────────────────────────────────────────────────
# Scalpel MCTS core (完全照抄 original/main.py 的逻辑)
# ─────────────────────────────────────────────────────────────────────────────

_OT = namedtuple("opt_task", "tup value terminal")


class _Node(ABC):
    @abstractmethod
    def find_children(self):
        return set()

    @abstractmethod
    def is_terminal(self):
        return True

    def __hash__(self):
        return 123456789

    def __eq__(self, other):
        return True


class OptTaskNode(_OT, _Node):
    """从 original/main.py 的 opt_task 复制"""
    
    def find_children(self, action, f, model, iteration=0, use_continuous=True):
        if self.terminal:
            return set()

        if use_continuous:
            base_step = 0.1 if f.dims <= 20 else 1.0
            step_size = base_step * (0.95 ** (iteration // 20))

            all_tup = []
            for index in action:
                tup = list(self.tup)
                flip = random.randint(0, 5)
                progress = min(1.0, iteration / 500.0)

                if flip == 0:
                    multiplier = 0.5 + random.random() if progress < 0.5 else 0.8 + 0.4 * random.random()
                    tup[index] += step_size * multiplier
                elif flip == 1:
                    multiplier = 0.5 + random.random() if progress < 0.5 else 0.8 + 0.4 * random.random()
                    tup[index] -= step_size * multiplier
                elif flip == 2:
                    idxs = np.random.randint(0, len(tup), int(f.dims / 5))
                    for i in idxs:
                        tup[i] += np.random.randn() * step_size * 2
                elif flip == 3:
                    idxs = np.random.randint(0, len(tup), int(f.dims / 10))
                    for i in idxs:
                        tup[i] += np.random.randn() * step_size * 2
                elif flip in [4, 5]:
                    range_scale = 0.3 * (1 - 0.5 * progress)
                    tup[index] += np.random.randn() * (f.ub[index] - f.lb[index]) * range_scale

                tup = np.clip(tup, f.lb, f.ub)
                all_tup.append(tup)
        else:
            turn = 0.1 if f.dims <= 20 else 1.0
            aaa = []
            for i in range(f.dims):
                aaa_i = np.arange(f.lb[i], f.ub[i] + turn, turn)
                if len(aaa_i) == 0:
                    aaa_i = np.array([f.lb[i]])
                aaa.append(aaa_i)
            all_tup = []
            for index in action:
                tup = list(self.tup)
                flip = random.randint(0, 5)

                if flip == 0:
                    tup[index] += turn
                elif flip == 1:
                    tup[index] -= turn
                elif flip == 2:
                    idxs = np.random.randint(0, len(tup), int(f.dims / 5))
                    for i in idxs:
                        tup[i] = np.random.choice(aaa[i])
                elif flip == 3:
                    idxs = np.random.randint(0, len(tup), int(f.dims / 10))
                    for i in idxs:
                        tup[i] = np.random.choice(aaa[i])
                elif flip in [4, 5]:
                    tup[index] = np.random.choice(aaa[index])

                tup = np.clip(tup, f.lb, f.ub)
                all_tup.append(tup)

        all_value = model.predict(np.array(all_tup))
        return {OptTaskNode(tuple(t), v.item(), False) for t, v in zip(all_tup, all_value)}

    def is_terminal(self):
        return self.terminal


class ScalpelCore:
    """完全照抄 original/main.py 的 Scalpel 逻辑"""

    _ROLLOUT_ROUNDS = {
        'ackley': 200, 'rastrigin': 200, 'rosenbrock': 200,
        'griewank': 200, 'levy': 200, 'schwefel': 200,
    }
    _RATIO = 0.02

    def __init__(self, func, func_name='ackley', dims=10,
                 use_continuous=True, device='cuda', rollout_rounds=None):
        self.func = func
        self.func_name = func_name
        self.dims = dims
        self.use_continuous = use_continuous
        self.device = device if torch.cuda.is_available() else 'cpu'

        # MCTS 状态
        self.N = defaultdict(int)
        self.children = dict()
        self.iteration = 0
        self.exploration_weight = self._RATIO

        # Rollout 参数
        self._rollout_rounds = rollout_rounds or self._ROLLOUT_ROUNDS.get(func_name, 200)

        # 模型（由 update() 设置）
        self.model = None

    # ── MCTS 核心方法（从 original/main.py 复制）───────────────────────────────

    def choose(self, node):
        """UCT-based child selection — 完全照抄 main.py"""
        if node.is_terminal():
            raise RuntimeError('choose called on terminal node')

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            return n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n] + 1)
            )

        media_node = max(self.children[node], key=uct)
        ind = np.random.choice(len(self.children[node]), 2, replace=True)
        children_list = list(self.children[node])
        node_rand = [children_list[i].tup for i in ind]

        if uct(media_node) > uct(node):
            return media_node, node_rand
        return node, node_rand

    def _expand(self, node):
        action = list(range(len(node.tup)))
        self.children[node] = node.find_children(
            action, self.func, self.model, self.iteration, self.use_continuous
        )

    def _backpropagate(self, path):
        self.N[path] += 1

    def do_rollout(self, node):
        self._expand(node)
        self._backpropagate(path=node)

    def data_process(self, X, boards):
        """过滤掉已经在 X 中的点（优化4：使用集合差集替代 Python 循环）。"""
        if len(boards) == 0:
            return np.array([])
        boards = np.unique(boards, axis=0)
        # O(n·D) 构建已访问集合，后续查找 O(1)
        visited_set = {tuple(xi) for xi in X}
        new_x = [b for b in boards if tuple(b) not in visited_set]
        return np.array(new_x)

    def most_visit_node(self, X, top_n):
        """返回访问次数最多的子节点（优化4：使用集合追踪已有点，减少 np.all 重复扫描）。"""
        if not self.children:
            return np.array([])
        # O(n·D) 构建已访问集合，后续查找 O(1)
        visited_set = {tuple(xi) for xi in X}
        unvisited = [(child, self.N[child]) for child in self.children if tuple(child.tup) not in visited_set]
        if not unvisited:
            return np.array([])
        children_vals, children_N = zip(*unvisited)
        idx = np.argsort(children_N)[-min(top_n, len(children_N)):]
        return np.array([children_vals[i].tup for i in idx])

    def single_rollout(self, X, rollout_round, board_uct, num_list=[5, 1, 1]):
        """
        单起点 MCTS rollout
        从 original/main.py 的 single_rollout 复制
        """
        boards, boards_rand = [], []

        for _ in range(rollout_round):
            self.do_rollout(board_uct)
            board_uct, board_rand = self.choose(board_uct)
            boards.append(list(board_uct.tup))
            boards_rand.append(list(board_rand))

        X_most_visit = self.most_visit_node(X, num_list[1])
        new_x = self.data_process(X, boards)

        try:
            new_pred = self.model.predict(new_x).reshape(-1)
        except:
            new_pred = np.array([])

        boards_rand = np.vstack(boards_rand) if len(boards_rand) > 0 else np.array([])
        new_rands = self.data_process(X, boards_rand) if len(boards_rand) > 0 else np.array([])

        top_X, X_rand2 = [], []

        if len(new_x) >= num_list[0]:
            ind = np.argsort(new_pred)[-num_list[0]:]
            top_X = new_x[ind]
            if len(new_rands) > 0:
                X_rand2 = new_rands[np.random.randint(0, len(new_rands), num_list[2])]
        elif len(new_x) == 0 and len(new_rands) > 0:
            pred_rand = self.model.predict(new_rands).reshape(-1)
            ind = np.argsort(pred_rand)[-min(num_list[0], len(pred_rand)):]
            top_X = new_rands[ind]
            X_rand2 = new_rands[np.random.randint(0, len(new_rands), num_list[2])]
        elif len(new_x) > 0:
            top_X = new_x
            needed = num_list[0] + num_list[2] - len(top_X)
            if len(new_rands) > 0:
                X_rand2 = new_rands[np.random.randint(0, len(new_rands), min(needed, len(new_rands)))]

        components = [arr for arr in [X_most_visit, top_X, X_rand2] if len(arr) > 0]
        if not components:
            return np.array([])
        try:
            return np.concatenate(components)
        except:
            return np.vstack(components)

    def rollout(self, X, y, iteration=0):
        """
        主要入口：从 original/main.py 的 rollout 方法复制
        """
        self.iteration = iteration

        if self.use_continuous and iteration % 10 == 0:
            base_step = 0.1 if self.dims <= 20 else 1.0
            current_step = base_step * (0.95 ** (iteration // 20))
            print(f"[Continuous] iter={iteration}, step_size={current_step:.4f}", flush=True)

        ratio = self._RATIO

        if self.func_name in ('rastrigin', 'ackley', 'levy'):
            index_max = np.argmax(y)
            initial_X = np.atleast_2d(X)[index_max]
            val = self.model.predict(initial_X.reshape(1, -1)).item()
            board_uct = OptTaskNode(tuple(initial_X), val, False)

            self.exploration_weight = ratio * abs(np.max(y))
            # 关键：使用正确的 num_list！
            num_list = [18, 2, 0] if self.func_name == 'rastrigin' else [15, 3, 2]
            return self.single_rollout(X, self._rollout_rounds, board_uct, num_list)
        else:
            UCT_low = (iteration % 100 >= 80)
            ind = np.argsort(y)
            candidates = np.unique(np.atleast_2d(X)[ind[-5:]], axis=0)
            X_top_all = []

            for i in range(min(3, len(candidates))):
                init_x = candidates[-(i + 1)]
                val = self.model.predict(init_x.reshape(1, -1)).item()
                exp_weight = ratio * abs(val) * (0.5 if UCT_low else 1.0)
                self.exploration_weight = exp_weight
                board_uct = OptTaskNode(tuple(init_x), val, False)
                res = self.single_rollout(X, self._rollout_rounds, board_uct)
                if len(res) > 0:
                    X_top_all.append(res)

            if not X_top_all:
                return np.array([])
            return np.vstack(X_top_all)[:20]

    # ── 辅助方法 ────────────────────────────────────────────────────────────────

    def update(self, X, y):
        """
        重新训练代理模型
        """
        X = np.atleast_2d(X)
        trainer = ModelTrainer(self.func_name, self.dims, device=self.device)
        self.model = trainer.train(X, np.asarray(y, dtype=np.float32))
