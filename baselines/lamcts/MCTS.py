# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import collections
import copy as cp
import math
from collections import OrderedDict
import os.path
import numpy as np
import time
import operator
import sys
import pickle
import os
import random
from datetime import datetime
from .Node import Node
from .utils import latin_hypercube, from_unit_cube
from torch.quasirandom import SobolEngine
import torch

class MCTS:
    #############################################

    def __init__(self, lb, ub, dims, ninits, func, Cp = 1.414, leaf_size = 20,
                 kernel_type = "rbf", gamma_type = "auto", gpu_id = None):
        self.dims                    =  dims
        self.samples                 =  []
        self.nodes                   =  []
        self.Cp                      = Cp
        self.lb                      = lb
        self.ub                      = ub
        self.ninits                  = ninits
        self.func                    = func
        self.curt_best_value         = float("-inf")
        self.curt_best_sample        = None
        self.best_value_trace        = []
        self.sample_counter          = 0
        self.visualization           = False
        self.gpu_id                 = gpu_id

        self.LEAF_SAMPLE_SIZE        = leaf_size
        self.kernel_type             = kernel_type
        self.gamma_type              = gamma_type

        self.solver_type             = 'turbo' #solver can be 'bo' or 'turbo'

        print("gamma_type:", gamma_type)

        #we start the most basic form of the tree, 3 nodes and height = 1
        root = Node( parent = None, dims = self.dims, reset_id = True,
                     kernel_type = self.kernel_type, gamma_type = self.gamma_type,
                     gpu_id = self.gpu_id, lb = self.lb, ub = self.ub )
        self.nodes.append( root )

        self.ROOT = root
        self.CURT = self.ROOT
        self.init_train()

    def populate_training_data(self):
        #only keep root
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root  = Node(parent = None,   dims = self.dims, reset_id = True,
                          kernel_type = self.kernel_type, gamma_type = self.gamma_type,
                          gpu_id = self.gpu_id, lb = self.lb, ub = self.ub )
        self.nodes.append( new_root )

        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_bag( self.samples )

    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if node.is_leaf() == True and len(node.bag) > self.LEAF_SAMPLE_SIZE and node.is_svm_splittable == True:
                status.append( True  )
            else:
                status.append( False )
        return np.array( status )

    def get_split_idx(self):
        split_by_samples = np.argwhere( self.get_leaf_status() == True ).reshape(-1)
        return split_by_samples

    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False

    def dynamic_treeify(self):
        """
        Rebuild the full search tree from all accumulated samples.

        Mirrors the original LaMCTS algorithm exactly:
        - Clear all nodes, keep root
        - Assign ALL samples to root bag
        - Repeatedly split any leaf with > LEAF_SAMPLE_SIZE samples

        Called EVERY iteration so UCT x_bar and n values propagate correctly.
        """
        self.populate_training_data()
        assert len(self.ROOT.bag) == len(self.samples)
        assert len(self.nodes) == 1

        while self.is_splitable():
            to_split = self.get_split_idx()
            for nidx in to_split:
                parent = self.nodes[nidx]
                assert len(parent.bag) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable == True
                good_kid_data, bad_kid_data = parent.train_and_split()
                assert len(good_kid_data) + len(bad_kid_data) == len(parent.bag)
                assert len(good_kid_data) > 0
                assert len(bad_kid_data) > 0
                good_kid = Node(parent=parent, dims=self.dims, reset_id=False,
                                kernel_type=self.kernel_type, gamma_type=self.gamma_type,
                                gpu_id=self.gpu_id, lb=self.lb, ub=self.ub)
                bad_kid = Node(parent=parent, dims=self.dims, reset_id=False,
                               kernel_type=self.kernel_type, gamma_type=self.gamma_type,
                               gpu_id=self.gpu_id, lb=self.lb, ub=self.ub)
                good_kid.update_bag(good_kid_data)
                bad_kid.update_bag(bad_kid_data)
                parent.update_kids(good_kid=good_kid, bad_kid=bad_kid)
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)

    def collect_samples(self, sample, value=None):
        """
        Collect a sample and update best value tracker.

        Sign convention (maximization internal format):
            - All values stored in self.samples are in MAXIMIZATION format.
            - For minimization objectives, the function value f is negated: value = -f.
            - Therefore higher value = better.
            - If `value` is provided, it MUST already be in maximization format.

        Args:
            sample: the input vector
            value:  pre-computed value in maximization format, or None to auto-evaluate
        """
        if value is None:
            # func returns minimization value; negate for maximization storage
            value = self.func(sample) * -1

        if value > self.curt_best_value:
            self.curt_best_value  = value
            self.curt_best_sample = sample
            self.best_value_trace.append((value, self.sample_counter))
        self.sample_counter += 1
        self.samples.append((sample, value))
        return value

    def init_train(self):

        # here we use latin hyper space to generate init samples in the search space
        init_points = latin_hypercube(self.ninits, self.dims)
        init_points = from_unit_cube(init_points, self.lb, self.ub)

        for point in init_points:
            self.collect_samples(point)

        print("="*10 + 'collect '+ str(len(self.samples) ) +' points for initializing MCTS'+"="*10)
        print("lb:", self.lb)
        print("ub:", self.ub)
        print("Cp:", self.Cp)
        print("inits:", self.ninits)
        print("dims:", self.dims)
        print("="*58)

    def print_tree(self):
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)

    def reset_to_root(self):
        self.CURT = self.ROOT

    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples)," samples" )

    def dump_agent(self):
        node_path = 'mcts_agent'
        print("dumping the agent.....")
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)

    def dump_samples(self):
        sample_path = 'samples_'+str(self.sample_counter)
        with open(sample_path, "wb") as outfile:
            pickle.dump(self.samples, outfile)

    def dump_trace(self):
        trace_path = 'best_values_trace'
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def greedy_select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        if self.visualization == True:
            curt_node.plot_samples_and_boundary(self.func)
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_xbar() )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            if curt_node.is_leaf() == False and self.visualization == True:
                curt_node.plot_samples_and_boundary(self.func)
            print("=>", curt_node.get_name(), end=' ' )
        print("")
        return curt_node, path

    def _compute_adaptive_Cp(self):
        """
        Adapt Cp dynamically based on search progress to avoid premature convergence.

        Strategy:
        - Early  phase (few samples):  high Cp → strong exploration of tree structure
        - Mid    phase:                moderate Cp → balanced search
        - Late   phase (many samples): low Cp  → exploit the learned partition

        We use the ratio of explored leaves to the current tree height as a proxy
        for how well the search space has been partitioned.
        """
        total_nodes = len(self.nodes)
        if total_nodes <= 1:
            return self.Cp  # no tree yet

        # Count leaf nodes and their visit stats
        leaf_nodes = [n for n in self.nodes if n.is_leaf()]
        n_leaves = len(leaf_nodes)
        if n_leaves == 0:
            return self.Cp

        # Average visits per leaf
        avg_leaf_visits = np.mean([n.n for n in leaf_nodes])
        max_leaf_visits = max(n.n for n in leaf_nodes)
        min_leaf_visits = min(n.n for n in leaf_nodes)

        # Global visit imbalance: how skewed is the distribution?
        visit_spread = (max_leaf_visits - min_leaf_visits + 1)
        imbalance = visit_spread / (avg_leaf_visits + 1)

        # Progress ratio: total samples vs. what we'd ideally have per leaf
        # If each leaf had at least LEAF_SAMPLE_SIZE samples, the tree would be balanced
        ideal_samples_per_leaf = self.LEAF_SAMPLE_SIZE
        progress_ratio = self.sample_counter / (n_leaves * ideal_samples_per_leaf + 1e-9)

        # Compute Cp multiplier based on imbalance
        # High imbalance → Cp should increase to force visits to under-explored leaves
        # Progress is high → Cp should decrease to focus exploitation
        if imbalance > 5.0:
            Cp_multiplier = 2.5   # strongly favour unexplored leaves
        elif imbalance > 2.0:
            Cp_multiplier = 1.8
        elif imbalance > 1.5:
            Cp_multiplier = 1.3
        elif progress_ratio < 0.5:
            Cp_multiplier = 1.5   # early stage: moderate exploration
        elif progress_ratio < 1.0:
            Cp_multiplier = 1.2
        elif progress_ratio < 2.0:
            Cp_multiplier = 1.0
        else:
            Cp_multiplier = 0.8   # late stage: exploit

        adaptive_Cp = self.Cp * Cp_multiplier
        # Clamp to reasonable range [0.2, 5.0]
        adaptive_Cp = float(np.clip(adaptive_Cp, 0.2, 5.0))
        return adaptive_Cp

    def select(self, adaptive_Cp=None):
        """
        UCT-based leaf selection with optional adaptive Cp.

        When adaptive_Cp is provided, it overrides self.Cp for this round.
        The leaf exploration bonus also gently pushes toward leaves that:
          - Have not been visited recently (n is low relative to siblings)
          - Are good-kid branches that have been under-visited
        """
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        use_Cp = adaptive_Cp if adaptive_Cp is not None else self.Cp

        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(use_Cp))
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
        return curt_node, path

    def backpropogate(self, leaf, acc):
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
            curt_node.n += 1
            curt_node = curt_node.parent

    def search(self, iterations):
        """Execute MCTS search loop with adaptive exploration.

        Every iteration:
          1. dynamic_treeify()  — rebuild tree from all samples
          2. adaptive Cp         — recompute Cp based on tree imbalance
          3. select()            — UCT leaf selection (uses adaptive Cp)
          4. propose()           — generate candidates from leaf
          5. collect_samples()  — evaluate (sign already handled)
          6. backpropogate()    — update x_bar and n along path
        """
        for idx in range(iterations):
            self.dynamic_treeify()
            adaptive_Cp = self._compute_adaptive_Cp()
            leaf, path = self.select(adaptive_Cp=adaptive_Cp)

            if self.solver_type == 'bo':
                samples = leaf.propose_samples_bo(1, path, self.lb, self.ub, self.samples)
                for s in samples:
                    # collect_samples expects maximization format; for minimization
                    # func returns f, so we negate: value = func(s) * -1
                    value = self.collect_samples(s)
                    self.backpropogate(leaf, value)

            elif self.solver_type == 'turbo':
                # propose_samples_turbo returns (proposed_X, fX) where
                # fX = -func(x)  (already negated for maximization format)
                proposed_X, fX = leaf.propose_samples_turbo(10, path, self.func)
                for j, s in enumerate(proposed_X):
                    # fX[j] is already negated: no additional negation needed
                    value = self.collect_samples(s, value=fX[j])
                    self.backpropogate(leaf, value)
            else:
                raise Exception("solver not implemented")


