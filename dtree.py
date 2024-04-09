from deap import gp
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.datasets import load_iris
import pandas as pd
from scipy.stats import rankdata
import itertools
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import f1_score
import copy
import random
from inspect import isclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re

import functools
from collections import deque


class MoreOrEqualCondition:
    def __init__(self, var_index):
        self.var_index = var_index
        self.__name__ = "if Xi[:," + str(var_index) + "] >= "


class LessCondition:
    def __init__(self, var_index):
        self.var_index = var_index
        self.__name__ = "if Xi[:," + str(var_index) + "] < "


class SelfCGA:

    def __init__(
        self, function, iters, pop_size, len_, tour_size=3, K=2, threshold=0.1
    ):

        self.function = function
        self.iters = iters
        self.pop_size = pop_size
        self.len_ = len_
        self.tour_size = tour_size

        self.thefittest = {"individ": None, "fitness": None}

        self.arr_pop_size = np.arange(pop_size, dtype=int)
        self.row_cut = np.arange(1, len_ - 1, dtype=int)
        self.row = np.arange(len_, dtype=int)

        self.funk_tour = lambda x: np.random.choice(
            self.arr_pop_size, size=self.tour_size, replace=False
        )

        self.m_sets = {
            "average": 1 / (self.len_),
            "strong": min(1, 3 / self.len_),
            "weak": 1 / (3 * self.len_),
        }
        self.c_sets = {
            "one_point": self.one_point_crossing,
            "two_point": self.two_point_crossing,
            "uniform": self.uniform_crossing,
        }
        self.s_sets = {
            "proportional": self.proportional_selection,
            "rank": self.rank_selection,
            "tournament": self.tournament_selection,
        }

        self.operators_list = [
            self.m_sets.keys(),
            self.c_sets.keys(),
            self.s_sets.keys(),
        ]

        self.K = K
        self.threshold = threshold

        self.stats_fitness = np.array([])
        self.stats_proba_m = np.zeros(shape=(0, len(self.m_sets.keys())))
        self.stats_proba_c = np.zeros(shape=(0, len(self.c_sets.keys())))
        self.stats_proba_s = np.zeros(shape=(0, len(self.s_sets.keys())))

    def mutation(self, population, probability):
        population = population.copy()
        roll = np.random.random(size=population.shape) < probability
        population[roll] = 1 - population[roll]
        return population

    def one_point_crossing(self, individ_1, individ_2):
        cross_point = np.random.choice(self.row, size=1)[0]
        if np.random.random() > 0.5:
            offspring = individ_1.copy()
            offspring[:cross_point] = individ_2[:cross_point]
            return offspring
        else:
            offspring = individ_2.copy()
            offspring[:cross_point] = individ_1[:cross_point]
            return offspring

    def two_point_crossing(self, individ_1, individ_2):
        c_point_1, c_point_2 = np.sort(
            np.random.choice(self.row, size=2, replace=False)
        )
        if np.random.random() > 0.5:
            offspring = individ_1.copy()
            offspring[c_point_1:c_point_2] = individ_2[c_point_1:c_point_2]
        else:
            offspring = individ_2.copy()
            offspring[c_point_1:c_point_2] = individ_1[c_point_1:c_point_2]
        return offspring

    def uniform_crossing(self, individ_1, individ_2):
        roll = np.random.random(size=individ_1.shape[0]) > 0.5
        if np.random.random() > 0.5:
            offspring = individ_1.copy()
            offspring[roll] = individ_2[roll]
        else:
            offspring = individ_2.copy()
            offspring[roll] = individ_1[roll]
        return offspring

    def proportional_selection(self, population, fitness):
        max_ = fitness.max()
        min_ = fitness.min()
        if max_ == min_:
            fitness_n = np.ones(fitness.shape)
        else:
            fitness_n = (fitness - min_) / (max_ - min_)

        probability = fitness_n / fitness_n.sum()
        offspring = population[
            np.random.choice(self.arr_pop_size, size=1, p=probability)
        ][0].copy()
        return offspring

    def tournament_selection(self, population, fitness):
        tournament = np.random.choice(
            self.arr_pop_size, size=self.tour_size, replace=False
        )
        max_fit_id = np.argmax(fitness[tournament])
        return population[tournament[max_fit_id]]

    def rank_selection(self, population, fitness):
        ranks = rankdata(fitness)
        probability = ranks / np.sum(ranks)
        offspring = population[
            np.random.choice(self.arr_pop_size, size=1, p=probability)
        ][0]
        return offspring

    def __update_thefittest(self, population, fitness):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id]
        if temp_best_fitness > self.thefittest["fitness"]:
            self.thefittest["fitness"] = temp_best_fitness
            self.thefittest["individ"] = population[temp_best_id].copy()

    def __update_statistic(self, fitness, m_proba, c_proba, s_proba):
        self.stats_fitness = np.append(self.stats_fitness, np.max(fitness))
        self.stats_proba_m = np.vstack([self.stats_proba_m, c_proba])
        self.stats_proba_c = np.vstack([self.stats_proba_c, c_proba])
        self.stats_proba_s = np.vstack([self.stats_proba_s, s_proba])
        for proba, proba_list in zip(
            [m_proba, c_proba, s_proba],
            [self.stats_proba_m, self.stats_proba_c, self.stats_proba_s],
        ):
            proba_list = np.append(proba_list, proba)

    def __update_proba(self, proba, z, operators_fitness, fitness):
        new_proba = proba.copy()

        operators_fitness = np.vstack([operators_fitness, fitness]).T
        operators_fitness = operators_fitness[operators_fitness[:, 0].argsort()]
        cut_index = np.unique(operators_fitness[:, 0], return_index=True)[1]
        groups = np.split(operators_fitness[:, 1].astype(float), cut_index)[1:]

        mean_fit = np.array(list(map(np.mean, groups)))

        new_proba[mean_fit.argmax()] = (
            new_proba[mean_fit.argmax()] + self.K / self.iters
        )
        new_proba = new_proba - self.K / (z * self.iters)
        new_proba = new_proba.clip(self.threshold, 1)
        new_proba = new_proba / new_proba.sum()

        return new_proba

    def choice_operators(self, operators, proba):
        return np.random.choice(list(operators), self.pop_size, p=proba)

    def create_offspring(self, operators, popuation, fitness):
        mutation, crossover, selection = operators

        parent_1 = self.s_sets[selection](popuation, fitness)
        parent_2 = self.s_sets[selection](popuation, fitness)
        offspring_no_mutated = self.c_sets[crossover](parent_1, parent_2)
        offspring_mutated = self.mutation(offspring_no_mutated, self.m_sets[mutation])
        return offspring_mutated

    def fit(self):

        z_list = [len(self.m_sets), len(self.c_sets), len(self.s_sets)]

        m_proba = np.full(z_list[0], 1 / z_list[0])
        c_proba = np.full(z_list[1], 1 / z_list[1])
        s_proba = np.full(z_list[2], 1 / z_list[2])

        population = np.random.randint(
            low=2, size=(self.pop_size, self.len_), dtype=np.byte
        )
        fitness = self.function(population)

        self.thefittest["individ"] = population[np.argmax(fitness)].copy()
        self.thefittest["fitness"] = fitness[np.argmax(fitness)].copy()
        self.__update_statistic(fitness, m_proba, c_proba, s_proba)

        for i in range(1, self.iters):

            chosen_operators = list(
                map(
                    self.choice_operators,
                    self.operators_list,
                    [m_proba, c_proba, s_proba],
                )
            )
            chosen_operators = np.array(chosen_operators).T
            func = lambda x: self.create_offspring(x, population, fitness)
            population = np.array(list(map(func, chosen_operators)))
            fitness = self.function(population)

            func = lambda x, y, z: self.__update_proba(x, y, z, fitness)
            m_proba, c_proba, s_proba = list(
                map(func, [m_proba, c_proba, s_proba], z_list, chosen_operators.T)
            )
            fitness[-1] = self.thefittest["fitness"].copy()
            population[-1] = self.thefittest["individ"].copy()

            self.__update_statistic(fitness, m_proba, c_proba, s_proba)
            self.__update_thefittest(population, fitness)

        return self


class SamplingGrid:

    def __init__(self, borders, parts):
        self.borders = borders
        self.parts = parts
        self.h = np.abs(borders["right"] - borders["left"]) / (2.0**parts - 1)

    @staticmethod
    def __decoder(population_parts, left_i, h_i):
        ipp = population_parts.astype(int)
        int_convert = np.sum(ipp * (2 ** np.arange(ipp.shape[1], dtype=int)), axis=1)
        return left_i + h_i * int_convert

    def transform(self, population):
        splits = np.add.accumulate(self.parts)
        p_parts = np.split(population, splits[:-1], axis=1)
        fpp = [
            self.__decoder(p_parts_i, left_i, h_i)
            for p_parts_i, left_i, h_i in zip(p_parts, self.borders["left"], self.h)
        ]
        return np.vstack(fpp).T


class DecitionTree:
    def __init__(self, some_tree):
        self.tree = some_tree
        self.threshold = None
        self.borders_left = None
        self.borders_right = None

    def init_threshold(self, some_X):
        threshold = []
        borders_left = []
        borders_right = []
        for node in self.tree:
            type_ = node.name[:2]
            if type_ == "if":
                var = int(re.search(r"\d+", node.name[:])[0])
                left = some_X[:, var].min()
                right = some_X[:, var].max()
                threshold.append(np.random.uniform(left, right))
                borders_left.append(left)
                borders_right.append(right)

        self.threshold = np.array(threshold)
        self.borders_left = np.array(borders_left)
        self.borders_right = np.array(borders_right)

    def compile_rules2d(self, threshold2d):
        rules = {}
        rules_stack = []
        rules_current = ""
        rules_markers = {}
        t_i = 0
        for k, node in enumerate(self.tree):
            rules_markers[k] = rules_current
            if node.arity == 0:
                rules[k] = (rules_current[:-2], int(node.value))
                if len(rules_stack) > 0:
                    rules_current = rules_stack.pop()
            else:
                rules_current += (
                    " ("
                    + node.name[2:]
                    + "np.array("
                    + str(list(threshold2d[:, t_i]))
                    + ")[:,np.newaxis]) *"
                )
                t_i += 1
                rules_current_turn = rules_current[::-1]
                match_more = re.search(r"=>", rules_current_turn)
                match_less = re.search(r"<", rules_current_turn)
                if match_more and match_less:
                    more_from = match_more.span(0)[0]

                    less_from = match_less.span(0)[0]
                    if more_from < less_from:
                        rules_current_turn = (
                            rules_current_turn[:more_from]
                            + "<"
                            + rules_current_turn[more_from + 2 :]
                        )

                    else:
                        rules_current_turn = (
                            rules_current_turn[:less_from]
                            + "=>"
                            + rules_current_turn[less_from + 1 :]
                        )
                elif match_more:
                    more_from = match_more.span(0)[0]
                    rules_current_turn = (
                        rules_current_turn[:more_from]
                        + "<"
                        + rules_current_turn[more_from + 2 :]
                    )
                else:
                    less_from = match_less.span(0)[0]
                    rules_current_turn = (
                        rules_current_turn[:less_from]
                        + "=>"
                        + rules_current_turn[less_from + 1 :]
                    )
                rules_stack.append(rules_current_turn[::-1])

        return np.array(list(rules.values()), dtype=object)

    def predict2d(self, Xi, threshold2d):
        rules2d = self.compile_rules2d(threshold2d)
        func = lambda x, Xi: eval(x)
        conds = np.array(list(map(functools.partial(func, Xi=Xi), rules2d[:, 0])))
        conds = conds.transpose((1, 0, 2))
        classes = np.kron(rules2d[:, 1].reshape(-1, 1), np.ones((1, len(Xi)))).astype(
            int
        )
        func = lambda x: classes.T[x.T]

        return np.array(list(map(func, conds)))

    def float_int_grid(self, X_float, n_bins, borders, parts):
        bins = np.array(list(itertools.product([0, 1], repeat=n_bins)), dtype="byte")
        all_bins = []
        all_float = []
        for i in range(len(X_float)):
            borders_i = {
                "left": np.array([borders["left"][i]]),
                "right": np.array([borders["right"][i]]),
            }
            parts_i = np.array([parts[i]])

            sg = SamplingGrid(borders_i, parts_i)
            bins_float = sg.transform(bins)
            argmin = np.argmin(np.abs(X_float[i] - bins_float))
            all_bins.append(bins[argmin])
            all_float.append(bins_float[argmin])
        return np.array(all_bins), np.array(all_float)

    def get_fitness(self, population_i, x_true, y_true):
        predicts = self.predict2d(x_true, population_i)
        func = lambda x: f1_score(y_true, x, average="macro")
        f1 = np.array(list(map(func, predicts)))
        return f1

    def SelfCGA_fit(self, some_Xi, some_yi, iters, pop_size, tour_size, n_bit):
        old_borders = self.threshold.copy()
        vars_ = len(old_borders)
        borders = {"left": self.borders_left, "right": self.borders_right}

        parts = np.full(vars_, n_bit)

        old_borders_bin, old_borders_grid = self.float_int_grid(
            old_borders, n_bit, borders, parts
        )
        grid_model = SamplingGrid(borders, parts)

        function = lambda x: self.get_fitness(grid_model.transform(x), some_Xi, some_yi)
        model_opt = SelfCGA(
            function, iters, pop_size, np.sum(parts), tour_size=tour_size
        )

        model_opt.fit()

        thefittest = model_opt.thefittest
        float_borders = grid_model.transform(thefittest["individ"].reshape(1, -1))[0]

        self.threshold = float_borders
        return model_opt


class SelfCGPDTClassifier:

    def __init__(self, iters, pop_size, max_height=5, tour_size=5, K=2, threshold=0.1):
        self.iters = iters
        self.pop_size = pop_size
        self.max_height = max_height
        self.tour_size = tour_size

        self.K = K
        self.threshold = threshold

        self.pset = None

        self.thefittest = {"individ": None, "fitness": None, "net": None}
        self.pset = None

        self.um_low = lambda x: self.uniform_mutation(x, 0.25)
        self.um_mean = lambda x: self.uniform_mutation(x, 1)
        self.um_strong = lambda x: self.uniform_mutation(x, 4)

        self.pm_low = lambda x: self.one_point_mutation(x, 0.25)
        self.pm_mean = lambda x: self.one_point_mutation(x, 1)
        self.pm_strong = lambda x: self.one_point_mutation(x, 4)

        self.operators_list = ["mutation", "crossing", "selection"]

        self.m_sets = {
            "uniform_low": self.um_low,
            "uniform_mean": self.um_mean,
            "uniform_strong": self.um_strong,
            "point_low": self.pm_low,
            "point_mean": self.pm_mean,
            "point_strong": self.pm_strong,
        }

        self.s_sets = {
            "tournament": self.tournament_selection,
            "rank": self.rank_selection,
            "proportional": self.proportional_selection,
        }

        self.c_sets = {
            "standart": self.standart_crossing,
            "one_point": self.one_point_crossing,
            "empty": self.empty_crossing,
        }

        self.stats = {
            "fitness": pd.DataFrame(columns=["max", "median", "min", "std"]),
            "proba": {
                "mutation": pd.DataFrame(columns=self.m_sets.keys()),
                "crossing": pd.DataFrame(columns=self.c_sets.keys()),
                "selection": pd.DataFrame(columns=self.s_sets.keys()),
            },
        }

        self.arr_pop_size = np.arange(pop_size, dtype=int)

        self.runs = 0

        self.fittest_history = []

    def init_pset(self, num_vars, num_outs):
        self.pset = gp.PrimitiveSet("MAIN", num_outs)
        for i in range(num_vars):
            self.pset.addPrimitive(MoreOrEqualCondition(i), 2)
        for i in range(num_vars):
            self.pset.addPrimitive(LessCondition(i), 2)

        for i in range(num_outs):
            eval("self.pset.renameArguments(ARG" + str(i) + "='" + str(i) + "')")

    def generate_tree(self):
        return gp.PrimitiveTree(gp.genHalfAndHalf(self.pset, 2, 5))

    @staticmethod
    def mark_tree(tree):
        stack = []
        current = ""
        n_arg = "0"
        markers = np.array([])
        for k, node in enumerate(tree):
            current += n_arg
            markers = np.append(markers, current)
            if node.arity == 0:
                if len(stack) > 0:
                    n_arg = "1"
                    current = stack.pop()
            elif node.arity == 1:
                n_arg = "0"
            else:
                stack.append(current)
                n_arg = "0"
        return markers

    def expr_mut(self, pset, type_, len_):
        return gp.genGrow(pset, 0, len_, type_)

    @staticmethod
    def replace_node(node, pset):
        filter_ = lambda x: x != node

        if node.arity == 0:  # Terminal
            pool = list(filter(filter_, pset.terminals[node.ret]))
            term = random.choice(pool)
            if isclass(term):
                term = term()
            return term
        else:  # Primitive
            pool = list(filter(filter_, pset.primitives[node.ret]))
            prims = [p for p in pool if p.args == node.args]
            return random.choice(prims)

    def one_point_mutation(self, some_net, proba):
        some_net = copy.deepcopy(some_net)
        proba = proba / len(some_net)
        for i, node in enumerate(some_net):
            if np.random.random() < proba:

                some_net[i] = self.replace_node(node, self.pset)

        return some_net

    def uniform_mutation(self, some_net, proba):
        some_net = copy.deepcopy(some_net)
        proba = proba / len(some_net)
        for i, node in enumerate(some_net[1:]):
            i = i + 1
            if np.random.random() < proba:
                slice_ = some_net.searchSubtree(i)
                type_ = node.ret
                temp = gp.PrimitiveTree(some_net[slice_])
                some_net[slice_] = self.expr_mut(
                    pset=self.pset, type_=type_, len_=temp.height
                )
                break

        return some_net

    def rank_selection(self, population, fitness):
        ranks = rankdata(fitness)
        probability = ranks / np.sum(ranks)
        ind = np.random.choice(self.arr_pop_size, size=1, p=probability)
        offspring = population[ind][0]
        return copy.deepcopy(offspring), fitness[ind][0]

    def tournament_selection(self, population, fitness):
        tournament = np.random.choice(
            self.arr_pop_size, size=self.tour_size, replace=False
        )
        max_fit_id = np.argmax(fitness[tournament])
        return (
            copy.deepcopy(population[tournament[max_fit_id]]),
            fitness[tournament[max_fit_id]],
        )

    def proportional_selection(self, population, fitness):
        max_ = fitness.max()
        min_ = fitness.min()
        if max_ == min_:
            fitness_n = np.ones(fitness.shape)
        else:
            fitness_n = (fitness - min_) / (max_ - min_)

        probability = fitness_n / fitness_n.sum()
        ind = np.random.choice(self.arr_pop_size, size=1, p=probability)
        offspring = population[ind][0]

        return copy.deepcopy(offspring), fitness[ind][0]

    @staticmethod
    def standart_crossing(ind_1, ind_2):
        offs_1, offs_2 = gp.cxOnePoint(copy.deepcopy(ind_1), copy.deepcopy(ind_2))
        if np.random.random() > 0.5:
            return offs_1
        else:
            return offs_2

    def one_point_crossing(self, ind1, ind2):
        ind1 = copy.deepcopy(ind1)
        ind2 = copy.deepcopy(ind2)
        if len(ind1) < 2 or len(ind2) < 2:
            if np.random.random() > 0.5:
                return ind1
            else:
                return ind2
        mark_1 = self.mark_tree(ind1)
        mark_2 = self.mark_tree(ind2)
        common, c_1, c_2 = np.intersect1d(mark_1, mark_2, return_indices=True)

        index = random.choice(range(1, len(c_1)))
        index1 = c_1[index]
        index2 = c_2[index]

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

        if np.random.random() > 0.5:
            return ind1
        else:
            return ind2

    def empty_crossing(self, ind1, ind2):
        if np.random.random() > 0.5:
            return ind1
        else:
            return ind2

    def train_and_test(
        self, ind, dtree, X_train_i, y_train_i, X_test_i, y_test_i, n_iters, n_size
    ):

        if len(ind) > 5000:
            return 0, dtree
        dtree.init_threshold(X_train_i)
        height = ind.height
        if height > self.max_height:
            fine_h = height
        else:
            fine_h = 0.0

        res = dtree.SelfCGA_fit(X_train_i, y_train_i, n_iters, n_size, 5, 12)

        predict = dtree.predict2d(X_test_i, dtree.threshold.reshape(1, -1))[0]
        len_ = len(dtree.threshold)
        fitnes_value = (
            f1_score(y_test_i, predict, average="macro") - 0.01 * fine_h - 0.001 * len_
        )

        # print(n_iters, n_size, fitnes_value, res.thefittest["fitness"], height)

        return fitnes_value, dtree

    def __update_statistic(self, fitness, m_proba, c_proba, s_proba):
        self.stats["fitness"] = self.stats["fitness"].append(
            {
                "max": fitness.max(),
                "min": fitness.min(),
                "median": np.median(fitness),
                "std": fitness.std(),
            },
            ignore_index=True,
        )

        for proba, oper in zip([m_proba, c_proba, s_proba], self.operators_list):
            self.stats["proba"][oper] = self.stats["proba"][oper].append(
                proba.copy(), ignore_index=True
            )

    def __update_proba(self, some_proba, operator, some_history, z):
        mutate_avg = some_history.groupby(operator).mean()["fitness"]
        argmax_mutate = mutate_avg.idxmax()
        some_proba[argmax_mutate] = some_proba[argmax_mutate] + self.K / self.iters
        new_proba = some_proba - self.K / (z * self.iters)
        new_proba = new_proba.clip(self.threshold, 1)
        return new_proba / new_proba.sum()

    def __update_thefittest(self, population, population_net, fitness):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id]
        if temp_best_fitness > self.thefittest["fitness"]:
            self.fittest_history.append(
                [
                    str(copy.deepcopy(population[temp_best_id])),
                    copy.copy(temp_best_fitness),
                ]
            )
            self.thefittest["fitness"] = temp_best_fitness
            self.thefittest["individ"] = copy.deepcopy(population[temp_best_id])
            self.thefittest["net"] = copy.deepcopy(population_net[temp_best_id])

    @staticmethod
    def get_m_n(all_, a):
        return int(all_ * (1 + a)), int(all_ * (1 / (1 + a)))

    def fit(self, some_X, some_y, some_X_test, some_y_test):
        runs_must = 0
        resource_min = 1250
        resource_max = 1250
        resource_h = (resource_max - resource_min) / (self.iters - 1)
        resource = np.full(self.pop_size, resource_min)

        proba_history = pd.DataFrame(
            np.empty((self.pop_size, 4)),
            columns=["mutation", "crossing", "selection", "fitness"],
            dtype=object,
        )

        z_m = len(self.m_sets)
        z_c = len(self.c_sets)
        z_s = len(self.s_sets)

        m_proba = pd.Series(np.full(z_m, 1 / z_m), index=self.m_sets.keys())
        c_proba = pd.Series(np.full(z_c, 0.9 / (z_c - 1)), index=self.c_sets.keys())
        c_proba["empty"] = 0.1
        s_proba = pd.Series(np.full(z_s, 1 / z_s), index=self.s_sets.keys())

        X_train, X_test, y_train, y_test = train_test_split(
            some_X, some_y, stratify=some_y, test_size=0.5
        )

        population = np.array(
            [self.generate_tree() for _ in range(self.pop_size)], dtype=object
        )
        population_temp = copy.deepcopy(population)
        population_dtree = np.array([DecitionTree(ind) for ind in population])

        fitness = np.zeros(len(population))
        for i, ind, rbase in zip(
            range(self.pop_size), population[:], population_dtree[:]
        ):

            all_ = np.sqrt(resource[i])
            m, n = self.get_m_n(all_, 0)
            # print(i, m, n, ind)
            fitness[i], population_dtree[i] = self.train_and_test(
                ind, rbase, X_train, y_train, X_test, y_test, int(n), int(m)
            )
            self.runs += m * n
            runs_must += round(resource[i])

        self.thefittest["individ"] = copy.deepcopy(population[np.argmax(fitness)])
        self.thefittest["fitness"] = fitness[np.argmax(fitness)].copy()
        self.thefittest["net"] = copy.deepcopy(population_dtree[np.argmax(fitness)])
        self.fittest_history.append(
            [
                str(copy.deepcopy(population[np.argmax(fitness)])),
                fitness[np.argmax(fitness)].copy(),
            ]
        )
        self.__update_statistic(fitness, m_proba, c_proba, s_proba)
        for i in range(1, self.iters):

            predict = self.thefittest["net"].predict2d(
                some_X_test, self.thefittest["net"].threshold.reshape(1, -1)
            )[0]
            fitnes_value = f1_score(some_y_test, predict, average="macro")
            print(i, "------", self.thefittest["fitness"], fitnes_value)
            for type_, list_, proba in zip(
                self.operators_list,
                [m_proba.index, c_proba.index, s_proba.index],
                [m_proba, c_proba, s_proba],
            ):
                proba_history[type_] = np.random.choice(
                    list_, self.pop_size, p=proba.values
                )
            for (
                j,
                m_o,
                c_o,
                s_o,
            ) in zip(
                range(self.pop_size),
                proba_history["mutation"],
                proba_history["crossing"],
                proba_history["selection"],
            ):

                parent_1, fitness_1 = self.s_sets[s_o](population, fitness)
                parent_2, fitness_2 = self.s_sets[s_o](population, fitness)

                offspring = self.c_sets[c_o](parent_1, parent_2)
                offspring = self.m_sets[m_o](offspring)

                population_temp[j] = copy.deepcopy(offspring)

            resource = resource + resource_h
            # print(resource.mean(), "resource.mean()")
            population_temp[-1] = copy.deepcopy(self.thefittest["individ"])

            population = copy.deepcopy(population_temp)
            population_dtree = np.array([DecitionTree(ind) for ind in population])
            for i, ind, rbase in zip(
                range(self.pop_size), population[:], population_dtree[:]
            ):
                all_ = np.sqrt(resource[i])
                m, n = self.get_m_n(all_, 0)
                # print(i, m, n, ind)
                fitness[i], population_dtree[i] = self.train_and_test(
                    ind, rbase, X_train, y_train, X_test, y_test, int(n), int(m)
                )
                self.runs += m * n
                runs_must += round(resource[i])

            proba_history["fitness"] = fitness

            m_proba = self.__update_proba(m_proba, "mutation", proba_history, z_m)
            c_proba = self.__update_proba(c_proba, "crossing", proba_history, z_c)
            s_proba = self.__update_proba(s_proba, "selection", proba_history, z_s)

            self.__update_thefittest(population, population_dtree, fitness)
            self.__update_statistic(fitness, m_proba, c_proba, s_proba)

            fitness[-1] = self.thefittest["fitness"]
            population_dtree[-1] = self.thefittest["net"]

        temp_ind = copy.deepcopy(self.thefittest["individ"])
        temp_net = copy.deepcopy(self.thefittest["net"])
        temp_fit = self.thefittest["fitness"].copy()

        temp_r = runs_must - self.runs
        all_ = all_ = np.sqrt(temp_r)

        m, n = self.get_m_n(all_, 0)
        new_fit, new_net = self.train_and_test(
            temp_ind, temp_net, X_train, y_train, X_test, y_test, m, n
        )
        if new_fit > temp_fit:
            self.thefittest["net"] = new_net
            self.thefittest["fitness"] = new_fit
            print("дообучилась")
            print(temp_fit, "->", new_fit)
        else:
            print(temp_fit, "->", temp_fit)

        return self

    def predict(self, some_X):
        dtree = self.thefittest["net"]
        predict = dtree.predict2d(some_X, np.array([dtree.threshold]))[0]
        return predict


def from_string(string, pset):

    tokens = re.split("[\t\n\r\f\v()]|, ", string)
    expr = []
    ret_types = deque()
    for token in tokens:
        if token == "":
            continue
        if len(ret_types) != 0:
            type_ = ret_types.popleft()
        else:
            type_ = None

        if token in pset.mapping:
            primitive = pset.mapping[token]

            if type_ is not None and not issubclass(primitive.ret, type_):
                raise TypeError(
                    "Primitive {} return type {} does not "
                    "match the expected one: {}.".format(
                        primitive, primitive.ret, type_
                    )
                )

            expr.append(primitive)
            if isinstance(primitive, gp.Primitive):
                ret_types.extendleft(reversed(primitive.args))
        else:
            try:
                token = eval(token)
            except NameError:
                raise TypeError("Unable to evaluate terminal: {}.".format(token))

            if type_ is None:
                type_ = type(token)

            if not issubclass(type(token), type_):
                raise TypeError(
                    "Terminal {} type {} does not "
                    "match the expected one: {}.".format(token, type(token), type_)
                )

            expr.append(gp.Terminal(token, False, type_))
    return gp.PrimitiveTree(expr)


def print_dtree(some_dtree, ax=None):
    G = nx.Graph()

    nodes, edges, labels = gp.graph(some_dtree.tree)
    nodes = list(range(len(some_dtree.tree)))
    labels = dict()

    uniform_colors = np.random.uniform(0.3, 0.9, size=(3 * len(nodes), 3))
    uniform_colors = np.hstack([uniform_colors, np.ones([3 * len(nodes), 1])])

    k = 0
    test = {}
    colors = []
    for i, node in enumerate(some_dtree.tree):
        type_ = node.name[:2]
        var = int(re.search(r"\d+", node.name[:])[0])
        if type_ == "if":
            labels[i] = node.name + str(round(some_dtree.threshold[k], 2))
            k += 1
            if var not in test:
                if len(test) == 0:
                    temp_int = np.random.choice(3 * len(nodes))
                    test[var] = uniform_colors[temp_int]

                else:
                    distance = (
                        uniform_colors - np.array(list(test.values()))[:, np.newaxis]
                    )
                    distance = np.sum(distance**2, axis=2)
                    mean_distance = np.mean(distance, axis=0)
                    argmax = np.argmax(mean_distance)
                    test[var] = uniform_colors[argmax]
                pass
            colors.append(test[var])
        else:
            labels[i] = node.value
            colors.append([0.0, 0.746, 0.996, 1])

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    positions = nx.nx_pydot.graphviz_layout(G, prog="dot")
    positions = {int(k): v for k, v in positions.items()}

    nx.draw_networkx_nodes(
        G, positions, ax=ax, node_color=colors, edgecolors="black", linewidths=0.5
    )
    nx.draw_networkx_edges(G, positions, ax=ax)
    nx.draw_networkx_labels(G, positions, labels, ax=ax, font_size=8)


def save_dtree(
    some_dtree, some_tree, path, train_acc, test_acc, fitness_hist, tree_hist, all_stats
):
    all_stats_columns = list(all_stats.columns)

    data = pd.Series(
        {
            "tree": str(some_tree),
            "weights": str(list(some_dtree.threshold)),
            "train_acc": train_acc,
            "test_acc": test_acc,
            "fitness_hist": fitness_hist,
            "tree_hist": tree_hist,
            "flatten_stats": list(all_stats.values.flatten()),
            "size": len(all_stats),
            "stats_columns": list(all_stats_columns),
        }
    )
    data.to_csv(path)
    return data


data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33)

model = SelfCGPDTClassifier(2, 5, max_height=10, tour_size=3)

model.init_pset(X_train.shape[1], len(set(y_train)))

model.fit(X_train, y_train, X_test, y_test)


predict_test = model.predict(X_test)
predict_train = model.predict(X_train)

acc_test = f1_score(y_test, predict_test, average="macro")
cm_test = confusion_matrix(y_test, predict_test)
acc_train = f1_score(y_train, predict_train, average="macro")
cm_train = confusion_matrix(y_train, predict_train)

print(acc_train)
print(cm_train)
print(acc_test)
print(cm_test)

stats = model.stats
all_stats = pd.concat(
    [
        stats["proba"]["mutation"],
        stats["proba"]["crossing"],
        stats["proba"]["selection"],
        stats["fitness"],
    ],
    axis=1,
)

fittest_history = np.array(model.fittest_history, dtype=object)
fittest_history = pd.DataFrame(
    {"tree": fittest_history[:, 0], "fitness": fittest_history[:, 1]}
)

final_tree = model.thefittest["individ"]
final_dtree = model.thefittest["net"]

print_dtree(model.thefittest["net"])
