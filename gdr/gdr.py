"""
# GDR
####  22.03.2022 by Radeev Nikita
"""
import deap
from time import time
from typing import Dict
import geppy as gep
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from deap import creator
from deap import base
from deap import tools
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from time_logger import TimeLogger


def binary_resample(data, target, sample_size, balanced=True, random_state=72):
    value_counts = data[target].value_counts()
    target_distrib = value_counts.tolist()
    target_values = data[target].unique()
    if len(data) < sample_size:
        return data
    smallest_frac = min(target_distrib)
    if smallest_frac < sample_size // 2:
        if balanced:
            first_class_sample = data[data[target] == target_values[0]].sample(
                                    n=smallest_frac, random_state=random_state
                                    )
            second_cls_smpl = data[data[target] == target_values[1]].sample(
                                    n=smallest_frac, random_state=random_state
                                    )
            result_df = pd.concat(
                            [first_class_sample,
                             second_cls_smpl],
                            ignore_index=True)
        else:
            clses = [k for k in value_counts.to_dict().keys()]
            smaller_cls = clses[0] if value_counts.to_dict().get(
                                clses[0]) <= smallest_frac else clses[1]
            bigger_cls = clses[0] if smaller_cls != clses[0] else clses[1]
            bigger_cls_frac = min(
                                sample_size - smallest_frac,
                                value_counts.to_dict().get(bigger_cls))
            smaller_cls_sample = data[data[target] == smaller_cls].sample(
                                    n=smallest_frac,
                                    random_state=random_state)
            bigger_class_sample = data[data[target] == bigger_cls].sample(
                                    n=bigger_cls_frac,
                                    random_state=random_state)
            result_df = pd.concat(
                            [smaller_cls_sample,
                             bigger_class_sample],
                            ignore_index=True)
    else:
        first_class_sample = data[data[target] == target_values[0]].sample(
                                n=sample_size // 2, random_state=random_state
                                )
        second_cls_smpl = data[data[target] == target_values[1]].sample(
                                n=sample_size // 2, random_state=random_state
                                )
        result_df = pd.concat(
                        [first_class_sample,
                         second_cls_smpl],
                        ignore_index=True)
    return result_df


def find_max_fit(fitness_list):
    max_fit = 0.0
    max_fit_id = None
    for id, fit_vals in enumerate(fitness_list):
        if np.mean(fit_vals) > max_fit:
            max_fit = np.mean(fit_vals)
            max_fit_id = id
    return fitness_list[max_fit_id]


def terminal_feature_select(data,
                            target,
                            number_of_features,
                            mode="important"):
    """
    :param mode: important, random, all
    """
    X = data
    y = target
    if len(X.columns) <= number_of_features:
        return X.columns
    elif mode == "random":
        cols = list(X.columns)
        X = X[random.sample(cols, number_of_features)]
    elif mode == "all":
        pass
    elif mode == "important":
        lgmb = LGBMClassifier().fit(X, y, eval_names=X.columns)
        feat_imprtncs = [feat_tup
                         for feat_tup
                         in zip(X.columns, lgmb.feature_importances_)]
        feat_imprtncs.sort(reverse=True, key=(lambda x: x[1]))
        terminal_features = [tup[0]
                             for tup
                             in feat_imprtncs[:number_of_features]]
        X = X[terminal_features]
    else:
        print(f"\nIncorrect mode value: {mode}\n")
    terminal_features = X.columns
    return terminal_features


def estimate_quality(X, y):
    score = np.mean(
                cross_val_score(
                    LogisticRegression(),
                    X,
                    y,
                    cv=3,
                    scoring=make_scorer(roc_auc_score)))
    return score


def normalize(x):
    x_temp = x.reshape(-1, 1)
    result = MinMaxScaler().fit_transform(x_temp) + 0.0001
    return result.reshape(1, -1)


def oper_sum(x1, x2):
    return x1 + x2


def oper_sub(x1, x2):
    return x1 - x2


def oper_mul(x1, x2):
    x1_norm = normalize(x1)
    x2_norm = normalize(x2)
    return x1_norm * x2_norm


def oper_div(x1, x2):
    x1_norm = normalize(x1)
    x2_norm = normalize(x2)
    return x1_norm / x2_norm


def oper_ln(x):
    return np.log(np.absolute(x) + 0.00001)


def oper_exp(x):
    return normalize(np.exp(x.clip(max=43)))


def oper_cos(x):
    return np.cos(x)


def oper_abs(x):
    return np.absolute(x)


def oper_max(x1, x2):
    return np.array([elem1 if elem1 > elem2 else elem2
                     for elem1, elem2
                     in zip(x1.flatten(), x2.flatten())])


def oper_mean(x1, x2):
    return np.array([(elem1 + elem2) / 2.0 for elem1, elem2 in zip(x1, x2)])


def scale_distance(distance):
    return 2.0 / (1.0 + np.exp(-10.0 * distance)) - 1.0


class GDR(BaseEstimator, TransformerMixin):
    def __init__(self,
                 population_size=1600,
                 generations_number=21,
                 generate_features_num=3,
                 head_length=10,
                 feature_select_number=8,
                 terminal_selection='all',
                 sample_size=8192,
                 mutation_prob=0.25,
                 crossover_prob=0.25,
                 tournsize=4,
                 random_state=72):
        # for reproduction
        self.population_size = population_size
        self.generations_number = generations_number
        self.gen_feats_num = generate_features_num
        self.head_length = head_length
        self.feature_select_number = feature_select_number
        self.terminal_selection_mode = terminal_selection
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournsize = tournsize
        self.random_state = random_state
        self.sample_size = sample_size
        self.n_genes = 1   # can't be non 1 in this configuration
        self._trans_train_df = pd.DataFrame()
        self._trans_val_df = pd.DataFrame()
        self.compiled_hof = []
        self._epoch_counter = 0
        self.hof = []
        self.log_dict = dict()
        self._terminal_features = []
        self._is_id = False
        self._time_logger = TimeLogger()
        self._time_loggers = []
        self._generation_metrics = [[] for _ in range(self.gen_feats_num)]
        self._epochs_metrics = []
        self._top_gdr_features = None
        if self.gen_feats_num <= 2:
            self._fit_list = [
                ('lr', LogisticRegression(random_state=self.random_state)),
                ('distance', None),
            ]
        else:
            self._fit_list = [
                ('lr', LogisticRegression(random_state=self.random_state)),
                ('dt', DecisionTreeClassifier(random_state=self.random_state)),
                ('distance', None),
                # ('svc', LinearSVC(max_iter=100)),
                # ('lgbm', LGBMClassifier(random_state=self.random_state)),
                # ('rf', RandomForestClassifier(n_estimators=20,
                #                               random_state=self.random_state)),
            ]
        self._mixin_model = LinearSVC(max_iter=100)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        :param X: 2d-array of numerical features
        :param y: 1d-array of target values
        :return: fitted GDR
        """
        df = X
        df['target'] = y
        df_sample = binary_resample(
                    data=df,
                    target='target',
                    sample_size=self.sample_size,
                    balanced=True,
                    random_state=self.random_state)
        X = df_sample.drop(['target'], axis=1)
        y = df_sample['target']

        print(f'sample size now is: {len(X)}')

        self._terminal_features = terminal_feature_select(
                        data=X,
                        target=y,
                        number_of_features=self.feature_select_number,
                        mode=self.terminal_selection_mode)
        X_train_as_list = [X[f"{col}"].to_numpy().astype(float)
                           for col in self._terminal_features]
        X_val_as_list = None
        if X_val is not None:
            X_val_as_list = [X_val[f"{col}"].to_numpy().astype(float)
                             for col in self._terminal_features]
        if len(X_train_as_list) == 0:
            self._is_id = True
            return self

        """Terminals"""
        pset = gep.PrimitiveSet('Main', input_names=self._terminal_features)
        """Operations"""
        pset.add_function(oper_sum, 2)
        pset.add_function(oper_sub, 2)
        pset.add_function(oper_mul, 2)
        pset.add_function(oper_div, 2)
        pset.add_function(oper_max, 2) 
        pset.add_function(oper_mean, 2)  
        pset.add_function(oper_ln, 1)
        pset.add_function(oper_exp, 1)
        pset.add_function(oper_cos, 1)
        pset.add_function(oper_abs, 1)

        """Genetic algorithm entities"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", gep.Chromosome,
                       fitness=creator.FitnessMax,
                       a=float, b=float)

        toolbox = gep.Toolbox()
        toolbox.register('gene_gen', gep.Gene,
                         pset=pset,
                         head_length=self.head_length)
        toolbox.register('individual', creator.Individual,
                         gene_gen=toolbox.gene_gen,
                         n_genes=self.n_genes)
        toolbox.register("population", tools.initRepeat,
                         list,
                         toolbox.individual)

        # compile utility: which translates an individual
        # into an executable function (Lambda)
        toolbox.register('compile', gep.compile_, pset=pset)

        fit_cycle_len = len(self._fit_list)

        epochs_number = self.gen_feats_num

        def evaluate(individual):
            """Evalute the fitness of an individual"""
            time_eval_start = time()
            indiv_val = self._trans_train_df
            func = toolbox.compile(individual)
            new_feature = pd.Series(func(*X_train_as_list).flatten())
            indiv_index = f'gdr_feature_{len(self.compiled_hof)}'
            indiv_val[indiv_index] = new_feature
            # "kill" individuals with low deviation
            if indiv_val[indiv_index].std() < 0.000001:
                return 0.0, 0.0
            norm_indiv_val = StandardScaler().fit_transform(indiv_val)

            mixin_score = np.mean(cross_val_score(
                self._mixin_model,
                norm_indiv_val,
                y,
                cv=3,
                scoring=make_scorer(roc_auc_score)))
            fit_list_index = self._epoch_counter % fit_cycle_len
            if self._fit_list[fit_list_index][0] != 'distance':
                time_crit_start = time()
                fitness_score = np.mean(cross_val_score(
                    self._fit_list[fit_list_index][1],
                    norm_indiv_val,
                    y, cv=3,
                    scoring=make_scorer(roc_auc_score)))
                time_crit_finish = time()
                time_crit = time_crit_finish - time_crit_start
                self._time_logger.add_criterion_time(
                    self._fit_list[fit_list_index][0],
                    time_crit)
            else:
                time_dist_crit_start = time()
                first_set = norm_indiv_val[y == 0]
                second_set = norm_indiv_val[y != 0]
                distance_score = np.mean(pairwise_distances_argmin_min(
                    first_set,
                    second_set)[1])
                fitness_score = scale_distance(distance_score)
                time_dist_crit_finish = time()
                time_dist_crit = time_dist_crit_finish - time_dist_crit_start
                self._time_logger.add_criterion_time(
                    'distance',
                    time_dist_crit)

            time_eval_finish = time()
            time_eval = time_eval_finish - time_eval_start
            self._time_logger.add_fitness_time(time_eval)
            return fitness_score, mixin_score

        toolbox.register('evaluate', evaluate)
        toolbox.register('select', tools.selTournament, tournsize=4)
        toolbox.register('mut_uniform', gep.mutate_uniform,
                         pset=pset, pb=2 / (2 * self.head_length + 1))
        # Alternatively, assign the probability along
        # with registration using the pb keyword argument
        toolbox.register('mut_invert', gep.invert,
                         pb=self.mutation_prob)
        toolbox.register('mut_is_ts', gep.is_transpose,
                         pb=self.mutation_prob)
        toolbox.register('mut_ris_ts', gep.ris_transpose,
                         pb=self.mutation_prob)

        # general crossover whose aliases start with 'cx'
        toolbox.register('cx_1p', gep.crossover_one_point,
                         pb=self.crossover_prob)
        toolbox.register('cx_2p', gep.crossover_two_point,
                         pb=self.crossover_prob)

        stats = tools.Statistics(key=lambda ind: ind)
        stats.register("avg", (lambda ind_list: np.mean(
            [np.mean(ind.fitness.values) for ind in ind_list])))
        stats.register("std", (lambda ind_list: np.std(
            [np.mean(ind.fitness.values) for ind in ind_list])))
        stats.register("min", (lambda ind_list: np.min(
            [np.mean(ind.fitness.values) for ind in ind_list])))
        stats.register("max", (lambda ind_list: np.max(
            [np.mean(ind.fitness.values) for ind in ind_list])))
        stats.register("max_base", (lambda ind_list: np.max(
            [ind.fitness.values[1] for ind in ind_list])))
        stats.register("max_mixin", (lambda ind_list: np.max(
            [ind.fitness.values[0] for ind in ind_list])))
        stats.register("cur_time", (lambda ind_list: time()))
        # stats.register("best", (lambda ind_list: ind_list))
        # start evolution
        for i in range(epochs_number):
            pop = toolbox.population(n=self.population_size)
            self._epoch_counter = i
            hof = tools.HallOfFame(1)  # record the best individuals
            # Exploration phase
            pop, log = gep.gep_simple(
                        pop,
                        toolbox,
                        n_generations=1,
                        n_elites=10,
                        stats=stats,
                        hall_of_fame=hof,
                        verbose=True)
            ind = hof[0]
            func = toolbox.compile(ind)
            self._trans_train_df[f'gdr_feature_{i}'] = pd.Series(
                            func(*X_train_as_list).flatten())
            if X_val_as_list is not None and self.save_loss_curves:
                self._trans_val_df[f'gdr_feature_{i}'] = pd.Series(
                                func(*X_val_as_list).flatten())
                train_metric = estimate_quality(self._trans_train_df, y)
                val_metric = estimate_quality(self._trans_val_df, y_val)
                self._generation_metrics[i].append(
                    (len(self._generation_metrics[i]),
                     train_metric,
                     val_metric))

            pop = deap.tools.selBest(pop, k=self.population_size // 10)
            time_log = log.select('cur_time')
            # Exploitation phase
            for j in range(self.generations_number - 1):
                pop, log = gep.gep_simple(
                            pop,
                            toolbox,
                            n_generations=1,
                            n_elites=1,
                            stats=stats,
                            hall_of_fame=hof,
                            verbose=True)
                ind = hof[0]
                func = toolbox.compile(ind)
                self._trans_train_df[f'gdr_feature_{i}'] = pd.Series(
                                func(*X_train_as_list).flatten())
                if X_val_as_list is not None and self.save_loss_curves:
                    self._trans_val_df[f'gdr_feature_{i}'] = pd.Series(
                                    func(*X_val_as_list).flatten())
                    train_metric = estimate_quality(self._trans_train_df, y)
                    val_metric = estimate_quality(self._trans_val_df, y_val)
                    self._generation_metrics[i].append(
                        (len(self._generation_metrics[i]),
                         train_metric,
                         val_metric))

            time_log.append(log.select('cur_time')[0])
            for j in range(len(time_log) - 1):
                self._time_logger.add_epoch_time(time_log[j + 1] - time_log[j])

            self.log_dict.update({i: log})
            ind = hof[0]
            self.hof.append(ind)
            func = toolbox.compile(ind)
            self.compiled_hof.append(func)
            self._trans_train_df[f'gdr_feature_{i}'] = pd.Series(
                            func(*X_train_as_list).flatten())
            if X_val_as_list is not None and self.save_loss_curves:
                self._trans_val_df[f'gdr_feature_{i}'] = pd.Series(
                                func(*X_val_as_list).flatten())
                train_metric = estimate_quality(self._trans_train_df, y)
                val_metric = estimate_quality(self._trans_val_df, y_val)
                self._epochs_metrics.append(
                    (len(self._epochs_metrics),
                     train_metric,
                     val_metric))
            print(f'Best individual {i}: {ind.fitness}')
            print(ind)
            self._time_logger.agg_criterion_time()
            self._time_logger.agg_epoch_time()
            self._time_logger.agg_fitness_time()
            self._time_loggers.append(self._time_logger)
            self._time_logger = TimeLogger()

        self._top_gdr_features = self.transform(X).columns
        return self

    def transform(self, X):
        """
        :param X: 2d-array of numerical features
        :return: X transformed by GDR
        """
        if self._is_id:
            return X
        X = X[self._terminal_features]
        result_df = pd.DataFrame()
        X_as_list = [X[f"{col}"].to_numpy().astype(float) for col in X.columns]
        for i, func in enumerate(self.compiled_hof):
            result_df[f'gdr_feature_{i}'] = pd.Series(
                func(*X_as_list).flatten())
        for col in result_df.columns:
            result_df[col] = StandardScaler().fit_transform(
                result_df[col].to_numpy().reshape(-1, 1))
        if self._top_gdr_features is not None:
            result_df = result_df[self._top_gdr_features]
        return result_df

    def get_hof(self):
        return self.hof

    def get_logs(self) -> Dict:
        return self.log_dict

    def save_hof_visualization(self, path, dataset_name, fold):
        for i, ind in enumerate(self.hof):
            rename_labels = {
                'oper_sum': '+',
                'oper_sub': '-',
                'oper_mul': '*',
                'oper_div': '/',
                'oper_max': 'max',
                'oper_abs': 'abs',
                'oper_cos': 'cos(x)',
                'oper_ln': 'ln',
                'oper_exp': 'exp',
                'oper_mean': 'avg'}
            pic_path = f'{path}numerical_expression_tree_'
            pic_path += f'{dataset_name}_f{fold}_{i}.png'
            gep.export_expression_tree(ind, rename_labels, pic_path)
            plt.close('all')

    def save_logs_visualization(self, path, dataset_name, fold):
        import matplotlib.pyplot as plt
        for i in range(len(self.log_dict.keys())):
            log = self.log_dict.get(i)
            gen = log.select('gen')
            fit_max = log.select('max')
            fit_std = log.select('std')
            # fit_min = log.select('min')
            fit_avg = log.select('avg')

            fig, ax1 = plt.subplots()
            # line_fit_min = ax1.plot(gen, fit_min, 'b-', label='Fitness Min')
            line_fit_max = ax1.plot(gen, fit_max, 'r', label='Fitness Max')
            line_fit_avg = ax1.plot(gen, fit_avg, 'g', label='Fitness Avg')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')

            # lns = line_fit_min + line_fit_max + line_fit_avg
            lns = line_fit_max + line_fit_avg
            labels = [line.get_label() for line in lns]
            ax1.legend(lns, labels, loc='lower right')

            save_path = f'{path}logs_fitness_'
            save_path += f'{dataset_name}_feat_f{fold}_{i}.png'
            plt.savefig(save_path)

            fig, ax2 = plt.subplots()
            line_std = ax2.plot(gen, fit_std, 'b-', label='Fitness Std')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Fitness Std')

            lns = line_std
            labels = [line.get_label() for line in lns]
            ax2.legend(lns, labels, loc='upper left')

            save_path = f'{path}logs_fitness_std_'
            save_path += f'{dataset_name}_f{fold}_feat_{i}.png'
            plt.savefig(save_path)
            plt.close('all')

    def save_time_logs(self, path, dataset_name, fold):
        save_path = f'{path}time_logs_{dataset_name}_f{fold}.csv'
        result_list = []
        for i, time_logger in enumerate(self._time_loggers):
            data_to_save = {'dataset': dataset_name, 'feature': f'{i}'}
            data_to_save.update(time_logger.get_fitness_time())
            data_to_save.update(time_logger.get_epoch_time())
            criterion_times = time_logger.get_criterion_time()
            for crit in criterion_times.keys():
                data_to_save.update(criterion_times.get(crit))
            result_list.append(data_to_save)
        pd.DataFrame(result_list).to_csv(save_path)

    def save_loss_curves(self, path, dataset_name, fold):
        save_path = f'{path}feature_curve_{dataset_name}_f{fold}.csv'
        epochs_df = pd.DataFrame(
            self._epochs_metrics,
            columns=['feature_generation_order', 'train_aucroc', 'val_aucroc'])
        epochs_df.to_csv(save_path, index=False)
        for i in range(len(self._generation_metrics)):
            save_path = f'{path}gen_curves_{dataset_name}_f{fold}_feat{i}.csv'
            gen_df = pd.DataFrame(
                self._generation_metrics[i],
                columns=['generation_order', 'train_aucroc', 'val_aucroc'])
            gen_df.to_csv(save_path, index=False)
