from typing import Iterable, Optional, NamedTuple
from collections import defaultdict
from itertools import cycle
import pathlib
import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Results(NamedTuple):
    results_file_path: pathlib.Path
    args_file_path: pathlib.Path
    taw_acc: pd.DataFrame
    tag_acc: pd.DataFrame
    taw_forg: pd.DataFrame
    tag_forg: pd.DataFrame
    args: dict


class ResultsFiles(NamedTuple):
    results_file_path: pathlib.Path
    args_file_path: pathlib.Path


class GridSearchResult(NamedTuple):
    alpha: float
    lr: float
    acc: float
    acc_std: float
    forg: float
    intr: float

class FinetuningGridSearchResult(NamedTuple):
    lr: float
    acc: float
    acc_std: float
    forg: float
    intr: float

class CovMatrixResults(NamedTuple):
    acc: float
    acc_std: float
    forg: float
    intr: float


class ResultsReader:
    """
    Simple reader of experiments results 
    stored as text files in typical FACIL way
    """
    results_path: os.PathLike
    args_path: os.PathLike
    num_class: int
    num_tasks: int
    num_class_per_task: Optional[Iterable[int]]

    def __init__(self, num_class: int, num_tasks: int, num_class_per_task: Optional[Iterable[int]] = None):
        self.num_class = int(num_class)
        self.num_tasks = int(num_tasks)
        if num_class_per_task is not None:
            self.num_class_per_task = list(num_class_per_task)
        else:
            self.num_class_per_task = self.estimate_num_class_per_task(self.num_class, self.num_tasks)

    def read(self, results_path: pathlib.Path, args_path: pathlib.Path) -> Results:
        taw_acc, tag_acc, taw_forg, tag_forg = self._read_results(results_path)
        args = self._read_args(args_path)
        return Results(
            results_file_path=results_path,
            args_file_path=args_path,
            taw_acc=taw_acc,
            tag_acc=tag_acc,
            taw_forg=taw_forg,
            tag_forg=tag_forg,
            args=args
        )

    def _read_results(self, path: pathlib.Path) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        self.results_path = pathlib.Path(path)
        num_last_lines = self.num_tasks*4 + 4 + 2 + 2 + 3

        with open(self.results_path, 'r') as results_file:
            results = results_file.readlines()[-num_last_lines:]

            taw_acc = results[:self.num_tasks]
            tag_acc = results[self.num_tasks+3:self.num_tasks*2+3]
            taw_forg = results[self.num_tasks*2+6:self.num_tasks*3+6]
            tag_forg = results[self.num_tasks*3+8:self.num_tasks*4+8]

            taw_acc, tag_acc, taw_forg, tag_forg = list(map(
                self._process_to_dataframe,
                (taw_acc, tag_acc, taw_forg, tag_forg)
            ))

            return (taw_acc, tag_acc, taw_forg, tag_forg)

    def _read_args(self, path: pathlib.Path) -> dict:
        self.args_path = pathlib.Path(path)

        with open(self.args_path, 'r') as file:
            return json.loads(file.read())

    @staticmethod
    def estimate_num_class_per_task(num_class: int, num_tasks: int) -> Iterable[int]:
        class_per_task = num_class // num_tasks
        class_first_task = num_class - class_per_task*(num_tasks-1)
        return [class_first_task] + ([class_per_task] * (num_tasks-1))

    @classmethod
    def _process_to_dataframe(cls, results: Iterable[str]) -> pd.DataFrame:
        results = list(map(cls._extract_values, results))
        return pd.DataFrame({
            f'task{i+1}': values for i, values in enumerate(results)
        })

    @classmethod
    def _extract_values(cls, line: str):
        line = cls._remove_special_tags(line)
        line = line.replace(' ', '')
        line = line.replace('Avg.:', '')
        line = line.split('%')
        line = line[:-2]
        line = list(map(float, line))
        return line

    @classmethod
    def _remove_special_tags(cls, text: str):
        text = text.replace('\t', '')
        text = text.replace('\n', '')
        return text


class ResultsFilesFinder:
    """
    Simple finder of results files 
    contains used dataset name
    """
    dataset: str
    root: pathlib.Path

    def __init__(self, dataset: str, root: pathlib.Path):
        self.dataset = str(dataset)
        self.root = pathlib.Path(root)

    def find(self) -> list[ResultsFiles]:
        dataset_subdirs = list(map(
            lambda d: os.path.join(self.root, d),
            filter(
                lambda d: os.path.isdir(os.path.join(self.root, d)) and str(d).lower().startswith(self.dataset),
                os.listdir(self.root)
            )
        ))

        files = []
        for subdir in dataset_subdirs:
            subdir_elements = os.listdir(subdir)

            results_file = list(filter(
                lambda d: os.path.isfile(os.path.join(subdir, d)) and str(d).lower().startswith('stdout'),
                subdir_elements
            ))

            args_file = list(filter(
                lambda d: os.path.isfile(os.path.join(subdir, d)) and str(d).lower().startswith('args'),
                subdir_elements
            ))

            files.append(
                ResultsFiles(
                    results_file_path=os.path.join(subdir, results_file[0]),
                    args_file_path=os.path.join(subdir, args_file[0])
                )
            )

        return files


class Metric:
    @staticmethod
    def intransigence(result: pd.DataFrame) -> float:
        last_task_results = result[result.columns[-1]].to_numpy()[:-1]
        after_task_results = np.diag(result)[:-1]
        return (after_task_results - last_task_results).mean()

    @staticmethod
    def avg_inc_accuracy(results: pd.DataFrame) -> float:
        num_tasks = len(results.columns)
        avgs = np.array([
            results[f'task{task+1}'].to_numpy()[:task+1].mean() 
            for task in range(num_tasks)
        ])
        return avgs.mean()

    @staticmethod
    def avg_inc_forg(results: pd.DataFrame) -> float:
        num_tasks = len(results.columns)
        forgs = np.array([
            results[f'task{task+1}'].to_numpy()[:task+1].mean()
            for task in range(num_tasks)
        ])
        return forgs.mean()


def plot_grid_search_results(experiments: dict[str, ResultsReader], result_paths: list[os.PathLike], datasets_sets: tuple[tuple[str]]):   
    lr_order = (0.1, 0.05, 0.01, 0.005, 0.001)
    ds_names_mapper = {
        'cifar100': 'CIFAR-100',
        'skin7': 'SKIN7',
        'skin8': 'SKIN8',
        'pathmnist': 'PathMNIST',
        'organamnist': 'OrganAMNIST'
    }
    
    for ds_sets in datasets_sets:
        fig, axs = plt.subplots(nrows=1, ncols=len(ds_sets), figsize=(8*len(ds_sets), 6))
        for ds_idx, dataset in enumerate(ds_sets):
            axis = axs if len(ds_sets) == 1 else axs[ds_idx]
            results, alphas = defaultdict(list), set()
            for finder in (ResultsFilesFinder(dataset, path) for path in result_paths):
                for results_file, args_file in finder.find():
                    exp_results = experiments[dataset].read(results_path=results_file, args_path=args_file)
                    results[exp_results.args['lr']].append(GridSearchResult(
                        acc=Metric.avg_inc_accuracy(exp_results.tag_acc),
                        forg=Metric.avg_inc_forg(exp_results.tag_forg),
                        intr=Metric.intransigence(exp_results.tag_acc),
                        alpha=exp_results.args['alpha'],
                        lr=exp_results.args['lr'],
                        acc_std=0
                    ))
                    alphas.add(exp_results.args['alpha'])
            alphas = sorted(alphas)

            for lr in results.keys():
                combined_results = []
                for alpha in alphas:
                    lr_alpha_results = list(filter(lambda r: r.alpha == alpha, results[lr]))
                    accs = list(map(lambda r: r.acc, lr_alpha_results))
                    combined_results.append(
                        GridSearchResult(
                            alpha=alpha, lr=lr, forg=0, intr=0,
                            acc=np.mean(accs), acc_std=np.std(accs)
                    ))
                results[lr] = combined_results
            
            xs = list(range(len(alphas)))
            markers = ('o', 'x', 's', 'D', 'h')
            line_styles = cycle(('-', '--', '-.', ':'))
            acc_best, lr_best, alpha_best = 0, 0, 0
            for lr, marker, line_style in zip(lr_order, markers, line_styles):
                res = results[lr]
                accs = list(map(lambda r: r.acc, res))
                accs_stds = list(map(lambda r: r.acc_std, res))
                axis.errorbar(xs, accs, yerr=accs_stds, fmt=f'-.{marker}', capsize=5, label=f'lr={lr}', ls=line_style)
                
                acc_max = max(accs)
                if acc_max > acc_best:
                    acc_best, lr_best = acc_max, lr
                    alpha_best = list(filter(lambda r: r.acc == acc_best, res))[0].alpha
            
            axis.legend()
            axis.set_xticks(xs, alphas)
            axis.set_title(ds_names_mapper[finder.dataset])
            axis.set_ylabel('Średnia dokładność [%]')
            axis.set_xlabel('Alfa')
            
            print(20*'=', dataset.capitalize(), f'(lr={lr_best}, alpha={alpha_best}, acc={acc_best:.5f})', 20*'=')
            
            res_df = pd.DataFrame()
            for lr, res in results.items():
                res_df[f'{lr}'] = list(map(lambda r: r.acc, res))
                res_df[f'{lr}_std'] = list(map(lambda r: r.acc_std, res))
            res_df.index = alphas
            
            print(res_df, '\n')
            
        plt.tight_layout()
        plt.show()
        
def plot_ft_gridsearch_results(experiments: dict[str, ResultsReader], result_paths: list[os.PathLike], datasets_sets: tuple[tuple[str]]):   
    ds_names_mapper = {
        'cifar100': 'CIFAR-100',
        'skin7': 'SKIN7',
        'skin8': 'SKIN8',
        'pathmnist': 'PathMNIST',
        'organamnist': 'OrganAMNIST'
    }
    
    for ds_sets in datasets_sets:
        fig, axs = plt.subplots(nrows=1, ncols=len(ds_sets), figsize=(8*len(ds_sets), 6))
        for ds_idx, dataset in enumerate(ds_sets):
            axis = axs if len(ds_sets) == 1 else axs[ds_idx]
            results = defaultdict(list)
            for finder in (ResultsFilesFinder(dataset, path) for path in result_paths):
                for results_file, args_file in finder.find():
                    exp_results = experiments[dataset].read(results_path=results_file, args_path=args_file)
                    results[exp_results.args['lr']].append(FinetuningGridSearchResult(
                        acc=Metric.avg_inc_accuracy(exp_results.tag_acc),
                        forg=Metric.avg_inc_forg(exp_results.tag_forg),
                        intr=Metric.intransigence(exp_results.tag_acc),
                        lr=exp_results.args['lr'],
                        acc_std=0
                    ))
            
            for lr in results.keys():
                accs = list(map(lambda r: r.acc, results[lr]))
                results[lr] = [
                    FinetuningGridSearchResult(
                        lr=lr, forg=0, intr=0,
                        acc=np.mean(accs), acc_std=np.std(accs)
                )]
            
            lr_order = (0.1, 0.05, 0.01, 0.005, 0.001)
            width = 0.25
            for x, lr in enumerate(lr_order):
                lr_result = results[lr]
                lr_result = lr_result[0] if len(lr_result) > 0 else FinetuningGridSearchResult(lr=lr, acc=1, acc_std=1, forg=0, intr=0)
                axis.bar(x, lr_result.acc, width, yerr=lr_result.acc_std, capsize=5, label=str(lr))
            
            axis.set_xticks(list(range(len(lr_order))), lr_order)
            axis.set_title(ds_names_mapper[finder.dataset])
            axis.set_ylabel('Średnia dokładność [%]')
            axis.set_xlabel('Współczynnik uczenia (LR)')
            
            print(20*'=', dataset.capitalize(), 20*'=')
            
            res_df = pd.DataFrame()
            for lr, res in results.items():
                res_df[f'{lr}'] = list(map(lambda r: r.acc, res))
                res_df[f'{lr}_std'] = list(map(lambda r: r.acc_std, res))
            print(res_df, '\n')
            
        plt.tight_layout()
        plt.show()


def plot_cov_matrix_results(experiments: dict[str, ResultsReader], result_paths: list[os.PathLike]):   
    ds_names_mapper = {
        'cifar100': 'CIFAR-100',
        'skin7': 'SKIN7',
        'skin8': 'SKIN8',
        'pathmnist': 'PathMNIST',
        'organamnist': 'OrganAMNIST'
    }
    
    alg_name_mapper = {
        'full': 'Pełna macierz',
        'diag': 'Przekątna macierzy',
        'nmc': 'Klasyfikator najbliższej średniej'
    }
    
    def full_matrix_filter(path: str):
        return 'full_cov_matrix' in path
    
    def matrix_diag_filter(path: str):
        return 'diagonal_cov_matrix' in path
    
    def nmc_filter(path: str):
        return 'nearest_mean_classifier' in path

    results = {ds_name: defaultdict(list) for ds_name in ds_names_mapper.keys()}
    
    for dataset, _ in experiments.items():
        for finder in (ResultsFilesFinder(dataset, path) for path in result_paths):
            for results_file, args_file in finder.find():
                exp_result = experiments[dataset].read(results_path=results_file, args_path=args_file)
                
                result_path = exp_result.results_file_path
                alg_type = 'full' if full_matrix_filter(result_path) else 'diag' if matrix_diag_filter(result_path) else 'nmc' if nmc_filter(result_path) else None
                
                results[dataset][alg_type].append(CovMatrixResults(
                    acc=Metric.avg_inc_accuracy(exp_result.tag_acc),
                    forg=Metric.avg_inc_forg(exp_result.tag_forg),
                    intr=Metric.intransigence(exp_result.tag_acc),
                    acc_std=0
                ))
    
    combined_results = {dataset: dict() for dataset in ds_names_mapper.keys()}
    for dataset, ds_results in results.items():
        for alg_type, alg_type_results in ds_results.items():
            accs = list(map(lambda r: r.acc, alg_type_results))
            combined_results[dataset][alg_type] = CovMatrixResults(
                acc=np.mean(accs), acc_std=np.std(accs), forg=0, intr=0
            )
    
    x = np.arange(len(combined_results.keys()))  # the label locations
    width = 0.25  # the width of the bars
    measures_keys = ('full', 'diag', 'nmc')
    hatches = ('', 'XX', '..')
    ds_order = ('cifar100', 'skin7', 'skin8', 'pathmnist', 'organamnist')
    round_digit = 2

    fig, ax = plt.subplots(layout='constrained')

    for (i, alg), hatch in zip(enumerate(measures_keys), hatches):
        accs = [
            round(combined_results[dataset][alg].acc, round_digit)
            for dataset in ds_order
        ]
        acc_stds = [
            round(combined_results[dataset][alg].acc_std, round_digit)
            for dataset in ds_order
        ]
        offset = width * (i+1)
        rects = ax.bar(x + offset, accs, width, 
                       yerr=acc_stds, 
                       capsize=5, 
                       label=alg_name_mapper[alg],
                       hatch=hatch)
        ax.bar_label(rects, padding=3)
        
    for dataset in ds_order:
        print(10*'=', dataset, 10*'=')
        for alg in measures_keys:
            acc = round(combined_results[dataset][alg].acc, round_digit)
            acc_std = round(combined_results[dataset][alg].acc_std, round_digit)
            print(f'{alg}: acc={acc} +- {acc_std}')
            

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Średnia dokładność [%]')
    ax.set_xlabel('Zbiór danych')
    ax.set_title('Porównanie średnich dokładności wśród różnych zbiorów danych')
    ax.set_xticks(x + width*2, [ds_names_mapper[ds_name] for ds_name in ds_order])
    ax.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    base_path = os.path.join('..', '..', 'results')
    no_measures = 3

    experiments = {
        'skin7': ResultsReader(7, 4),
        'skin8': ResultsReader(8, 4),
        'pathmnist': ResultsReader(9, 4),
        'organamnist': ResultsReader(11, 4),
        'cifar100': ResultsReader(100, 10)
    }

    ds_sets = (
        ('cifar100',), 
        ('skin7', 'skin8'), 
        ('organamnist', 'pathmnist')
    )

    print(25*'*', 'ImageNet Pretrained', 25*'*', '\n')
    plot_grid_search_results(
        experiments=experiments,
        result_paths=[os.path.join(base_path, 'gridsearch', 'imagenet_pretrained', f'seed_{i}') for i in range(no_measures)], 
        datasets_sets=ds_sets
    )

    print(25*'*', 'First Task Pretrained', 25*'*', '\n')
    plot_grid_search_results(
        experiments=experiments, 
        result_paths=[os.path.join(base_path, 'gridsearch', 'first_task_pretraining', f'seed_{i}') for i in range(no_measures)],
        datasets_sets=ds_sets
    )

    print(25*'*', 'Covariance Matrix Algorithms', 25*'*', '\n')
    plot_cov_matrix_results(
        experiments=experiments,
        result_paths=[os.path.join(base_path, 'cov_matrix', f'seed_{i}') for i in range(no_measures)]
    )
    
    print(25*'*', 'ONLINE Gridsearch - ImageNet Pretrained - SEED', 25*'*', '\n')
    plot_grid_search_results(
        experiments=experiments,
        result_paths=[os.path.join(base_path, 'online', 'seed', f'seed_{i}') for i in range(no_measures)], 
        datasets_sets=ds_sets
    )
    
    print(25*'*', 'ONLINE Gridsearch - ImageNet Pretrained - FINETUNING', 25*'*', '\n')
    plot_ft_gridsearch_results(
        experiments=experiments,
        result_paths=[os.path.join(base_path, 'online', 'finetuning', f'seed_{i}') for i in range(no_measures)], 
        datasets_sets=ds_sets
    )