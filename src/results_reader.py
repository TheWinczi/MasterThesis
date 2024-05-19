from typing import Iterable, Optional, NamedTuple
from collections import defaultdict
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
            markers = ('o', 'x', 's', 'D', '*')
            acc_best, lr_best, alpha_best = 0, 0, 0
            for (lr, res), marker in zip(results.items(), markers):
                accs = list(map(lambda r: r.acc, res))
                accs_stds = list(map(lambda r: r.acc_std, res))
                axis.errorbar(xs, accs, yerr=accs_stds, fmt=f'-.{marker}', capsize=5, label=f'lr={lr}')
                
                acc_max = max(accs)
                if acc_max > acc_best:
                    acc_best, lr_best = acc_max, lr
                    alpha_best = list(filter(lambda r: r.acc == acc_best, res))[0].alpha
            
            axis.legend()
            axis.set_xticks(xs, alphas)
            axis.set_title(finder.dataset.capitalize())
            axis.set_ylabel('Średnia skuteczność [%]')
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


if __name__ == '__main__':
    base_path = os.path.join('..', '..', 'results', 'gridsearch')
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
        result_paths=[os.path.join(base_path, 'imagenet_pretrained', f'seed_{i}') for i in range(no_measures)], 
        datasets_sets=ds_sets
    )

    print(25*'*', 'First Task Pretrained', 25*'*', '\n')
    plot_grid_search_results(
        experiments=experiments, 
        result_paths=[os.path.join(base_path, 'first_task_pretraining', f'seed_{i}') for i in range(no_measures)],
        datasets_sets=ds_sets
    )
