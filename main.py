#!/usr/bin/python3
from time import time
import pandas as pd
from GES import ges
from data_generate import generate_dag_with_density, generate_data_from_graph,equal_frequency_discretize, generate_data_from_graph_multi
from evaluate import *
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
import numpy as np
from utils import initialize_general_graph_from_file, adjacency_matrix_to_general_graph


def kernel_midwidth_rbf(X):
    n = len(X)
    # ----- width of X -----
    Xmed = X

    G = np.sum(Xmed * Xmed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * (Xmed @ Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))

    return width_x


def rbf_midwidth_general(Data, max_size=500):
    Data0 = Data[:max_size, :]
    node_num = Data0.shape[1]
    width_all = []
    for i in range(node_num):
        X = Data0[:, i].reshape(-1, 1)
        width_x = kernel_midwidth_rbf(X)
        width_all.append(width_x)
    return width_all


def rbf_midwidth_general_multi(Data, max_size=500, dlabel=None):
    Data0 = Data[:max_size, :]
    node_num = len(dlabel)
    width_all = []
    for i in range(node_num):
        X = Data0[:, dlabel[i]].reshape(-1, 1)
        width_x = kernel_midwidth_rbf(X)
        width_all.append(width_x)
    return width_all


def log_results(log_file, results, label):
    with open(log_file, 'a+') as file:
        print(f"{label}:{results}", file=file)
    print(f"{label}:{results}")


def run_method(method_name, graph_density, size, m=None, dataset='synthetic', epoch=None):
    if m is not None:
        return np.array(run(method_name, graph_density, size, m, dataset=dataset, epoch=epoch))
    else:
        return np.array(run(method_name, graph_density, size, dataset=dataset, epoch=epoch))


def run_and_log_method(log_file, method_name, graph_density, size, m=None, dataset='synthetic', epoch=None):
    results = run_method(method_name, graph_density, size, m, dataset, epoch=epoch)
    log_results(log_file, results, method_name)
    return results


def generate_synthetic_data(number_var, graph_density, epoch):
    causal_graphs = [generate_dag_with_density(number_var, density=d, epoch=epoch) for d in graph_density]
    all_data = []
    for graph in causal_graphs:
        # for size in sample_size:
        data = generate_data_from_graph(graph, num_samples=2000)
        all_data.append(data)
    # Save an example data to a CSV file
    for i, example_data in enumerate(all_data):
        example_data.to_csv(f"./synthetic/generated_data_example{graph_density[i]}_epoch{epoch}.csv", index=False)


def generate_synthetic_data_multi(number_var, graph_density, epoch, dmax):
    causal_graphs = [generate_dag_with_density(number_var, density=d, epoch=epoch, multi=True) for d in graph_density]
    for i, graph in enumerate(causal_graphs):
        # for size in sample_size:
        data, label = generate_data_from_graph_multi(graph, num_samples=2000, dmax=dmax)
        data.to_csv(f"./synthetic_multi/generated_data_example{graph_density[i]}_epoch{epoch}.csv", index=False)
        label = np.array(label)
        np.save(f'./synthetic_multi/generated_data_example{graph_density[i]}_epoch{epoch}_labels.npy', label)


def run(method, graph_density, size, m=100, dataset='synthetic', epoch=None):
    result = []
    d_label = []
    print(f'{method}')
    for density_index in range(len(graph_density)):
        if dataset.startswith('synthetic'):
            X = pd.read_csv(
                f"synthetic/generated_data_example{graph_density[density_index]}_epoch{epoch}.csv").to_numpy()
            true = pd.read_csv(f'synthetic/adjacency_matrix{graph_density[density_index]}_epoch{epoch}.csv',
                               index_col=0).to_numpy().astype(int)
            true_dag = adjacency_matrix_to_general_graph(true)
            true_cpdag = dag2cpdag(true_dag)

            if dataset == 'synthetic_mixed':
                X, discrete_indices = equal_frequency_discretize(X, 5, 0.5, graph_density[density_index], epoch)
                # X,discrete_indices=read_discretized_data(f'generated_discreted_data_{graph_density[density_index]}_epoch{epoch}.csv')

            elif dataset == 'synthetic_discrete':
                X = equal_frequency_discretize(X, 5, 1, graph_density[density_index], epoch)
                print(X[1])
                # X=pd.read_csv(f'generated_discreted_data_{graph_density[density_index]}_epoch{epoch}.csv')

        elif dataset == 'multi_synthetic':
            X = pd.read_csv(
                f"synthetic_multi/generated_data_example{graph_density[density_index]}_epoch{epoch}.csv").to_numpy()

            true = pd.read_csv(f'synthetic_multi/adjacency_matrix{graph_density[density_index]}_epoch{epoch}.csv',
                               index_col=0).to_numpy().astype(int)
            true_dag = adjacency_matrix_to_general_graph(true)
            true_cpdag = dag2cpdag(true_dag)
            label = np.load(
                f'./synthetic_multi/generated_data_example{graph_density[density_index]}_epoch{epoch}_labels.npy')
            d_label = []
            for start, end in label:
                a = []
                while start <= end:
                    a.append(start)
                    start += 1
                d_label.append(a)

        elif dataset == 'sachs':
            X = np.loadtxt('./data/sachs.txt', skiprows=1)
            true_dag = initialize_general_graph_from_file('structure/sachs.graph.txt')
            true_cpdag = dag2cpdag(true_dag)

        elif dataset == 'child':
            X = np.loadtxt('./data/child.txt', skiprows=1)
            true_dag = initialize_general_graph_from_file('structure/child.graph.txt')
            true_cpdag = dag2cpdag(true_dag)

        else:
            raise ValueError('No such dataset')
        if dataset == 'child' or dataset == 'sachs':
            # indices = np.loadtxt(f"./indecs/{dataset}_{size}_{epoch}.txt", dtype=int)
            indices = np.random.choice(X.shape[0], size, replace=False)
        if 'synthetic' in dataset:
            # indices = np.loadtxt(f'./indecs/synthetic_{size}_epoch{epoch}.txt').astype(int)
            indices = np.random.choice(X.shape[0], size, replace=False)
        X = X[indices]
        if d_label:
            n = len(d_label)
        else:
            n = X.shape[1]

        param2 = None
        print((X.shape[0], n))
        if method == 'local_score_cv_multi_nym':
            ms = 500
            m = m
            MidWidth = rbf_midwidth_general_multi(X, max_size=ms, dlabel=d_label)
            param_nym = {"lambda": 0.01, "kfold": 10, "m": m, "max_sample": ms, "MidWidth": MidWidth, 'dlabel': d_label}
            param2 = param_nym
            start = time()
            Record = ges(X, method, parameters=param2)
            end = time() - start
            G = Record['G']
        else:
            if method == 'local_score_cv_general_nym' or method == 'local_score_cv_discrete_nym' or method == 'local_score_cv_general_nym':
                ms = 500
                m = m
                X = np.array(X)
                param_nym = {"lambda": 0.01, "kfold": 10, "m": m, "max_sample": ms}
                param2 = param_nym
            elif method == 'local_score_CV_multi':
                param2 = {"lambda": 0.01, "kfold": 10, 'dlabel': d_label}

            start = time()
            Record = ges(X, method, parameters=param2)
            end = time() - start
            G = Record['G']

        print(G)
        print(true_cpdag)
        shd1 = SHD(true_cpdag, G).get_shd() / (n * (n - 1) / 2)
        f1 = calculate_f1_score2(true_cpdag, G)
        result.append((f1, shd1, end))
        print(method + " " + str(graph_density[density_index]) + " " + str(end))
        with open(log_file, "a+") as file:
            print(X.shape, file=file)
            print(true_dag, file=file)
            print(G, file=file)
            print(f1, shd1, file=file)
            print(method + " " + str(graph_density[density_index]) + " " + str(end), file=file)

    return result


def run_experiment(log_file, m, graph_density, sample_size, epoch, dataset, number_var, methods=None, generetae=False,
                   multi_dmax=1):
    methods = methods
    results_different_epoch = {method: [] for method, _ in methods}
    for j in range(epoch):
        if generetae is True:
            if dataset.startswith('synthetic'):
                # 虚拟数据生成
                generate_synthetic_data(number_var, graph_density, j)
            elif dataset == 'multi_synthetic':
                generate_synthetic_data_multi(number_var, graph_density, j, dmax=multi_dmax)

        results_different_size = {method: [] for method, _ in methods}
        for size in sample_size:
            for method, method_m in methods:
                result = run_and_log_method(log_file, method, graph_density, size, method_m, dataset, epoch=j)
                results_different_size[method].append(result)

        for method in results_different_size:
            results_different_size[method] = np.stack(results_different_size[method], axis=0)
        for method, _ in methods:
            results_different_epoch[method].append(results_different_size[method])

    for method, _ in methods:
        results_different_epoch[method] = np.stack(results_different_epoch[method], axis=0)

    mean_values = {method: results.mean(axis=0) for method, results in results_different_epoch.items()}
    variance_values = {method: np.var(results, axis=0) for method, results in results_different_epoch.items()}

    for method, mean in mean_values.items():
        print(f"Mean {method}: {mean}")
    for method, var in variance_values.items():
        print(f"Variance {method}: {var}")

    return mean_values, variance_values


def save_results(filename, m, dataset, graph_density, sample_size, mean_values, variance_values, epoch):
    with open(filename, 'a+') as f:
        f.write(f'm={m}\nepoch={epoch}\n')
        f.write(f'data:{dataset}     density: {[i for i in graph_density]}\n')
        f.write(f'size: {[i for i in sample_size]}  size*density*(f1 shd)\n')
        for method in mean_values:
            f.write(f"{method}_mean: " + np.array2string(mean_values[method]) + "\n")
            f.write(f"{method}_Var: " + np.array2string(variance_values[method]) + "\n")


def main(log_file, result_file):
    log_file = log_file
    # m is the hyper parameter of nym
    m=100
    epoch = 1
    # if use synthetic dataset or set it to 0.2 it will not infulence real data
    # graph_density = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    graph_density=[0.2]
    number_var = 7
    sample_size = [200,500,1000]
    #  dataset:synthetic_continuous synthetic_mixed synthetic_discrete multi_synthetic sachs child
    dataset = 'sachs'
    print(epoch),
    methods = [
        # ('local_score_CV_general',None),
        # ('local_score_CV_multi',None),
        # ('local_score_cv_general_nym', m),
        # ('local_score_cv_multi_nym', m),
        ('local_score_cv_discrete_nym', m)
    ]
    mean_values, variance_values = run_experiment(
        log_file,
        m,
        graph_density=graph_density,
        sample_size=sample_size,
        epoch=epoch,
        dataset=dataset,
        number_var=number_var,
        methods=methods,
        generetae=True,
        multi_dmax=5
    )
    save_results(
        result_file,
        m,
        dataset=dataset,
        graph_density=graph_density,
        sample_size=sample_size,
        mean_values=mean_values,
        variance_values=variance_values,
        epoch=epoch,
    )


if __name__ == '__main__':
    log_file = './log/child_log.txt'
    result_file = './result/child_result.txt'
    main(log_file, result_file)
