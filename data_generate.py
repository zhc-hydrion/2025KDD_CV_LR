#!/usr/bin/python3
import inspect

import networkx as nx
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, uniform, gamma

import networkx as nx
import random

from sklearn.cluster import KMeans


def Continuous2Discrete(data, bins_nums=20):
    for i in range(data.shape[-1]):
        data_c = data[:, i]
        max = np.max(data_c)
        min = np.min(data_c)
        bin_nums = bins_nums
        bins = np.linspace(min, max, num=bin_nums)
        data_d = np.digitize(data_c, bins=bins)
        data[:, i] = data_d
    return data




def equal_frequency_discretize(data, bins_nums=20, ratio=0.5, density=0.5, epoch=1):
    n = data.shape[-1]
    DisIdx = np.random.choice(np.arange(n), size=int(ratio * n), replace=False)
    for i in DisIdx:
        data_c = data[:, i]
        quantiles = np.percentile(data_c, np.linspace(0, 100, bins_nums + 1))
        data_d = np.digitize(data_c, bins=quantiles, right=True)
        data[:, i] = data_d

    if ratio == 1:
        filename = f'synthetic_discrete/synthetic_discreted_data_{density}_epoch{epoch}.csv'
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return data
    else:
        df = pd.DataFrame(data)
        discrete_continuous_info = ['discrete' if i in DisIdx else 'continuous' for i in range(n)]
        discrete_indices=[i for i, val in enumerate(discrete_continuous_info) if val == 'discrete']
        df.loc[-1] = discrete_continuous_info
        df.index = df.index + 1
        df = df.sort_index()
        filename = f'synthetic_mixed/synthetic_mixed_data_{density}_epoch{epoch}.csv'
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return data,discrete_indices





def Part2Discrete(data, bins_nums=20, ratio=0.5, density=0.5, epoch=1):
    n = data.shape[-1]
    # DisIdx = np.random.choice(np.arange(n), size=int(ratio * n), replace=False)
    # for i in DisIdx:
    #     data_c = data[:, i]
    #     max_val = np.max(data_c)
    #     min_val = np.min(data_c)
    #     bins = np.linspace(min_val, max_val, num=bins_nums)
    #     data_d = np.digitize(data_c, bins=bins)
    #     data[:, i] = data_d
    if ratio == 1:
        for i in range(n):
            data_c = data[:, i]
            max_val = np.max(data_c)
            min_val = np.min(data_c)
            bins = np.linspace(min_val, max_val, num=bins_nums)
            data_d = np.digitize(data_c, bins=bins)
            data[:, i] = data_d
        filename = f'synthetic_discrete/generated_discreted_data_{density}_epoch{epoch}.csv'
    else:
        filename = f'synthetic_discrete/generated_mixed_data_{density}_epoch{epoch}.csv'


    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return data


def generate_dag_with_density(num_nodes=10, density=0,epoch=None,multi=False):

    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))


    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    random.shuffle(possible_edges)
    # print(possible_edges)


    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_edges_to_add = int(density * max_possible_edges)
    # print(num_edges_to_add)
    # num_edges_to_add=num
    edges_added = 0

    for (i, j) in possible_edges:
        if edges_added >= num_edges_to_add:
            break
        graph.add_edge(i, j)
        if not nx.is_directed_acyclic_graph(graph):
            graph.remove_edge(i, j)
        else:
            edges_added += 1
    adj_matrix = nx.to_numpy_array(graph)
    # print(adj_matrix)


    df = pd.DataFrame(adj_matrix)

    if not multi:
        df.to_csv(f'./synthetic/adjacency_matrix{density}_epoch{epoch}.csv')
        # df.to_csv(f'./compare_data/adjacency_matrix')
    else:
        df.to_csv(f'./synthetic_multi/adjacency_matrix{density}_epoch{epoch}.csv')

    return graph


import numpy as np
import networkx as nx
import random
from scipy.stats import norm, uniform, gamma
import pandas as pd


def generate_data_from_graph_multi(graph, num_samples=500, trail=10000, dmax=5):
    for i in range(trail):
        num_nodes = graph.number_of_nodes()
        N = num_nodes
        data = {}
        G = nx.to_numpy_array(graph)
        D = np.zeros(N, dtype=int)

        functions = [
            (lambda x, y: x * 1.7 / (y + 1), 2),
            (np.sin, 1),
            (np.cos, 1),
            (np.tanh, 1),
            (lambda x: np.log(1 + abs(x)), 1)
            # (lambda x: x,1)
        ]

        distortions = [
            (lambda x: x * (1 + (2 - 1) * random.random()), 1),
            (lambda x, power: np.power(x, power), 2),
            (lambda x: np.exp(-abs(x)), 1)
            # (lambda x:x,1)
        ]

        noise_distributions1 = [
            lambda size: norm.rvs(loc=0, scale=1, size=size),
            lambda size: uniform.rvs(loc=-0.5, scale=1, size=size),
            lambda size: gamma.rvs(a=2.0, scale=0.5, size=size)
            # lambda size:np.ones(size),
        ]
        noise_distributions2 = [
            lambda size: norm.rvs(loc=0, scale=0.5, size=size),
            lambda size: uniform.rvs(loc=-0.25, scale=0.5, size=size),
            lambda size: gamma.rvs(a=2.0, scale=0.2, size=size)
            # lambda size:np.ones(size)
        ]

        for node in nx.topological_sort(graph):
            parents = list(graph.predecessors(node))
            nPA = len(parents)
            # D = random.randint(1, dmax)
            D[node]=random.randint(1, dmax)

            if nPA != 0:
                parent_data = np.concatenate([data[parent] for parent in parents], axis=1)
                weight=np.ones((parent_data.shape[1],D[node]))
                noise = random.choice(noise_distributions2)((num_samples, D[node]))
                # print(parent_data)
            else:

                parent_data = np.zeros((num_samples, 0))
                noise = random.choice(noise_distributions1)((num_samples, D[node]))

                data[node] = noise

                continue

            f_i, num_params = random.choice(functions)
            g_i, gnum_params = random.choice(distortions)

            if num_params == 2:
                if gnum_params == 2:
                    data[node] = g_i(f_i(parent_data@weight, nPA) + noise, np.random.randint(1, 4))
                else:
                    data[node] = g_i(f_i(parent_data@weight, nPA) + noise)
            else:
                if gnum_params == 2:
                    data[node] = g_i(f_i(parent_data@weight) + noise, np.random.randint(1, 4))
                else:
                    data[node] = g_i(f_i(parent_data@weight) + noise)

        combined_data = np.concatenate([data[node] for node in range(num_nodes)], axis=1)
        d_label=[]
        d_label_end=0
        for i in range(N):
            d_label.append(np.array([d_label_end,D[i]+d_label_end-1]))
            d_label_end+=D[i]
        C = np.corrcoef(combined_data, rowvar=False)
        sign = False
        for ii in range(N):
            for jj in range(N):
                if G[ii, jj] == 1:
                    tmpC = C[np.ix_(d_label[ii], d_label[jj])]
                    if np.sum(np.abs(tmpC)) < 0.3 or np.any(np.abs(tmpC) > 0.9):
                        sign = True
                        break
            if sign:
                break

        if not sign:
            print('ok')
            columns = []
            for i, (start, end) in enumerate(d_label, start=1):
                # 为每个维度生成列名，注意 range 是包含起始索引，不包含结束索引+1
                for dim in range(start, end + 1):
                    columns.append(f'x{i}_dim{dim - start + 1}')
            return pd.DataFrame(combined_data, columns=columns),d_label

    print('No suitable data generated.')
    return None


def generate_data_from_graph(graph, num_samples=500, trail=10000):
    for i in range(trail):
        num_nodes = graph.number_of_nodes()
        N = num_nodes
        data = np.zeros((num_samples, num_nodes))
        G = nx.to_numpy_array(graph)
        functions = [
            (lambda x, y: x * 1.7 / (y + 1), 2),
            (np.sin, 1),
            (np.cos, 1),
            (np.tanh, 1),
            (lambda x: np.log(1+abs(x)),1)

        ]

        distortions = [
            (lambda x: x * (1 + (2 - 1) * random.random()),1),
            (lambda x,power:np.power(x,power),2),
            (lambda x:x+np.exp(-abs(x)),1)
        ]

        noise_distributions1 = [
            lambda size: norm.rvs(loc=0, scale=1, size=size),
            lambda size: uniform.rvs(loc=-0.5, scale=1, size=size),
            lambda size: gamma.rvs(a=2.0, scale=0.5,size=size)
        ]
        noise_distributions2 = [
            lambda size: norm.rvs(loc=0, scale=0.5, size=size),
            lambda size: uniform.rvs(loc=-0.25, scale=0.5, size=size),
            lambda size: gamma.rvs(a=2.0,  scale=0.2,size=size)
        ]


        for node in nx.topological_sort(graph):
            parents = list(graph.predecessors(node))
            nPA = len(parents)
            if len(parents) != 0:
                parent_data = np.sum(data[:, parents], axis=1)
                noise = random.choice(noise_distributions2)(num_samples)

            else:
                parent_data = np.zeros(num_samples)
                noise = random.choice(noise_distributions1)(num_samples)
                data[:, node] = noise
                continue
            # print(parent_data[0])
            f_i, num_params = random.choice(functions)
            g_i, gnum_params = random.choice(distortions)

            # 根据参数数量调用函数
            if num_params == 2:
                if gnum_params==2:
                    data[:, node] = g_i(f_i(parent_data, nPA) + noise,np.random.randint(1,4))
                else:
                    data[:, node] = g_i(f_i(parent_data, nPA) + noise)
            else:
                if gnum_params==2:
                    data[:, node] = g_i(f_i(parent_data) + noise,np.random.randint(1,4))
                else:
                    data[:, node] = g_i(f_i(parent_data) + noise)

        C = np.corrcoef(data, rowvar=False)
        sign = False
        for ii in range(N):
            for jj in range(N):
                if G[ii, jj] == 1:
                    tmpC = C[np.ix_([ii], [jj])]
                    if np.sum(np.abs(tmpC)) < 0.3 or np.any(np.abs(tmpC) > 0.9):
                        sign = True
                        break
            if sign:
                break

        if not sign:
            print('ok')
            return pd.DataFrame(data, columns=[f'X{i + 1}' for i in range(num_nodes)])
        else:
            continue


def generate_synthetic_data_multi(number_var, graph_density,epoch):
    causal_graphs = [generate_dag_with_density(number_var, density=d,epoch=epoch) for d in graph_density]
    all_data = []
    for i,graph in enumerate(causal_graphs):
        # for size in sample_size:
        data,label = generate_data_from_graph_multi(graph, num_samples=2000)
        data.to_csv(f"./synthetic_multi/generated_data_example{graph_density[i]}_epoch{epoch}.csv", index=False)
        label=np.array(label)
        np.save(f'./synthetic_multi/generated_data_example{graph_density[i]}_epoch{epoch}_labels.npy',label)



