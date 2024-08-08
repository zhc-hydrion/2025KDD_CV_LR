#!/usr/bin/python3
import pandas as pd
import numpy as np
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


def initialize_general_graph_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    nodes_line = lines[1].strip().split(';')
    nodes_names = [name.strip() for name in nodes_line]
    nodes = [GraphNode(name) for name in nodes_names]

    for i, line in enumerate(lines):
        if line.strip().startswith('Graph Edges:'):
            start_index = i + 1

            break
    edges_lines = lines[start_index:]
    edges = []
    for line in edges_lines:
        parts = line.strip().split()
        node1_name = parts[1]
        node2_name = parts[3]
        endpoint1 = Endpoint.TAIL
        endpoint2 = Endpoint.ARROW
        node1 = next(node for node in nodes if node.get_name() == node1_name)
        node2 = next(node for node in nodes if node.get_name() == node2_name)
        edge = Edge(node1, node2, endpoint1, endpoint2)
        edges.append(edge)

    graph = GeneralGraph(nodes)
    for edge in edges:
        graph.add_edge(edge)
    # print(graph)
    return graph


def find_discrete_lable(dataset, discrete_indices=None):
    if dataset == 'sachs':
        X = np.loadtxt('./data/sachs.txt', skiprows=1)
        var_types = ['discrete'] * X.shape[1]
        state_counts = [3] * X.shape[1]
        state_counts = [3] * X.shape[1]

    elif dataset == 'child':
        X = np.loadtxt('./data/child.txt', skiprows=1)
        var_types = ['discrete'] * X.shape[1]
        state_counts = [2, 2, 3, 3, 5, 2, 2, 3, 3, 2, 5, 6, 2, 3, 2, 3, 4, 3, 3, 2]
    elif dataset == 'synthetic_continuous':
        X = pd.read_csv(f"synthetic/generated_data_example0.2_epoch0.csv").to_numpy()
        var_types = ['continuous'] * X.shape[1]
        state_counts = [1] * X.shape[1]

    elif dataset == 'synthetic_discrete':
        X = pd.read_csv(f"synthetic_discrete/generated_discreted_data_0.2_epoch0.csv").to_numpy()
        var_types = ['discrete'] * X.shape[1]
        state_counts = [1] * 7
        for i in range(len(var_types)):
            feature_data = X[:, i]
            unique_values = np.unique(feature_data)
            unique_count = len(unique_values)
            # print(unique_count)
            state_counts[i] = unique_count
    elif dataset == 'synthetic_mixed':
        X = pd.read_csv(f"synthetic_discrete/generated_discreted_data_0.2_epoch0.csv").to_numpy()
        var_types = ['continuous'] * X.shape[1]
        for i in discrete_indices:
            var_types[i] = 'discrete'
        state_counts = [1] * 7
        for i in discrete_indices:
            feature_data = X[:, i]
            unique_values = np.unique(feature_data)
            unique_count = len(unique_values)
            # print(unique_count)
            state_counts[i] = unique_count
    else:
        return None
    return var_types, state_counts


def read_discretized_data(filename):
    df = pd.read_csv(filename)
    discrete_info = df.iloc[0].tolist()
    discrete_indices = [i for i, val in enumerate(discrete_info) if val == 'discrete']
    data = df.drop(index=0).to_numpy(dtype=float)
    return data, discrete_indices


def random_indecs_generate(n, sizes):
    for i in range(20):
        for size in sizes:
            indices = np.random.choice(n, size, replace=False)
            np.savetxt(f"./indecs/synthetic_{size}_epoch{i}.txt", indices, fmt='%d')


def transformer_GeneralGraph_to_adjacent(G: GeneralGraph):
    matrix = G.graph
    n = G.graph.shape[1]
    adj_matrix = np.zeros_like(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i, j] == -1 and matrix[j, i] == 1:
                adj_matrix[i, j] = 1

    return adj_matrix


def adjacency_matrix_to_general_graph(adj_matrix: np.ndarray) -> GeneralGraph:
    """
    Convert an adjacency matrix to a GeneralGraph object.

    Parameters:
    adj_matrix (np.ndarray): The adjacency matrix to convert.

    Returns:
    GeneralGraph: The resulting GeneralGraph object.
    """
    num_nodes = adj_matrix.shape[0]
    nodes = [GraphNode(f'X{i + 1}') for i in range(len(adj_matrix))]
    graph = GeneralGraph(nodes)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)  # i -> j
                graph.add_edge(edge)
            elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL)  # i -> j
                graph.add_edge(edge)

    return graph


if __name__ == '__main__':
    # find_discrete_lable('synthetic_discrete')
    random_indecs_generate(2000, [200, 500, 1000])
    # a=np.loadtxt(f"./indecs/child_{2000}_{1}.txt", dtype=int)
    # print(a)
