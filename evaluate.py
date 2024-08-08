#!/usr/bin/python3
import cdt
import numpy as np
from causallearn.graph.Endpoint import Endpoint


def get_rpf_stru(true_G, pre_G):
    true_edges = np.sum(true_G)
    predicted_edges = np.sum(pre_G)
    true_positive = np.sum(np.logical_and(true_G, pre_G))
    precision = true_positive / predicted_edges if predicted_edges > 0 else 0
    recall = true_positive / true_edges if true_edges > 0 else 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def SHD_nparray(gt, est, double_for_anticausal=True):
    n = gt.shape[-1]
    SHD = cdt.metrics.SHD(gt, est, double_for_anticausal) / (n * (n - 1))
    return SHD

def get_shd(adj_matrix1, adj_matrix2):
    if adj_matrix1.shape != adj_matrix2.shape:
        raise ValueError("two matrix must have same shape")

    n = adj_matrix1.shape[0]
    shd_value = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                if adj_matrix1[i, j] != adj_matrix2[i, j]:
                    shd_value += 1
                elif adj_matrix1[i, j] == 1 and adj_matrix2[j, i] == 1:
                    shd_value += 1
    n = adj_matrix1.shape[0]
    max_shd = n * (n - 1)
    normalized_shd_value = shd_value / max_shd
    return normalized_shd_value


def get_edges(graph):
    edges = set()
    for edge in graph.get_graph_edges():
        node1 = edge.get_node1().get_name()
        node2 = edge.get_node2().get_name()
        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()

        if endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:
            edges.add((node2, node1))
        elif endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:
            edges.add((node1, node2))
        elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.ARROW:
            edges.add((node1, node2))
            edges.add((node2, node1))
        elif endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.TAIL:
            edges.add((node1, node2))
            edges.add((node2, node1))

    return edges


def calculate_f1_score(graph_true, graph_pred):
    edges_true = get_edges(graph_true)
    edges_pred = get_edges(graph_pred)
    tp = len(edges_true & edges_pred)
    fp = len(edges_pred - edges_true)
    fn = len(edges_true - edges_pred)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0


    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1
def calculate_f1_score2(graph_true, graph_pred):

    edges_true = {frozenset(edge) for edge in get_edges(graph_true)}
    edges_pred = {frozenset(edge) for edge in get_edges(graph_pred)}


    tp = len(edges_true & edges_pred)
    fp = len(edges_pred - edges_true)
    fn = len(edges_true - edges_pred)


    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1

def calculate_f1_cpdag(truth, est):
    """
    Calculate the F1 score between the truth and estimated CPDAGs.
    """

    truth_node_map = {node.get_name(): node_id for node, node_id in truth.node_map.items()}
    est_node_map = {node.get_name(): node_id for node, node_id in est.node_map.items()}
    TP = 0
    FP = 0
    FN = 0

    for node_i_name, truth_node_i_id in truth_node_map.items():
        for node_j_name, truth_node_j_id in truth_node_map.items():
            if truth_node_j_id <= truth_node_i_id:
                continue  # Avoid self-loops and duplicate edges
            est_node_i_id, est_node_j_id = est_node_map[node_i_name], est_node_map[node_j_name]
            truth_ij_edge = (
            truth.graph[truth_node_i_id, truth_node_j_id], truth.graph[truth_node_j_id, truth_node_i_id])
            est_ij_edge = (est.graph[est_node_i_id, est_node_j_id], est.graph[est_node_j_id, est_node_i_id])

            # Determine if the edge is present in truth and/or est
            truth_present = truth_ij_edge[0] != 0 or truth_ij_edge[1] != 0
            est_present = est_ij_edge[0] != 0 or est_ij_edge[1] != 0

            # Calculate TP, FP, FN
            if truth_present and est_present:
                if truth_ij_edge == est_ij_edge or (
                        truth_ij_edge[0] == truth_ij_edge[1] == 1 and est_ij_edge[0] == est_ij_edge[1] == 1):
                    TP += 1
                else:
                    FP += 1
                    FN += 1
            elif est_present:
                FP += 1
            elif truth_present:
                FN += 1

    # Calculate Precision, Recall, and F1 score
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score

