#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict


def create_empty_matrix(nodes):
    table = []
    for i in range(len(nodes)):
        table.append([0 * len(nodes)])
    return table


def get_edges(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    nodes = []
    d = defaultdict(int)
    d_edges = {}
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
        d_edges[int(parts[0])] = [*d_edges.get(int(parts[0]), []), *[int(parts[1])]]
        nodes = [*nodes, *[int(parts[0]), int(parts[1])]]

    import operator

    for n in nodes:
        d[n] += 1

    sorted_x = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

    return edges, list(set(nodes)), sorted_x


if __name__ == '__main__':

    file_location = '/home/javi/PycharmProjects/discrete-optimization/coloring/data/gc_100_1'
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(get_edges(input_data))