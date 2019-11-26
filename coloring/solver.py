#!/usr/bin/python
# -*- coding: utf-8 -*-


import operator
from collections import defaultdict


class GraphColoring(object):
    def __init__(self, edges):
        self.edges = edges
        self.nodes, self.d_edges = self.get_edges()
        self.degree_dict, self.degree_sorted, self.degree_sorted_reversed = self.get_degree()
        self.colors = self.get_colors()

    def get_colors(self):
        nodes = list(set(self.nodes))
        return [-1] * len(nodes)

    def get_edges(self):
        nodes = []
        d_edges = {}
        for parts in self.edges:
            d_edges[int(parts[0])] = d_edges.get(int(parts[0]), []) + [int(parts[1])]
            d_edges[int(parts[1])] = d_edges.get(int(parts[1]), []) + [int(parts[0])]
            nodes = nodes + [int(parts[0]), int(parts[1])]
        return nodes, d_edges

    def get_sub_graph(self, list_of_nodes):
        sub_graph = {n: [e for e in v if e in list_of_nodes] for n, v in self.d_edges.items()
                     if (n in list_of_nodes) & (len([e for e in v if e in list_of_nodes]) > 0)}
        return sub_graph

    def get_list_of_feasible_nodes(self, current_node):
        list_of_feasible_nodes = []
        for pairs in self.degree_sorted:
            node = pairs[0]
            if node not in self.d_edges[current_node]:
                list_of_feasible_nodes.append(node)
        # sub_graph = self.get_sub_graph(list_of_feasible_nodes)
        # d = {k: len(v) for k, v in sub_graph.items()}
        # sorted_d = sorted(d.items(), key=operator.itemgetter(1))
        # list_of_feasible_nodes.reverse()
        return list_of_feasible_nodes

    def get_degree(self):
        d = defaultdict(int)
        for n in self.nodes:
            d[n] += 1
        sorted_d_reverse = sorted(d.items(), key=operator.itemgetter(1))
        sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        return d, sorted_d, sorted_d_reverse

    def coloring(self):
        color = 0
        while -1 in self.colors:
            print("colorin", color)
            for pairs in self.degree_sorted:
                if (-1 in self.colors) & (self.colors[pairs[0]] == -1):
                    # print("pares", pairs)
                    # print(self.d_edges[pairs[0]])
                    # self.colors[pairs[0]] = color
                    for pairs2 in self.degree_sorted:
                        node = pairs2[0]
                        # print("nodo", node)
                        feasibility = self.feasibility_checking(node, color, self.colors)
                        if (node not in self.d_edges[pairs[0]]) & (self.colors[node] == -1) & feasibility:
                            # print(self.colors)
                            # print(self.d_edges[pairs[0]])
                            # print(edge)
                            self.colors[node] = color
                            # print(self.colors)
                    color += 1
        # print(self.colors)

    def feasibility_checking(self, node, color, color_sub_array):
        """
        A constraint checks if it can be satisfied given the values in the domains of its variables.
        :return:
        """
        for i in range(len(color_sub_array)):
            if self.colors[i] == color:
                if node in self.d_edges[i]:
                    return False
        return True

    def pruning(self):
        """
        A constraint determines which values in the domains cannot be part of any solution.
        :return:
        """
        pass


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # build a trivial solution
    # every node has its own color
    if node_count != 1000:
        gc = GraphColoring(edges)
        gc.coloring()
        solution = gc.colors
        # nodes, d_edges = gc.get_edges()
        # print(gc.d_edges)
        # print(gc.degree_sorted)
        # print(gc.get_subgraph(list(range(20))))
        node_count = len(set(solution))
    else:
        solution = list(range(node_count))

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

