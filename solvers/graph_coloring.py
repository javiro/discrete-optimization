import operator
from collections import defaultdict


class GraphColoring(object):
    def __init__(self, edges):
        self.edges = edges
        self.nodes, self.d_edges = self.get_edges()
        self.degree_dict, self.degree_sorted = self.get_degree()
        self.colors = self.get_colors()

    def get_colors(self):
        return [-1] * len(self.nodes)

    def get_edges(self):
        nodes = []
        d_edges = {}
        for parts in self.edges:
            d_edges[int(parts[0])] = [*d_edges.get(int(parts[0]), []), *[int(parts[1])]]
            d_edges[int(parts[1])] = [*d_edges.get(int(parts[1]), []), *[int(parts[0])]]
            nodes = [*nodes, *[int(parts[0]), int(parts[1])]]
        return nodes, d_edges

    def get_degree(self):
        d = defaultdict(int)
        for n in self.nodes:
            d[n] += 1
        sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        return d, sorted_d

    def coloring(self):
        color = 0
        while -1 in self.colors:
            for pairs in self.degree_sorted:
                self.colors[pairs[0]] = color
                for edge in range(len(self.nodes)):
                    if (edge not in self.d_edges[pairs[0]]) & (self.colors[edge] == -1):
                        self.colors[edge] = color
                color += 1

    def feasibility_checking(self):
        """
        A constraint checks if it can be satisfied given the values in the domains of its variables.
        :return:
        """
        pass

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

    gc = GraphColoring(edges)
    gc.coloring()
    solution = gc.colors
    nodes, d_edges = gc.get_edges()
    # print(gc.d_edges[88])
    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':

    file_location = '/home/javi/PycharmProjects/discrete-optimization/coloring/data/gc_100_1'
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))