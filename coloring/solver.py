#!/usr/bin/python
# -*- coding: utf-8 -*-


import operator
from collections import defaultdict


class GraphColoring(object):
    def __init__(self, edges):
        self.edges = edges
        self.nodes, self.d_edges = self.get_edges()
        self.degree_dict, self.degree_sorted = self.get_degree()
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

    def get_degree(self):
        d = defaultdict(int)
        for n in self.nodes:
            d[n] += 1
        sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        return d, sorted_d

    def coloring(self):
        color = 0
        while -1 in self.colors:
            # print("colorin", color)
            for pairs in self.degree_sorted:
                if (-1 in self.colors) & (self.colors[pairs[0]] == -1):
                    # print("pares", pairs)
                    # self.colors[pairs[0]] = color
                    for node in range(len(self.colors)):
                        # print("nodo", node)
                        if (node not in self.d_edges[pairs[0]]) & (self.colors[node] == -1):
                            # print(self.colors)
                            # print(self.d_edges[pairs[0]])
                            # print(edge)
                            self.colors[node] = color
                            # print(self.colors)
                    color += 1
        # print(self.colors)

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


#!/usr/bin/python
# -*- coding: utf-8 -*-
# from typing import Generic, TypeVar, Dict, List, Optional
# from abc import ABC, abstractmethod

# V = TypeVar('V')  # variable type
# D = TypeVar('D')  # domain type


# Base class for all constraints
class Constraint(object):
    # The variables that the constraint is between
    def __init__(self, variables):
        self.variables = variables

    # Must be overridden by subclasses
    def satisfied(self, assignment):
        raise NotImplementedError('subclasses must override satisfied()!')


# A constraint satisfaction problem consists of variables of type V
# that have ranges of values known as domains of type D and constraints
# that determine whether a particular variable's domain selection is valid
class CSP(object):
    def __init__(self, variables, domains):
        self.variables = variables # variables to be constrained
        self.domains = domains # domain of each variable
        self.constraints = {}
        for variable in self.variables:
            self.constraints[variable] = []
            if variable not in self.domains:
                raise LookupError("Every variable should have a domain assigned to it.")

    def add_constraint(self, constraint):
        for variable in constraint.variables:
            if variable not in self.variables:
                raise LookupError("Variable in constraint not in CSP")
            else:
                self.constraints[variable].append(constraint)

    # Check if the value assignment is consistent by checking all constraints
    # for the given variable against it
    def consistent(self, variable, assignment):
        for constraint in self.constraints[variable]:
            if not constraint.satisfied(assignment):
                return False
        return True

    def backtracking_search(self, assignment={}):
        # assignment is complete if every variable is assigned (our base case)
        if len(assignment) == len(self.variables):
            return assignment
        # get all variables in the CSP but not in the assignment
        unassigned = [v for v in self.variables if v not in assignment]
        # get the every possible domain value of the first unassigned variable

        first = unassigned[0]
        for value in self.domains[first]:
            local_assignment = assignment.copy()
            local_assignment[first] = value
            # if we're still consistent, we recurse (continue)
            if self.consistent(first, local_assignment):
                result = self.backtracking_search(local_assignment)
                # if we didn't find the result, we will end up backtracking
                if result is not None:
                        return result
        return None


class MapColoringConstraint(object):
    def __init__(self, place1, place2):
        self.variables = [place1, place2]
        self.place1 = place1
        self.place2 = place2

    def satisfied(self, assignment):
        # If either place is not in the assignment, then it is not
        # yet possible for their colors to be conflicting
        if self.place1 not in assignment or self.place2 not in assignment:
            return True
        # check the color assigned to place1 is not the same as the
        # color assigned to place2
        return assignment[self.place1] != assignment[self.place2]


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
    nodes, d_edges = gc.get_edges()

    variables = list(set(nodes))
    domains = {}
    for variable in variables:
        domains[variable] = list(range(7))

    csp = CSP(variables, domains)
    for e in edges:
        csp.add_constraint(MapColoringConstraint(e[0], e[1]))

    dict_solution = csp.backtracking_search()
    node_count = len(set(dict_solution.values()))
    # nodes, d_edges = gc.get_edges()
    # print(gc.d_edges)
    # print(gc.degree_sorted)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, dict_solution.values()))

    return output_data



import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

