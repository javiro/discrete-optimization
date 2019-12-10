#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import random

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def calculate_tour_length(points, solution, nodeCount):
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])
    return obj


def get_t_interval(lower, upper, length):
    return (upper - x*(upper-lower)/length for x in range(length))


def simulated_annealing(points, solution, nodeCount, t_initial, length, num_iteration):
    interval = get_t_interval(t_initial, 0, length)
    for t in interval:
        solution = metropolis(points, solution, nodeCount, num_iteration, t)
    return solution


def random_search(points, solution, nodeCount, num_iteration):
    solution_length = calculate_tour_length(points, solution, nodeCount)
    for i in range(num_iteration):
        random_state = random.sample(solution, len(solution))
        tour_length = calculate_tour_length(points, random_state, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = random_state
    return solution


def metropolis(points, solution, nodeCount, num_iteration, t):
    final_solution = solution
    solution_length = calculate_tour_length(points, solution, nodeCount)
    final_length = solution_length
    for i in range(num_iteration):
        random_state = random.sample(solution, len(solution))
        tour_length = calculate_tour_length(points, random_state, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = random_state
            if tour_length <= final_length:
                final_length = tour_length
                final_solution = random_state
        else:
            delta = tour_length - solution_length
            exponential = math.exp(-1 * delta / t)
            if random.random() < exponential:
                solution_length = tour_length
                solution = random_state

    return final_solution


def random_initial_state(points, solution, nodeCount, num_iteration):
    solution_length = calculate_tour_length(points, solution, nodeCount)
    for i in range(num_iteration):
        random_state = random.sample(solution, len(solution))
        solution2 = two_opt(random_state, points)
        tour_length = calculate_tour_length(points, solution2, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = solution2
    return solution

# ###################################


def get_closest_point(idx_point1, list_of_points, points):
    point = points[list_of_points[0]]
    l_init = length(points[list_of_points[idx_point1]], point)
    for point2 in list_of_points[1:]:
        l = length(points[list_of_points[idx_point1]], points[point2])
        if l < l_init:
            l_init = l
            point = point2
    list_of_points.remove(point)
    return list_of_points, point


def get_route(initial_point, list_of_points, points):
    route = [initial_point]
    list_of_points.remove(initial_point)
    for i in range(len(points)):
        points, point = get_closest_point(route[i], list_of_points, points)
        route.append(point)
    return route


def reverse_segment_if_better(points, tour, j, k):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    # Given tour [...A-B...C-D...E-F...]
    C, D, E, F = tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
    d0 = length(points[C], points[D]) + length(points[E], points[F])
    d2 = length(points[C], points[E]) + length(points[D], points[F])

    if d0 > d2:
        tour[j:k] = reversed(tour[j:k])
        return -d0 + d2
    return 0


def two_opt(tour, points):
    """Iterative improvement based on 3 exchange."""
    while True:
        delta = 0
        for (b, c) in all_segments(len(tour)):
            delta += reverse_segment_if_better(points, tour, b, c)
        if delta >= 0:
            break
    return tour


def all_segments(n):
    """Generate all segments combinations"""
    return ((i, j)
        for i in range(n)
        for j in range(i + 2, n))


def check_if_in_tabu_list(l, tabu_list):
    if l in tabu_list:
        return False
    else:
        tabu_list.pop()
        tabu_list
        return True


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    print(nodeCount)
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    sol = list(zip(list(range(0, nodeCount)), points))
    sol.sort(key=lambda x: x[1].y)
    sol.sort(key=lambda x: x[1].x)
    solution = [i[0] for i in sol]

    solution = get_route(5, list(range(0, nodeCount)), points)

    # solution = simulated_annealing(points, solution, nodeCount, 1000, 100000)
    # if nodeCount != 33810:
    #     solution = two_opt(solution, points)
    #     if nodeCount <= 100:
    #         solution = random_initial_state(points, solution, nodeCount, 10000)
    #         # solution = simulated_annealing(points, solution, nodeCount, 1000, 10000, 100)
    # else:
    #     solution = random_search(points, solution, nodeCount, 1000)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        points = solve_it(input_data)
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
