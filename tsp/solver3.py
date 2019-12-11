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


def get_t_interval(upper, length):
    for x in range(length):
        yield upper - x * (upper - 1) / length
    for x in range(10, 0, -1):
        yield x * 0.1
    # for x in range(10, 0, -1):
    #     yield x * 0.01


def simulated_annealing(points, solution, nodeCount, t_initial, length, num_iteration, d):
    interval = get_t_interval(t_initial, length)
    # solution = random.sample(list(range(nodeCount)), nodeCount)
    for t in interval:
        print(t)
        # solution = metropolis(points, solution, nodeCount, num_iteration, t)
        solution = iterated_local_search_2_opt_metropolis(solution, points, nodeCount, num_iteration, t, d)
        # solution = generate_new_solution(solution)
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


def restart_sa(points, final_solution, nodeCount, t_initial, length, num_iteration, num_of_restarts, d):
    solution_length = calculate_tour_length(points, final_solution, nodeCount)
    for i in range(num_of_restarts):
        print(i)
        # random_state = get_neighbour(final_solution, int(nodeCount * 0.5))
        random_state = random.sample(final_solution, len(final_solution))
        solution = simulated_annealing(points, random_state, nodeCount, t_initial, length, num_iteration, d)
        tour_length = calculate_tour_length(points, solution, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            final_solution = solution
        print(solution_length)
    return final_solution


def greedy_closest_node(points, solution, nodeCount):
    final_solution = solution
    solution_length = calculate_tour_length(points, solution, nodeCount)
    final_length = solution_length
    for i in random.sample(range(nodeCount), 1):
        idx = list(range(0, nodeCount))
        solution = get_route(i, idx, points)
        tour_length = calculate_tour_length(points, solution, nodeCount)
        if tour_length <= final_length:
            final_length = tour_length
            final_solution = solution
    return final_solution


def get_neighbour(route, num):
    output_route = route[:]
    pair = random.sample(route, num)
    pair_reverse = pair[:]
    pair_reverse.reverse()
    pairs = zip(pair, pair_reverse)
    for p in pairs:
        output_route[p[0]] = route[p[1]]
    return output_route


def metropolis(points, solution, nodeCount, t):
    solution_length = calculate_tour_length(points, solution, nodeCount)
    random_state = generate_new_solution(solution)
    tour_length = calculate_tour_length(points, random_state, nodeCount)
    if tour_length <= solution_length:
        solution = random_state
    else:
        delta = tour_length - solution_length
        exponential = math.exp(-1 * delta / t)
        if random.random() < exponential:
            solution = random_state
    return solution


def iterated_local_search_2_opt(points, nodeCount, num_iteration):
    solution = list(range(nodeCount))
    random_state = random.sample(solution, len(solution))
    solution_length = calculate_tour_length(points, solution, nodeCount)
    for i in range(num_iteration):
        print(i, solution_length)
        solution2 = two_opt(random_state, points)
        tour_length = calculate_tour_length(points, solution2, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = solution2
        random_state = generate_new_solution(solution2)
    return solution


def generate_new_solution(solution2):
    output_route = solution2[:]
    pair = random.sample(solution2, 2)
    pair.sort()
    route_reversed = output_route[pair[0]:pair[1]]
    route_reversed.reverse()
    output_route[pair[0]:pair[1]] = route_reversed
    return output_route


def iterated_local_search_2_opt_metropolis(solution, points, nodeCount, num_iteration, t, d):
    solution_length = calculate_tour_length(points, solution, nodeCount)
    for i in range(num_iteration):
        random_state = metropolis(points, solution, nodeCount, t)
        solution2 = two_opt_metropolis(random_state, points, d)
        tour_length = calculate_tour_length(points, solution2, nodeCount)
        print(tour_length)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = solution2
    return solution


def reverse_segment_if_better_metropolis(points, tour, j, k):
    """If reversing tour[i:j] would make the tour shorter, then do it.
    d(t1, t2)+d(t3, t4) > d(t2,t4)+d(t1, t3). I.e. distance d(t2, t4)
    can't be too large for such a move to be actually
    useful. Thus, I would only examine candidate t4's that are not too far from t2,"""
    # Given tour [...A-B...C-D...E-F...]
    C, D, E, F = tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
    d0 = length(points[C], points[D]) + length(points[E], points[F])
    d2 = length(points[C], points[E]) + length(points[D], points[F])
    if d0 > d2:
        tour[j:k] = reversed(tour[j:k])


def two_opt_metropolis(tour, points, d):
    """Iterative improvement based on 3 exchange."""
    for (b, c) in all_segments_k(len(tour), d):
        reverse_segment_if_better_metropolis(points, tour, b, c)
    return tour


def all_segments_k(n, d):
    return ((i, j) for i in range(n) for j in d[i])

# ###################################


def get_closest_point(idx_point1, index_of_points, points):
    point = points[index_of_points[0]]
    p = index_of_points[0]
    l_init = length(points[idx_point1], point)
    for point2 in index_of_points[1:]:
        l = length(points[idx_point1], points[point2])
        if l < l_init:
            l_init = l
            p = point2
    index_of_points.remove(p)
    return index_of_points, p


def get_route(initial_point, index_of_points, points):
    route = [initial_point]
    index_of_points.remove(initial_point)
    for i in range(len(points) - 1):
        index_of_points, point = get_closest_point(route[i], index_of_points, points)
        route.append(point)
    return route


def reverse_segment_if_better(points, tour, j, k):
    """If reversing tour[i:j] would make the tour shorter, then do it.
    d(t1, t2)+d(t3, t4) > d(t2,t4)+d(t1, t3). I.e. distance d(t2, t4)
    can't be too large for such a move to be actually
    useful. Thus, I would only examine candidate t4's that are not too far from t2,"""
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


def all_segments_k(n, d):
    return ((i, j) for i in range(n) for j in d[i])


def get_k_neighbors(points, k):
    d = {}
    for i in range(len(points)):
        point1 = points[i]
        distances = []
        points_aux = list(range(len(points)))
        points_aux.remove(i)
        for j in points_aux:
            point2 = points[j]
            distances.append([j, length(point1, point2)])
        distances.sort(key=lambda x: x[1])
        d[i] = (item[0] for item in distances[:k])
    return d


def get_points(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    print(nodeCount)
    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
    return points


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

    #
    # d = get_k_neighbors(points, 100)
    from scipy.spatial import cKDTree
    tree_catalogue = cKDTree(points)
    # distance, d = tree_catalogue.query(points, 10)

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    sol = list(zip(list(range(0, nodeCount)), points))
    sol.sort(key=lambda x: x[1].y)
    sol.sort(key=lambda x: x[1].x)
    solution = [i[0] for i in sol]
    #
    # solution = greedy_closest_node(points, solution, nodeCount)
    # solution = random.sample(list(range(0, nodeCount)), nodeCount)
    if nodeCount != 33810:
        if nodeCount <= 100:
            distance, d = tree_catalogue.query(points, 100)
            t_initial = 1000
            length_sa = 100
            num_iteration = 10
            num_of_restarts = 5
            solution = restart_sa(points, solution, nodeCount, t_initial, length_sa, num_iteration, num_of_restarts, d)
        else:
            distance, d = tree_catalogue.query(points, 100)
            t_initial = 10000
            length_sa = 10
            num_iteration = 10
            solution = list(range(0, nodeCount))
            solution = two_opt(solution, points)
            solution = simulated_annealing(points, solution, nodeCount, t_initial, length_sa, num_iteration, d)
    else:
        distance, d = tree_catalogue.query(points, 100)
        t_initial = 100
        length_sa = 10
        num_iteration = 5
        solution = simulated_annealing(points, solution, nodeCount, t_initial, length_sa, num_iteration, d)
    # solution = restart_sa(points, solution, nodeCount, t_initial, length_sa, 10, num_of_restarts, 'ooo')
    # else:
    #     sol = list(zip(list(range(0, nodeCount)), points))
    #     sol.sort(key=lambda x: x[1].y)
    #     sol.sort(key=lambda x: x[1].x)
    #     solution = [i[0] for i in sol]
    #     solution = two_opt(solution, points)

    # sol = list(zip(list(range(0, nodeCount)), points))
    # sol.sort(key=lambda x: x[1].y)
    # sol.sort(key=lambda x: x[1].x)
    # solution = [i[0] for i in sol]
    # solution = restart_sa(points, solution, nodeCount, t_initial, length_sa, num_iteration, num_of_restarts, d)

    # solution = iterated_local_search_2_opt(points, nodeCount, 5000, d)

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
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
