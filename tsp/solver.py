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


def simulated_annealing(points, solution, nodeCount, num_iteration):
    solution_length = calculate_tour_length(points, solution, nodeCount)
    for t in range(num_iteration, 0, -1):
        random_state = random.sample(solution, len(solution))
        tour_length = calculate_tour_length(points, random_state, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = random_state
        else:
            delta = tour_length - solution_length
            exponential = math.exp(-1 * delta / t)
            if random.random() < exponential:
                solution_length = tour_length
                solution = random_state

    for t in [0.00001 * k for k in range(100000, 0, -1)]:
        random_state = random.sample(solution, len(solution))
        tour_length = calculate_tour_length(points, random_state, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = random_state
        else:
            delta = tour_length - solution_length
            exponential = math.exp(-1 * delta / t)
            if random.random() < exponential:
                solution_length = tour_length
                solution = random_state

    return solution


def metropolis(points, solution, nodeCount, num_iteration):
    t = 1
    solution_length = calculate_tour_length(points, solution, nodeCount)
    for i in range(num_iteration):
        random_state = random.sample(solution, len(solution))
        tour_length = calculate_tour_length(points, random_state, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = random_state
        else:
            delta = tour_length - solution_length
            exponential = math.exp(-1 * delta / t)
            if random.random() < exponential:
                solution_length = tour_length
                solution = random_state

    return solution


def metropolis_2_opt(points, solution, nodeCount, num_iteration):
    t = 10
    solution_length = calculate_tour_length(points, solution, nodeCount)
    for i in range(num_iteration):
        random_state = random.sample(solution, len(solution))
        tour_length = calculate_tour_length(points, random_state, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = random_state
            solution = algorithm_2_opt(solution, points, nodeCount)
        else:
            delta = tour_length - solution_length
            exponential = math.exp(-1 * delta / t)
            if random.random() < exponential:
                solution_length = tour_length
                solution = random_state
                # solution = algorithm_2_opt(solution, points, nodeCount)

    return solution


def gen_pairs(solution):
    for i in solution:
        for j in solution[i + 1:]:
            yield(i, j)


def two_opt(solution):
    pairs = gen_pairs(range(len(solution)))
    for pair in pairs:
        solution_output = solution[:]
        solution_output[pair[0]] = solution[pair[1]]
        solution_output[pair[1]] = solution[pair[0]]
        yield solution_output


def algorithm_2_opt(solution, points, nodeCount):
    g = two_opt(solution)
    solution_length = calculate_tour_length(points, solution, nodeCount)
    for permutation in g:
        tour_length = calculate_tour_length(points, permutation, nodeCount)
        if tour_length <= solution_length:
            solution_length = tour_length
            solution = permutation[:]
    return solution


def algorithm_3_opt(solution, points, nodeCount):
    g2 = two_opt(solution)
    solution_length = calculate_tour_length(points, solution, nodeCount)
    for permutation_g2 in g2:
        g3 = two_opt(permutation_g2)
        for permutation in g3:
            tour_length = calculate_tour_length(points, permutation, nodeCount)
            if tour_length <= solution_length:
                solution_length = tour_length
                solution = permutation[:]
    return solution


# ###################################

def reverse_segment_if_better2(points, tour, j, k):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    # Given tour [...A-B...C-D...E-F...]
    C, D, E, F = tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
    d0 = length(points[C], points[D]) + length(points[E], points[F])
    d2 = length(points[C], points[E]) + length(points[D], points[F])

    if d0 > d2:
        tour[j:k] = reversed(tour[j:k])
        return -d0 + d2
    return 0


def two_opt2(tour, points):
    """Iterative improvement based on 3 exchange."""
    while True:
        delta = 0
        for (b, c) in all_segments2(len(tour)):
            delta += reverse_segment_if_better2(points, tour, b, c)
        if delta >= 0:
            break
    return tour


def all_segments2(n):
    """Generate all segments combinations"""
    return ((i, j)
        for i in range(n)
        for j in range(i + 2, n))


def reverse_segment_if_better(points, tour, i, j, k):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    # Given tour [...A-B...C-D...E-F...]
    A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
    d0 = length(points[A], points[B]) + length(points[C], points[D]) + length(points[E], points[F])
    d1 = length(points[A], points[C]) + length(points[B], points[D]) + length(points[E], points[F])
    d2 = length(points[A], points[B]) + length(points[C], points[E]) + length(points[D], points[F])
    d3 = length(points[A], points[D]) + length(points[E], points[B]) + length(points[C], points[F])
    d4 = length(points[F], points[B]) + length(points[C], points[D]) + length(points[E], points[A])

    if d0 > d1:
        tour[i:j] = reversed(tour[i:j])
        return -d0 + d1
    elif d0 > d2:
        tour[j:k] = reversed(tour[j:k])
        return -d0 + d2
    elif d0 > d4:
        tour[i:k] = reversed(tour[i:k])
        return -d0 + d4
    elif d0 > d3:
        tmp = tour[j:k] + tour[i:j]
        tour[i:k] = tmp
        return -d0 + d3
    return 0


def three_opt(tour, points):
    """Iterative improvement based on 3 exchange."""
    while True:
        delta = 0
        for (a, b, c) in all_segments(len(tour)):
            delta += reverse_segment_if_better(points, tour, a, b, c)
        if delta >= 0:
            break
    return tour


def all_segments(n):
    """Generate all segments combinations"""
    return ((i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0)))


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
    solution = range(0, nodeCount)

    # solution = metropolis(points, solution, nodeCount, 100)
    # solution = algorithm_3_opt(list(solution), points, nodeCount)
    # solution = metropolis_2_opt(points, solution, nodeCount, 1000000)
    if nodeCount != 33810:
        solution = two_opt2(list(solution), points)
        if nodeCount <= 200:
            # solution = simulated_annealing(points, solution, nodeCount, 100)
            # solution = metropolis(points, solution, nodeCount, 1000000)
            solution = three_opt(solution, points)
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

