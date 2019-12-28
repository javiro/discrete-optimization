#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import random
from scipy.spatial import cKDTree

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def michoice(population, weights):
    prob = [w / sum(weights) for w in weights]
    r = random.random()
    i = 0
    while (i < len(prob)) and (r > prob[i]):
        i += 1
    if i == len(population):
        return population[-1]
    else:
        return population[i]


class FacilitySolver(object):
    def __init__(self, facilities, customers, facility_count, customer_count):
        self.facilities = facilities
        self.customers = customers
        self.facility_count = facility_count
        self.customer_count = customer_count
        self.solution = [-1] * len(customers)
        self.used = [0]*len(facilities)
        self.capacity_remaining = [f.capacity for f in self.facilities]

    def get_cost(self):
        # calculate the cost of the solution
        obj = sum([f.setup_cost * self.used[f.index] for f in self.facilities])
        for customer in self.customers:
            obj += length(customer.location, self.facilities[self.solution[customer.index]].location)
        return obj

    def get_initial_solution(self):
        facility_index = 0
        for customer in self.customers:
            if self.capacity_remaining[facility_index] >= customer.demand:
                self.solution[customer.index] = facility_index
                self.capacity_remaining[facility_index] -= customer.demand
            else:
                facility_index += 1
                assert self.capacity_remaining[facility_index] >= customer.demand
                self.solution[customer.index] = facility_index
                self.capacity_remaining[facility_index] -= customer.demand

        for facility_index in self.solution:
            self.used[facility_index] = 1

    def get_closest_facilities(self, n):
        customers_location = [c.location for c in self.customers]
        facilities_location = [f.location for f in self.facilities]
        tree_customers = cKDTree(facilities_location)
        distance0, d0 = tree_customers.query(customers_location, 1)
        distance, d = tree_customers.query(customers_location, n)

        #print(alt_facility)
        #print(self.facility_count)
        #print(self.customer_count)
        self.solution = d0
        sol_ini = self.solution[:]
        n_violations, indexes, dict_vio = self.set_num_violations()
        cost = self.get_cost()
        print(n_violations, indexes, dict_vio)
        print(n_violations)
        #print(negative_d)
        # print(self.get_num_violations())
        current_vio = 1
        h = 0
        while current_vio > 0:
            sol_ini = self.solution[:]
            for i in range(len(self.solution)):
                self.solution[i] = michoice(d[self.solution[i]],
                                            [self.capacity_remaining[j] for j in d[self.solution[i]]])
            self.used = [0] * len(self.facilities)
            for facility_index in self.solution:
                self.used[facility_index] = 1
            current_vio, indexes, dict_vio = self.set_num_violations()
            current_cost = self.get_cost()

            if (current_cost < cost) and (current_vio <= n_violations):
                cost = current_cost
                n_violations = current_vio
            else:
                self.solution = sol_ini[:]
            self.used = [0] * len(self.facilities)
            for facility_index in self.solution:
                self.used[facility_index] = 1
            print(h, self.get_cost())
            h += 1

    def set_num_violations(self):
        violations = {}
        indexes = []
        self.capacity_remaining = [f.capacity for f in self.facilities]
        for i in range(len(self.solution)):
            self.capacity_remaining[self.solution[i]] -= self.customers[i].demand
            if self.capacity_remaining[self.solution[i]] < 0:
                violations[self.solution[i]] = self.capacity_remaining[self.solution[i]]
                indexes.append(i)

        return sum([1 for c in self.capacity_remaining if c < 0]), indexes,\
            {k: v for k, v in sorted(violations.items(), key=lambda item: item[1])}

    def get_num_violations(self):
        capacity = self.capacity_remaining[:]
        for i in range(len(self.solution)):
            capacity[self.solution[i]] -= self.customers[i].demand
        return sum([1 for c in capacity if c < 0])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    # solution = [-1]*len(customers)
    # capacity_remaining = [f.capacity for f in facilities]
    #
    # facility_index = 0
    # for customer in customers:
    #     if capacity_remaining[facility_index] >= customer.demand:
    #         solution[customer.index] = facility_index
    #         capacity_remaining[facility_index] -= customer.demand
    #     else:
    #         facility_index += 1
    #         assert capacity_remaining[facility_index] >= customer.demand
    #         solution[customer.index] = facility_index
    #         capacity_remaining[facility_index] -= customer.demand
    #
    # used = [0]*len(facilities)
    # for facility_index in solution:
    #     used[facility_index] = 1

    fl = FacilitySolver(facilities, customers, facility_count, customer_count)
    fl.get_closest_facilities(2)

    # calculate the cost of the solution
    obj = sum([f.setup_cost * fl.used[f.index] for f in fl.facilities])
    for customer in fl.customers:
        obj += length(customer.location, fl.facilities[fl.solution[customer.index]].location)

    solution = fl.solution
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

