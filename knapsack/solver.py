#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    # value = 0
    # weight = 0
    # taken = [0]*len(items)
    #
    # for item in items:
    #     if weight + item.weight <= capacity:
    #         taken[item.index] = 1
    #         value += item.value
    #         weight += item.weight

    # #################################################################
    class DepthFirst(object):
        def __init__(self, items, capacity):
            self.items = items
            self.capacity = capacity
            self.weights, self.values = self.get_weights_values()
            self.taken = [0] * len(self.items)
            self.best_value = 0
            self.best_taken = [0] * len(self.items)

        @staticmethod
        def estimate_optimistic_value(items):
            value = 0
            for item in items:
                value += item.value
            return value

        def get_weights_values(self):
            weights = []
            values = []
            for item in self.items:
                weights.append(item.weight)
                values.append(item.value)
            return weights, values

        def calculate_weight_value(self, position):
            weight = 0
            value = 0
            for i in range(position):
                weight += self.taken[i] * self.weights[i]
                value += self.taken[i] * self.values[i]
            return weight, value

        def calculate_density(self, t, position):
            density = []
            for i in range(position):
                density.append(t[i] * self.values[i] / self.weights[i])
            return density

        def linear_relaxation(self, t):
            s1 = self.calculate_density(t, len(self.taken))
            weights = self.weights
            values = self.values
            df = list(zip(*sorted(zip(s1, weights, values), reverse=True)))
            # print(df)
            # df.columns = ['density', 'weight', 'value', 'taken']
            # df = df.sort_values('density', ascending=False)
            cumulative_weight = 0
            i = 1
            while cumulative_weight < self.capacity:
                estimation_weight = cumulative_weight
                density = df[0][i - 1]
                cumulative_weight += df[1][i]
                i += 1
            # print(estimation_weight)
            x = (self.capacity - estimation_weight) * density
            return float(estimation_weight) + float(x)

        def depth_first_branch(self, position):
            # estimation = self.linear_relaxation(self.taken)
            # print(self.taken, estimation)
            weight = 0
            # taken = [0]*len(items)
            #
            # for item in items:
            #     if weight + item.weight <= capacity:
            for i in range(position, len(self.items)):
                weight, value = self.calculate_weight_value(i)
                if weight + self.items[i].weight <= self.capacity:
                    # print(weight + self.items[i].weight)
                    self.taken[i] = 1
                else:
                    self.taken[i] = 0
            final_weight, final_value = self.calculate_weight_value(len(self.taken))
            # print(self.best_value, final_value)
            if (self.best_value < final_value) & (final_weight < self.capacity):
                self.best_value = final_value
                self.best_taken = self.taken[:]

        def where(self, array, x):
            i = len(array) - 1
            while array[i] != x:
                if i > 0:
                    i -= 1
                else:
                    return None
            return i

        def go_up_tree(self):
            position = self.where(self.taken, 1)
            self.taken[position] = 0
            # print('self_taken', self.taken)
            aux_taken = self.taken[:position] + [1] * (len(self.taken) - position)
            # print('aux_taken', aux_taken)
            estimation = self.linear_relaxation(aux_taken)
            # print(estimation, self.best_value)
            if estimation > self.best_value * 0.85:
                try:
                    self.taken[position + 1] = 1
                    self.depth_first_branch(position + 1)
                except IndexError:
                    pass

        def depth_first_branch_and_bound(self):
            self.depth_first_branch(0)
            i = 0
            while not (self.taken == [0] * len(self.items)):
                # print(self.taken)
                if i % 10:
                    # print(i)
                    i += 1
                self.go_up_tree()

    class DynamicProgramming(object):
        def __init__(self, items, capacity):
            self.items = items
            self.capacity = capacity
            self.table = self.create_hash_table()
            self.weights, self.values = self.get_weights_values()
            self.taken = [0] * (1 + len(self.items))

        def get_weights_values(self):
            weights = [0.0]
            values = [0.0]
            for item in self.items:
                weights.append(item.weight)
                values.append(item.value)
            return weights, values

        def create_hash_table(self):
            table = []
            for i in range(self.capacity + 1):
                table.append([0] + ['unknown'] * len(self.items))
            return table

        def table_shape(self):
            return len(self.table) - 1, len(self.table[0]) - 1

        def populate_table(self):
            for item_index in range(1, self.table_shape()[1] + 1):
                for remaining_capacity in range(1, self.table_shape()[0] + 1):
                    self.table[remaining_capacity][item_index] = self.state_value_memoizing(item_index,
                                                                                            remaining_capacity)

        def state_value(self, item_index, remaining_capacity):
            if item_index == 0 | remaining_capacity == 0:
                return 0
            elif self.weights[item_index] > remaining_capacity:
                return self.state_value(item_index - 1, remaining_capacity)
            else:
                value_1 = self.state_value(item_index - 1, remaining_capacity)
                value_2 = self.values[item_index] + \
                          self.state_value(item_index - 1, remaining_capacity - self.weights[item_index])
                return max(value_1, value_2)

        def state_value_memoizing(self, item_index, remaining_capacity):
            if self.table[remaining_capacity][item_index] != 'unknown':
                return self.table[remaining_capacity][item_index]
            else:
                if (item_index == 0) | (remaining_capacity == 0):
                    return 0
                elif self.weights[item_index] > remaining_capacity:
                    # print(self.weights[item_index], remaining_capacity)
                    return self.state_value_memoizing(item_index - 1, remaining_capacity)
                else:
                    value_1 = self.state_value_memoizing(item_index - 1, remaining_capacity)
                    value_2 = self.values[item_index] + \
                              self.state_value_memoizing(item_index - 1, remaining_capacity - self.weights[item_index])
                    result = max(value_1, value_2)
                    self.table[remaining_capacity][item_index] = result
                    return result

        def trace_back(self):
            remaining_capacity = self.table_shape()[0]
            for item_index in range(self.table_shape()[1], 0, -1):
                if self.table[remaining_capacity][item_index] == self.table[remaining_capacity][item_index - 1]:
                    item_index -= 1
                else:
                    # item_index -= 1
                    self.taken[item_index] = 1
                    remaining_capacity -= self.weights[item_index]
            return self.taken

        def calculate_weight_value(self):
            weight = 0
            value = 0
            for i in range(len(self.weights)):
                weight += self.taken[i] * self.weights[i]
                value += self.taken[i] * self.values[i]
            return int(weight), int(value)

    # Depth First
    # d = DepthFirst(items, capacity)
    # d.depth_first_branch_and_bound()
    # value = d.best_value
    # taken = list(d.best_taken)

    # Dynamic Programming
    d = DynamicProgramming(items, capacity)
    # print(d.capacity)
    # print(d.table)
    d.populate_table()
    taken = d.trace_back()
    value, weight = d.calculate_weight_value()
    print(weight)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

