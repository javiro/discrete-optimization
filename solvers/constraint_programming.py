#!/usr/bin/python
# -*- coding: utf-8 -*-


class ConstraintProgramming(object):
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

    def propagate_engine(self):
        """
        propagate()
        {
            repeat
                select a constraint c
                if c infeasible given the domain store the
                    return failure
                else
                    apply the pruning algorithm associated with c
            until no constraint can remove any value from the domain of its variables
            return success

        :return:
        """
        pass

    def propagate_constraints(self):
        """
        a1x1 + ... + anxn >= b1y1 + ... + bmym
        ai, bi >= 0 are conntants
        xi, yj are decision variables with domains D(xi), D(yj)

        To test feasibility, take the biggest values on the left hand side and the smallest on the right.
        :return:
        """

    def test_feasibility_by_highest_lowest_values(self):
        # a1 * max(D(x1)) -- cp 2, 5 últimos minutos
        pass

    def reification(self):
        pass

    def element_constraint(self):
        """
        The ability to index a rate or a matrix with complex expressions involving variables.
        CP 3, min 13.
        :return:
        """

    def logical_combination_of_constraints(self):
        """
        The ability to implement logical combinations of constraints.
        :return:
        """
        pass

    def global_constraints(self):
        pass


class ConstraintStore(object):
    def __init__(self, items, capacity):
        self.items = items
        self.capacity = capacity
        self.weights, self.values = self.get_weights_values()
        self.taken = [0] * len(self.items)
        self.best_value = 0
        self.best_taken = [0] * len(self.items)


class Search(object):
    def __init__(self, items, capacity):
        self.items = items
        self.capacity = capacity
        self.weights, self.values = self.get_weights_values()
        self.taken = [0] * len(self.items)
        self.best_value = 0
        self.best_taken = [0] * len(self.items)
