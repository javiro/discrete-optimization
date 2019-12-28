#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import random
from scipy.spatial import cKDTree

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Generic, TypeVar, Dict, List, Optional
from abc import ABC, abstractmethod

V = TypeVar('V')  # variable type
D = TypeVar('D')  # domain type


# Base class for all constraints
class Constraint(Generic[V, D], ABC):
    # The variables that the constraint is between
    def __init__(self, variables: List[V]) -> None:
        self.variables = variables

    # Must be overridden by subclasses
    @abstractmethod
    def satisfied(self, assignment: Dict[V, D]) -> bool:
        ...


# A constraint satisfaction problem consists of variables of type V
# that have ranges of values known as domains of type D and constraints
# that determine whether a particular variable's domain selection is valid
class CSP(Generic[V, D]):
    def __init__(self, variables: List[V], domains: Dict[V, List[D]]) -> None:
        self.variables: List[V] = variables # variables to be constrained
        self.domains: Dict[V, List[D]] = domains # domain of each variable
        self.constraints: Dict[V, List[Constraint[V, D]]] = {}
        for variable in self.variables:
            self.constraints[variable] = []
            if variable not in self.domains:
                raise LookupError("Every variable should have a domain assigned to it.")

    def add_constraint(self, constraint: Constraint[V, D]) -> None:
        for variable in constraint.variables:
            if variable not in self.variables:
                raise LookupError("Variable in constraint not in CSP")
            else:
                self.constraints[variable].append(constraint)

    # Check if the value assignment is consistent by checking all constraints
    # for the given variable against it
    def consistent(self, variable: V, assignment: Dict[V, D]) -> bool:
        for constraint in self.constraints[variable]:
            if not constraint.satisfied(assignment):
                return False
        return True

    def backtracking_search(self, assignment: Dict[V, D] = {}) -> Optional[Dict[V, D]]:
        # assignment is complete if every variable is assigned (our base case)
        if len(assignment) == len(self.variables):
            return assignment
        # get all variables in the CSP but not in the assignment
        unassigned: List[V] = [v for v in self.variables if v not in assignment]
        # get the every possible domain value of the first unassigned variable

        first: V = unassigned[0]
        for value in self.domains[first]:
            local_assignment = assignment.copy()
            local_assignment[first] = value
            # if we're still consistent, we recurse (continue)
            if self.consistent(first, local_assignment):
                result: Optional[Dict[V, D]] = self.backtracking_search(local_assignment)
                # if we didn't find the result, we will end up backtracking
                if result is not None:
                        return result
        return None


class MixedIntegerProgramming(object):
    def __init__(self, facilities, customers, facility_count, customer_count):
        self.facilities = facilities
        self.customers = customers
        self.facility_count = facility_count
        self.customer_count = customer_count
        self.solution = [-1] * len(customers)
        self.used = [0]*len(facilities)
        self.capacity_remaining = [f.capacity for f in self.facilities]

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
