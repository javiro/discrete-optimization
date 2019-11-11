#!/usr/bin/python
# -*- coding: utf-8 -*-


class DepthFirst(object):
    def __init__(self, items, capacity):
        self.items = items
        self.capacity = capacity

    @staticmethod
    def estimate_optimistic_value(items):
        value = 0
        for item in items:
            value += item.value
        return value

    def depth_first_branch_and_bound(self):
        value = 0
        weight = 0
        taken = [0] * len(self.items)
        best_value = 0
        estimation = self.estimate_optimistic_value(self.items)

        for item in self.items:
            if weight + item.weight <= self.capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight
