#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This module implements the replay memory buffer used in DQL
"""
# other option use collections.deque
import random

from Memory.memory import Transition
from Memory.memory import Memory


class ReplayMemory(Memory):

    def __init__(self, capacity):
        """ creates the memory
            @param capacity: size of the memory
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition
           @param args: contain the data that should be saved in the memory: state, action, next_state, reward
        """
        assert len(args) == 4, "Wrong number of parameters: state, action, next_state, reward"
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
            @param batch_size: number of elements that should be returned from the memory, if the memory does not contains this many elements, returns the whole memory
            @return: a batch sample. this is a list[[Transition], [Transition]]
        """
        # check self.memory size in case batch_size is bigger, otherwise random.sample raises error
        n = min(len(self.memory), batch_size)
        result = [[x] for x in random.sample(self.memory, n)]
        return result

    def __len__(self):
        """return the current number of elements stored in the memory"""
        return len(self.memory)


if __name__ == '__main__':

    test = ReplayMemory(10)
    test.push(10, 1, 9, -1)
    print("memory size", len(test))
    for i in range(8):
        print("inserting", i + 1)
        test.push(random.randint(1, 10), random.randint(1, 3), random.randint(1, 10), random.randint(-1, 1))
    print("memory size", len(test))
    print("sample 3 elements:", test.sample(3))

    for i in range(3):
        print("inserting", i + 1)
        test.push(random.randint(1, 10), random.randint(1, 3), random.randint(1, 10), random.randint(-1, 1))
        print("\tmemory size", len(test))

    print("sample 3 elements:", test.sample(3))

    v = test.sample(1)[0]
    print("Select 1 element", v)
    print("Get fields", v._fields)
    print("Get element.state", v.state)




