#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This module implements the replay memory buffer used in DQL with multiple timesteps and multiple APs
"""
import numpy as np
import random
import logging
import pickle

from Memory.memory import Memory
from Memory.memory import Transition


class ReplayMemoryTuple(Memory):

    def __init__(self, capacity, timesteps=1, num_devices=1, log_level=logging.DEBUG):
        super().__init__(log_level=log_level)
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.timesteps = timesteps
        self.num_devices = num_devices

    def push(self, *args):
        """Saves a transition for each controlled device
           eg. ReplayMemoryTuple.push(states, actions, next_states, rewards)
           @param args: contain a tuple that should be saved in the memory
                        the lines in args should contain: state, action, next_state, reward
                        notice then that len(args) ==  4
                        e.g.
                        args = ([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15.0, 1, 81.0, 1.0051483918549375, 41.0, 52.0, 0.0, 0.0],
                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15.0, 1, 82.0, 0.9403146552637921, 56.0, 39.0, 0.0, 0.0]
                                 ],
                                 [71, 75],
                                 [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15.0, 1, 81.0, 1.0051625154422417, 41.0, 65.0, 677422.0, 19009.0],
                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15.0, 1, 82.0, 0.9403175577814756, 56.0, 39.0, 902099.0, 29509.0]
                                 ],
                                 [1.0, 3.0])

        """
        assert np.all([len(c) == self.num_devices for c in args]), "Error: all itens in 'args' must have {} elements".format(self.num_devices)
        if self.position + 1 > len(self.memory):
            self.memory.append(None)  # open space to put new element
        self.memory[self.position] = [Transition(*[c[i] for c in args]) for i in range(self.num_devices)]
        self.position = (self.position + 1) % self.capacity
        self.log.debug("Push - Last position: {}/{}".format(self.position, self.__len__()))

    def sample(self, batch_size):
        """
            @param batch_size: number of elements that should be returned from the memory, if the memory does not contains this many elements, returns the whole memory
            @return: a batch sample
        """
        # check self.memory size in case batch_size is bigger, otherwise random.sample raises error
        m = len(self.memory)
        lst = m - self.timesteps + 1  # should have at least "timesteps" elements saved to get data

        if lst < 1:
            # can't sample, not enough timesteps saved
            self.log.debug("not enough data! {} < {}".format(m, self.timesteps))
            return []

        n = min(lst * self.num_devices, batch_size)

        idxs = set()  # idx = list[(device, position)]
        # select different sequences
        while len(idxs) < n:
            v = (random.randint(0, lst - 1), random.randint(0, self.num_devices - 1),)
            idxs.add(v)
        idxs = list(idxs)

        samples = []  # save result
        for p, d in idxs:
            sequence = [self.memory[p + i][d] for i in range(self.timesteps)]
            samples.append(sequence)
        self.log.debug("Sampled **{}** elements".format(len(samples)))
        return samples

    def __len__(self):
        """
            @return: the number of elements and devices
            @rtype: (int, int)
        """
        return len(self.memory)

    def save(self, filename):
        pickle.dump(self.memory, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    def sample(n):
        s = np.random.randint(1, 10, num_devices).tolist()
        return s

    def print_batch(b):
        print("sample:")
        for i in range(len(b)):
            print(i + 1, ":", b[i])

    num_devices = 2
    timesteps = 2
    capacity = 10

    # return same values always
    np.random.seed(seed=1)
    random.seed(1)

    test = ReplayMemoryTuple(capacity=10, timesteps=timesteps, num_devices=num_devices)

    test.push(sample(num_devices), sample(num_devices), sample(num_devices), sample(num_devices))
    print("memory size: {}".format(len(test)))
    print("memory", test.memory)

    s = test.sample(batch_size=1)
    # should not return data, because timesteps == 2, and we have only one timestep saved
    print_batch(s)
    """
    not enough data! 1 < 2
    sample:
    """

    test.push(sample(num_devices), sample(num_devices), sample(num_devices), sample(num_devices))
    print("memory size: {}".format(len(test)))
    print("memory", test.memory)
    s = test.sample(batch_size=1)
    print_batch(s)
    """
    sample:
    1 : [Transition(state=6, action=6, next_state=1, reward=8), Transition(state=3, action=6, next_state=5, reward=5)]
    """

    s = test.sample(batch_size=3)
    print_batch(s)
    """
    sample:
    1 : [Transition(state=9, action=1, next_state=2, reward=7), Transition(state=5, action=3, next_state=3, reward=8)]
    2 : [Transition(state=6, action=6, next_state=1, reward=8), Transition(state=3, action=6, next_state=5, reward=5)]
    """

    test.push(sample(num_devices), sample(num_devices), sample(num_devices), sample(num_devices))
    print("memory", test.memory)
    """
    memory [[Transition(state=6, action=6, next_state=1, reward=8), Transition(state=9, action=1, next_state=2, reward=7)],
            [Transition(state=3, action=6, next_state=5, reward=5), Transition(state=5, action=3, next_state=3, reward=8)],
            [Transition(state=8, action=8, next_state=7, reward=7), Transition(state=2, action=1, next_state=8, reward=2)]]
    """
    print_batch(test.sample(batch_size=3))
    """
    sample:
    1 : [Transition(state=3, action=6, next_state=5, reward=5), Transition(state=8, action=8, next_state=7, reward=7)]
    2 : [Transition(state=6, action=6, next_state=1, reward=8), Transition(state=3, action=6, next_state=5, reward=5)]
    3 : [Transition(state=5, action=3, next_state=3, reward=8), Transition(state=2, action=1, next_state=8, reward=2)]
    """

    # insert 5 more elements
    for i in range(5):
        test.push(sample(num_devices), sample(num_devices), sample(num_devices), sample(num_devices))
    print("memory", test.memory)
    """
    memory [[Transition(state=6, action=6, next_state=1, reward=8), Transition(state=9, action=1, next_state=2, reward=7)],
            [Transition(state=3, action=6, next_state=5, reward=5), Transition(state=5, action=3, next_state=3, reward=8)],
            [Transition(state=8, action=8, next_state=7, reward=7), Transition(state=2, action=1, next_state=8, reward=2)],
            [Transition(state=1, action=9, next_state=4, reward=8), Transition(state=2, action=9, next_state=9, reward=4)],
            [Transition(state=7, action=2, next_state=5, reward=2), Transition(state=6, action=4, next_state=9, reward=5)],
            [Transition(state=1, action=3, next_state=5, reward=8), Transition(state=4, action=1, next_state=3, reward=8)],
            [Transition(state=9, action=4, next_state=8, reward=6), Transition(state=7, action=8, next_state=5, reward=4)],
            [Transition(state=7, action=1, next_state=8, reward=8), Transition(state=9, action=3, next_state=8, reward=4)]]
    """
    print_batch(test.sample(batch_size=3))
    """
    sample:
    1 : [Transition(state=1, action=9, next_state=4, reward=8), Transition(state=7, action=2, next_state=5, reward=2)]
    2 : [Transition(state=4, action=1, next_state=3, reward=8), Transition(state=7, action=8, next_state=5, reward=4)]
    3 : [Transition(state=7, action=8, next_state=5, reward=4), Transition(state=9, action=3, next_state=8, reward=4)]
    """
