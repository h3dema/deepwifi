#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This module defines the interface for the replay memory buffer
"""
from collections import namedtuple
from abc import abstractmethod
import logging

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', ))


class Memory(object):

    def __init__(self, log_level=logging.DEBUG):
        self.log = logging.getLogger('Replay')
        self.log.setLevel(log_level)

    @abstractmethod
    def push(self, *args):
        raise Exception("Should be implemented in descendent class")

    @abstractmethod
    def sample(self, batch_size):
        raise Exception("Should be implemented in descendent class")

    @abstractmethod
    def __len__(self):
        """return the current number of elements stored in the memory"""
        raise Exception("Should be implemented in descendent class")
