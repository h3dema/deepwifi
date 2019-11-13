#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    this defines the interface of the environment class environment
    all methods here should be implemented
"""
import logging
from abc import abstractmethod


class Interface_Env(object):
    def __init__(self,
                 LOG_NAME='environment',
                 log_level=logging.DEBUG):
        """
        @param LOG_NAME: the name assigned to the logger
        """
        self.log = logging.getLogger(LOG_NAME)
        self.log.setLevel(log_level)

        self.num_states = None  # this corresponds to the number of combinations of states
        self.dim_states = None  # this corresponds to the numpy shape
        self.num_actions = None

    @property
    def done(self):
        """ returns true if the objective is achieved
        """
        return False

    @property
    def state_size(self):
        """ this method is valid for discrete state space, where you can enumerate the total number of states

        @return: the number of states
        @rtype: int
        """
        self.log.debug("state_size: {}".format(self.num_states))
        return self.num_states

    @property
    def state_dim(self):
        """ the number of dimensions.
            For example, a discrete 1-D space can have state_size = 10 (i.e. 10 distinct states), and
            state_dim = 1 (because is 1D)

        @return: the number of dimensions in the state space
        @rtype: int
        """
        self.log.debug("dim_states: {}".format(self.dim_states))
        return self.dim_states

    @property
    def action_size(self):
        """ number of actions

        @return: the number of actions the system can perform
        @rtype: int
        """
        self.log.debug("action_size: {}".format(self.num_actions))
        return self.num_actions

    @abstractmethod
    def reward(self, **kwargs):
        """ should return a real number
        @return: the reward
        @rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self, state=None):
        """must be implemented in descendent

        @return: a list of all valid actions
        @rtype: list(int)
        """
        raise NotImplementedError

    @abstractmethod
    def get_states(self):
        """must be implemented in descendent
           should return a (list of) number (int) that represents the current state
        """
        raise NotImplementedError

    @abstractmethod
    def make_step(self, action):
        """must be implemented in descendent
           @param action: is a (list of) number (int) that represents the action to be taken

           @return: next_state: a (list of) number (int) that represents the next state
           @return: reward: a real number (reward feedback)
        """
        raise NotImplementedError
