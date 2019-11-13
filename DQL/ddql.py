#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This module implements Double Deep QL

"""
import numpy as np
import logging

from DQL.dql import DQL


"""LOG variable for the DDQL module"""
LOG = logging.getLogger('DDQL')
logging.basicConfig(level=logging.INFO)


class DDQL(DQL):

    """
        ref.
    """

    def get_q_max(self, sprime):
        """ the Q_max is calculated using the target network
            @param sprime: the sequence of next states (s')

            @return: the Qmax value used in the TD-error. to avoid overestimation used Q-function to predict the action a' , and
                     uses this value to obtain value of the Q(s', a') using the target network

            Q_max = Q_target(s', arg max Q(s', a'))
                                    a'
        """
        actions = np.argmax(self.model.predict(sprime), axis=1)  # uses Q-network to find the action

        # obtain the Q-max for each device
        q_target = self.target.predict(sprime)
        q_max = []
        for i, action in zip(range(len(actions)), actions):
            q_max.append(q_target[i][action])

        self.log.debug("action (from Q-network):{}".format(actions))
        self.log.debug("s': {} Q max (from Q-target): {}".format(sprime, q_max))
        return np.array(q_max)
