#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This module implements Deep QL with two networks: Q-network and target-network

    the DQL uses an MLP Q-network
    it is retrained only after 'episodes' iterations
    the training uses replay memory
"""
from copy import deepcopy
import numpy as np
import logging

from DQL.deepQL import DeepQL
from DQL.clone import clone_model


class DQL(DeepQL):

    """
        ref. https://keon.io/deep-q-learning/
             https://github.com/simoninithomas/deep_q_learning/blob/master/DQL%20Cartpole.ipynb
             https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
    """

    def __init__(self,
                 env,  # the environment object
                 model,
                 memory,
                 timesteps=1,
                 epsilon=0.1,  # exploration rate
                 epsilon_min=0.1,
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 gamma=0.95,
                 batch_size=32,  # size of the batch in the replay
                 episodes=30,  # number of iterations before replay occurs
                 epochs=1,
                 log_level=logging.DEBUG,
                 interaction_interval=30,  # wait 30 second, before another cycle
                 **kwargs):
        # call super
        super().__init__(env=env,
                         model=model,
                         memory=memory,
                         timesteps=timesteps,
                         epsilon=epsilon,
                         epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         batch_size=batch_size,
                         episodes=episodes,
                         epochs=epochs,
                         log_level=log_level,
                         interaction_interval=interaction_interval,
                         )
        # create a target model based on the self.model
        self.target = self.model
        self.copy_to_target()

    def copy_to_target(self):
        self.log.debug("copy model to target")
        try:
            model_copy = deepcopy(self.model)
        except TypeError:
            model_copy = clone_model(self.model)

        self.target = model_copy

    def copy_weights(self):
        weights = self.model.get_weights()
        self.target.set_weights(weights)

    def save_model(self, model_filename='model.json'):
        """ save the model and target networks to a json file and the weights to a h5 file
            overwritten method to save both networks

            @param model_filename: the filename with '.json' extension
        """
        super().save_model(model_filename=model_filename)  # this saves self.model
        self.target.save_weights(model_filename.replace('.json', '-target.h5'))

    def get_q_max(self, sprime):
        """ the Q_max is calculated using the target network
            @param sprime: the sequence of next states (s')

            @return: the Qmax value used in the TD-error, defined as the greedy move
            Q_max = max Q_target(s', a')
                     a'
        """
        q_prediction = self.target.predict(sprime)
        q_max = np.amax(q_prediction, axis=1)
        self.log.debug("s': {} Q max: {}".format(sprime, q_max))
        return q_max

    def replay(self):
        """
            produces the replay, that trains the model's parameters
            and if C replays occur then update target's parameters
        """
        # call the parent replay
        if super().replay() and self.replay_counter == self.episodes:
            # after C replays, the theta from model is copied to target
            self.copy_to_target()
            self.log.info("Updated target network in #{}".format(self.runs))
            self.replay_counter = 0  # zeros the count
