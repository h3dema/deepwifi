#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This module implements DeepQL - version 1

    the DeepQL uses an MLP Q-network
    it is retrained only after 'episodes' iterations
    the training uses replay memory

    bibliographic references:
    ========================

    VAN HASSELT, Hado; GUEZ, Arthur; SILVER, David. Deep reinforcement learning with double q-learning.
    In: Thirtieth AAAI conference on artificial intelligence. 2016.

    HASSELT, Hado V. Double Q-learning. In: Advances in Neural Information Processing Systems. 2010. p. 2613-2621.

    WANG, Ziyu et al. Dueling network architectures for deep reinforcement learning. arXiv preprint arXiv:1511.06581, 2015.

    CLEMENTE, Alfredo V.; CASTEJÓN, Humberto N.; CHANDRA, Arjun. Efficient parallel methods for deep reinforcement learning.
    arXiv preprint arXiv:1705.04862, 2017.

    MNIH, Volodymyr et al. Asynchronous methods for deep reinforcement learning.
    In: International conference on machine learning. 2016. p. 1928-1937.

"""
from __future__ import print_function
import time
from datetime import datetime
import numpy as np
from collections import deque
from copy import deepcopy
import pickle

import logging
logging.basicConfig(level=logging.INFO)


def softmax(z):
    """ returns the softmax function (probabilities) given an array z

        @param z: an 1D array of float
        @type z: np.array

        @return: softmax(x)
        @rtype: np.array
    """
    assert len(z.shape) == 1
    e_x = np.exp(z - np.max(z))
    return e_x / sum(e_x)


def softmax_2d(z):
    """
        @param z: an array (2, n)
        @type z: np.array
    """
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # to broadcast corretly
    return e_x / div


class DeepQL(object):

    """
        ref. https://keon.io/deep-q-learning/
             https://github.com/simoninithomas/deep_q_learning/blob/master/DeepQL%20Cartpole.ipynb
             https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
             https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
    """

    TO_MUCH_ERROR_IN_A_ROW = 20  # if that much error occurs, stop the program. None disable this test.

    def __init__(self,
                 env,  # the environment object
                 model,  # Keras model
                 memory,
                 timesteps=1,
                 epsilon=0.01,  # exploration rate
                 epsilon_min=0.1,
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 gamma=0.95,
                 batch_size=32,  # size of the batch in the replay
                 episodes=30,  # number of iterations before replay occurs
                 epochs=1,
                 interaction_interval=30,  # wait 30 second, before another cycle
                 log_level=logging.DEBUG,
                 **kwargs):
        """
            @param env: the environment class
            @param model: the Keras model used to approximate the Q-function
            @param memory: the replay memory implementation
        """
        self.interaction_interval = interaction_interval

        self.log = logging.getLogger('DeepQL')
        self.log.setLevel(log_level)
        # self.log.addHandler(logging.StreamHandler(sys.stdout))
        self.log.info("Debug {} activated".format("is" if log_level == logging.DEBUG else "is **NOT**"))

        assert epsilon_decay is not None and epsilon_decay >= 0 and epsilon_decay <= 1, 'epsilon should be in [0,1]'
        self.epsilon = epsilon

        assert epsilon_min is None or (epsilon_min >= 0 and epsilon_min <= epsilon_decay), 'min epsilon should be in [0, epsilon]'
        self.epsilon_min = epsilon_min

        assert epsilon_decay is None or (epsilon_decay >= 0 and epsilon_decay <= 1), 'epsilon_decay should be in [0,1]'
        self.epsilon_decay = epsilon_decay

        self.log.info("Epsilon {} Eps min {} Decay {}".format(epsilon, epsilon_min, epsilon_decay))

        self.env = env

        self.action_size = env.action_size
        self.log.info("Action size {}".format(self.action_size))

        self.state_size = env.state_size
        self.state_dim = env.state_dim
        self.log.info("State dim {}".format(self.state_dim))

        self.memory = memory
        self.replay_counter = 0
        self.batch_size = batch_size if batch_size > 0 else 1
        self.log.info("Replay memory: batch size: {}".format(self.batch_size))

        self.episodes = episodes if episodes > 0 else 1
        self.log.info("Episodes: {}".format(self.episodes))

        self.learning_rate = learning_rate
        self.gamma = gamma    # discount rate
        self.model = model

        self.epochs = epochs
        self.timesteps = timesteps

        self.can_run = True
        self.runs = 0
        # create a container to save 'timesteps' states for each AP
        self.prev_states = [deque(maxlen=timesteps) for i in range(len(env.aps))]

        self.log.info("Agent created")

    def save_model(self, model_filename='model.json'):
        """ save the model to a json file and the weights to a h5 file

            @param model_filename: the filename with '.json' extension
        """
        model_json = self.model.to_json()
        with open(model_filename, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_filename.replace('.json', '.h5'))
        opt = {'name': str(self.model.optimizer),
               'config': self.model.optimizer.get_config()
               }
        pickle.dump(opt, open(model_filename.replace('.json', '.p'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def remember(self, states, actions, next_states, rewards):
        """ pushes s, a, s', r
            @param states: list of initial state, one for each AP
            @param actions: list of actions taken, one for each AP
            @param next_states: list of next state, one for each AP
            @param rewards: list of rewards, one for each AP
        """
        self.log.debug("Push to replay: s:{} a:{} s'{} r{}".format(states, actions, next_states, rewards))
        self.memory.push(states, actions, next_states, rewards)

    def format_state_to_predict(self, values, batch_size=1):
        """ formats to use in predict, because
            predict needs first dimension ==> entries
            followed by the other dimension in values

            @param values: list of values to convert to a numpy array
            @param batch_size: defines the size of the batch (first dimension size)
            @return: a numpu array with self.timesteps,
                     composed by the self.prev_states values (saved from previous runs) and
                     the value passed as parameter
        """
        if type(values) == int:
            values = [values, ]
        v = np.array(values)
        if self.timesteps == 1:
            if len(v.shape) == 1:
                v = v.reshape(batch_size, -1)
        else:
            v = v.reshape(batch_size, self.timesteps, -1)
        self.log.debug('format_state_to_predict batch_size {} - self.timesteps {} > {}'.format(batch_size, self.timesteps, v))
        return v

    def get_q_max(self, sprime):
        """ the Q_max is calculated using the model network
            notice that you don't need to call self.format_state_to_predict() for sprime
            sprime format depends on the number of time steps, thus
            dim(s') = (1, timesteps, num_features)
            ... num_features = self.state_dim

            @param next_state: the next state s'
            @return: the Q_max for the state s'

            Q_max = max Q(s', a')
                     a'
        """
        q_prediction = self.model.predict(sprime)
        q_max = np.amax(q_prediction, axis=1)
        self.log.debug("s': {} Q max: {}".format(sprime, q_max))
        return q_max

    def update_epsilon(self):
        """ perform epsilon decay.
            To prevent the decay to occur, just set epsilon_decay to None.
            If epsilon_min is None, then decays forever.
            Otherwise decays while epsilon > epsilon_min
        """
        if self.epsilon_decay is not None and (self.epsilon_min is None or self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay
            self.log.info("New epsilon: {}".format(self.epsilon))

    def replay(self):
        """
            decides if the replay will occur, if not just returns
            uses the memory to recover a mini-batch that will be used to train the model network

        @return: True if replay occured
        """
        self.log.info('Doing replay')
        minibatch = self.memory.sample(batch_size=self.batch_size)
        self.log.debug("Minibatch:\t{}".format(minibatch))
        X = []
        Y = []
        for sequence in minibatch:
            self.log.debug("Replaying the sequence {}".format(sequence))
            # retrain the sequence (notice the steps):
            # 1) get the s' (sequence)
            sprime = self.format_state_to_predict([s.next_state for s in sequence])
            self.log.debug("next_state {}".format(sprime))

            # 2) calculate the Q-value of this element
            reward = sequence[-1].reward  # reward in the last step
            new_value = reward + self.gamma * self.get_q_max(sprime)
            self.log.debug("old reward {} --> predicted reward {}".format(reward, new_value))

            # 3) make the agent to approximately map the current state to future discounted reward
            #    has to pass to the Q-function the sequence of states (seq_state)
            #   also get w
            s = self.format_state_to_predict([x.state for x in sequence])
            self.log.debug("state {}".format(s))

            action = sequence[-1].action
            self.log.debug("action {}".format(action))

            # 4) get what is predicted for the state s (sequence)
            #    and change the returned value for the action chosen to the calculated 'new_value'
            target_f = self.model.predict(s)

            target_f[0][action] = new_value

            self.log.debug('target_f {} - shape {}'.format(target_f, target_f.shape))
            X.append(s)
            Y.append(target_f[0])

        # 6) retrain the network with the new value for the action
        # s = (batch_size, timesteps, num_features)
        # target_f = (batch_size, 1)
        n = len(minibatch)
        if n > 0:
            X = np.array(X).reshape(n, self.timesteps, -1)
            Y = np.array(Y).reshape(n, -1)

            self.log.info("X{} = {}".format(X.shape, X))  # remove print
            self.log.info("Y{} = {}".format(Y.shape, Y))  # remove print

            self.model.fit(X, Y, epochs=self.epochs, verbose=0)

            # perform decay
            self.update_epsilon()

            self.replay_counter += 1
            return True  # replay occured
        else:
            return False

    def predict_action_output(self, curr_state):
        """ Predict the reward value based on the given state
            this method formats 'curr_state' using self.format_state_to_predict() in order to call the keras predict()

            @param curr_state: the current state (one for each device)
            @return: values for all the actions
            @rtype: list
        """

        # compose the current state with the previous states to predict
        seq_states = deepcopy(self.prev_states)  # prevent from changing the original data
        n = len(curr_state)
        for i in range(n):
            s = curr_state[i]
            seq_states[i].append(s)
        batch_size = len(seq_states)
        seq_states = self.format_state_to_predict(seq_states, batch_size=batch_size)

        # self.log.debug("batch_size {}".format(batch_size))
        # self.log.debug("self.prev_states {}".format(self.prev_states))
        # self.log.debug("seq_states {} {}".format(seq_states, seq_states.shape))

        # predict the action using the Q-network
        act_values = self.model.predict(seq_states)  # return an array of action_size elements
        self.log.info("Predicted action: {}".format(act_values))
        return act_values

    def get_action_eps_greedy(self, curr_state):
        """ select the action (one for each ap), epsilon greedy way
            @param curr_state: a list of current states, one for each AP

            @return: list[int]: each entry is a number that represents the action for that state
        """
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            self.log.info("Random action")
            a = [np.random.choice(self.env.valid_actions(curr_state), 1)[0] for state in curr_state]
        else:
            # Pick the action based on the predicted reward
            act_values = self.predict_action_output(curr_state)
            self.log.info("act_values {}".format(act_values))  # remove print
            # todo: what happens if we have 2 or more actions with the same value in the same axis?
            #  you should select randomly between them
            a = np.argmax(act_values, axis=1)
        self.log.info('Selected action: {}'.format(a))
        return a

    def get_action_boltzmann(self, curr_state):
        """ select the action (one for each ap), using boltzmann
            @param curr_state: a list of current state, one for each AP

            @return: list[int]: each entry is a number that represents the action for that state
        """
        # Pick the action based on the predicted reward
        act_values = self.predict_action_output(curr_state)  # get the Q-value for each action in state

        # TODO: check the correctness of this code below
        #       probably error dealing with axis in softmax_2d
        p = softmax_2d(act_values)  # convert Q to probabilities)
        # select an action based on the probability
        a = [np.random.choice(self.n_actions, 1, p=p_action)[0] for p_action in p]
        self.log.info('Selected action: {}'.format(a))
        return a

    def get_action(self, states):
        """ overwrite this method to call self.get_action_eps_greedy() or self.get_action_boltzmann() to implement the search policy
            @param states: a list of states, one for each AP
        """
        return self.get_action_eps_greedy(states)

    def stop(self):
        """ change the stopping flag in the run(), so the program will stop at the end of the iteration
        """
        self.can_run = False

    @property
    def number_of_runs(self):
        """ number of times the program iteracted and acted upon the environment
            @rtype: int
        """
        return self.runs

    def stop_running(self):
        """
            change the flag that controls the while loop in run()
            so the agent stop at the end of that execution
        """
        self.can_run = False

    def run(self, run_id=1, wait_for_states=10, save_iterations=20):
        """ executes the control loop

            @param wait_for_states: how much time should sleep between get_states request
            @type wait_for_states: int

            @param save_iterations: every 'save_iterations' iterations, save the model and the weights

            @return: if the agents detected to much errors in a row
            @rtype: bool
        """
        assert wait_for_states > 0, "wait_for_states {} is not > 0".format(wait_for_states)

        # need to save (timesteps - 1) states to feed the network during prediction.
        states = None
        while states is None:
            # wait until a valid state is retrieved
            states = self.env.get_states()
            time.sleep(wait_for_states)
        # fill the previous state memory
        for i in range(len(self.env.aps)):
                s = states[i]
                self.prev_states[i] = deque([s for _ in range(self.timesteps)], maxlen=self.timesteps)
        self.log.info("len(self.prev_states) {}".format(len(self.prev_states)))
        self.log.info("len(self.env.aps) {}".format(len(self.env.aps)))
        self.log.info("")
        assert len(self.prev_states) == len(self.env.aps), "Something wrong with prev_states size"

        self.can_run = True
        self.runs_with_error = 0
        self.log.info("Starting running agent")

        none_states = 0
        while self.can_run and not self.env.done and (self.TO_MUCH_ERROR_IN_A_ROW is None or self.runs_with_error < self.TO_MUCH_ERROR_IN_A_ROW):
            self.log.info("Run #{}".format(self.runs))
            got_an_error = False
            t0 = datetime.now()
            self.log.info("Starting @{}".format(t0))

            # get states for each AP
            states = self.env.get_states()  # one for each AP
            self.log.info("States: {}".format(states))
            if states is None or None in states:
                got_an_error = True
                none_states += 1
            else:
                none_states = 0

                # the states are ok, so go on...
                # Decide action
                actions = self.get_action(states)
                self.log.info("Actions: {}".format(actions))

                # perform the selected actions, retrieve the next state and the reward
                next_states, rewards = self.env.make_step(actions)
                self.log.info("New state: {}".format(next_states))
                self.log.info("Rewards: {}".format(rewards))

                # check if an error occurred
                got_an_error = np.nan in rewards

            if not got_an_error:
                # save moves for posterior training
                self.remember(states, actions, next_states, rewards)

                # make next_state the new current state for the next frame.
                states = next_states

                self.runs += 1  # increment the number of iterations
                # train the model network
                # self.replay() decides if the replay will occur
                self.replay()

                # add actual state to prev_states
                for i in range(len(states)):
                    s = states[i]
                    self.prev_states[i].append(s)

                self.runs_with_error = 0  # reset the error counter
            else:
                # add another error in a row
                self.runs_with_error += 1

            t1 = datetime.now()
            _dur = (t1 - t0).seconds
            self.log.info("Ended @{} - duration {} s".format(t1, _dur))

            # wait interaction_interval seconds
            if self.interaction_interval is not None:
                _sleep = max(self.interaction_interval - _dur, 0)
                if _sleep > 0:
                    self.log.debug("Sleep {} s".format(_sleep))
                    time.sleep(_sleep)

            if self.runs % save_iterations == 0:
                # save the intermediate weights
                self.save_model()

        self.log.info("Total of {} steps".format(self.runs))
        if not self.can_run:
            self.log.info("Execution time expired.")

        if self.TO_MUCH_ERROR_IN_A_ROW is not None and self.runs_with_error > self.TO_MUCH_ERROR_IN_A_ROW:
            self.log.info("Out of tries. {} errors in a row during execution.".format(self.runs_with_error))

        if self.env.done:
            self.log.info("Game ended!!!")

        # return indicator if too much errors occured
        return self.TO_MUCH_ERROR_IN_A_ROW is not None and self.runs_with_error > self.TO_MUCH_ERROR_IN_A_ROW
