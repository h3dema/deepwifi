"""
Create a grid world

        x  0     1  ... dim_x-1
 y       +----+----+----+----+
 0       |    |    |    |    |
         +----+----+----+----+
 1       |    |    |    |    |
         +----+----+----+----+
 ..      |    |    |    |    |
         +----+----+----+----+
dim_y-1  |    |    |    |    |
         +----+----+----+----+

"""
import logging
import numpy as np
from copy import deepcopy

from Environment.interface_env import Interface_Env


LOG = logging.getLogger('grid_world')

# moves
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class grid_world(Interface_Env):

    def __init__(self, dim_x, dim_y):
        """
        @param aps: list of dictionary {id: int, ssh_user: str, shh_ip: string}
        """
        super().__init__()
        self.actions = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.actions)

        self.states = np.zeros((dim_x, dim_y))
        self.num_states = len(self.states.shape)  # number of dimensions
        self.dim_states = self.states.shape

        self.curr_state = [np.random.randint(0, dim_x), np.random.randint(0, dim_y)]
        self.end_state = [np.random.randint(0, dim_x), np.random.randint(0, dim_y)]
        LOG.info("Environment grid_world created.")

    #
    # the interface
    #
    @property
    def done(self):
        return np.all([self.curr_state[i] == self.end_state[i] for i in range(len(self.curr_state))])

    def reward(self, curr_state, **kwargs):
        """ minus the number of steps to the objective
        """
        n = len(self.end_state)
        assert len(curr_state) == n, "curr_state should be size {}".format(n)
        r = sum([-abs(curr_state[i] - self.end_state[i]) for i in range(n)])
        LOG.debug("State {} Reward {}".format(curr_state, r))
        return r

    def valid_actions(self, state=None):
        # return a list with all valid actions for a specific state,
        # if state == None, return all possible states
        actions = list(range(self.num_actions))
        if state is None:
            return actions
        else:
            invalid_actions = []
            if self.curr_state[0] == 0:  # leftmost column
                invalid_actions.append(LEFT)  # remove left action
            if self.curr_state[0] == self.states.shape[0] - 1:
                invalid_actions.append(RIGHT)  # remove right
            if self.curr_state[1] == 0:  # upper line
                invalid_actions.append(UP)  # remove up
            if self.curr_state[1] == self.states.shape[1] - 1:  # lower line
                invalid_actions.append(DOWN)  # remove down
            if len(invalid_actions) > 0:
                actions = list(set(actions).difference(invalid_actions))
            return actions

    def get_states(self):
        # position in the grid (linearized)
        return [self.curr_state, ]

    def make_step(self, action):
        """must be implemented in descendent
           @param action: is a (list of) number (int) that represents the action to be taken

           @return: next_state: a (list of) number (int) that represents the next state
           @return: reward: a real number (reward feedback)
           @rtype: list(int), list(int), float
        """
        LOG.debug("Make step: {}".format(action))
        invalid_move = False
        if action == [UP]:
            self.curr_state[1] -= 1
            if self.curr_state[1] < 0:
                self.curr_state[1] = 0
                invalid_move = True
        elif action == [DOWN]:
            self.curr_state[1] += 1
            if self.curr_state[1] >= self.states.shape[1]:
                self.curr_state[1] = self.states.shape[1] - 1
                invalid_move = True
        elif action == [LEFT]:
            self.curr_state[0] -= 1
            if self.curr_state[0] < 0:
                self.curr_state[0] = 0
                invalid_move = True
        elif action == [RIGHT]:
            self.curr_state[0] += 1
            if self.curr_state[0] >= self.states.shape[0]:
                self.curr_state[0] = self.states.shape[0] - 1
                invalid_move = True
        else:
            invalid_move = True

        if invalid_move:
            reward = -10
        elif self.done:
            # check if it is in the end location
            reward = 0
        else:
            reward = self.reward(self.curr_state)

        # return curr state, reward
        return self.get_states(), [reward, ]
