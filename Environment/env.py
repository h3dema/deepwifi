from abc import abstractclassmethod

from interface_env import Interface_Env


class environment(Interface_Env):

    def __init__(self, aps):
        """
        @param aps: list of dictionary {id: int, ssh_user: str, shh_ip: string}
        """
        super().__init__()

        assert type(aps) == list and len(aps) > 0, 'aps must contain a list with at least one ap'
        self.aps = aps
        self.n = len(self.aps)
        # define the state and action spaces
        self.num_states = None
        self.num_actions = None

    def ready(self):
        # check if the environment is ready:
        # 1 - all aps are up, and connected
        # 2 - all stations are up and connected
        return False

    def rewards(self):
        rewards = [0] * self.n
        for i in range(self.n):
            pass
        return rewards

    def act(self, actions):
        assert type(actions) == list and len(actions) == self.n, 'actions is a list with length equal to the number of controlled APs'
        # send the specified command to each ap
        for i in range(self.n):
            pass

    @property
    def done(self):
        """by defaut don't finish
           overwrite if necessary
        """
        return False

    #
    # the interface
    #
    @abstractclassmethod
    def reward(self, curr_state, **kwargs):
        """
        receives the current state, probes the environment and returns the reward
        @param curr_state: the current state
        @return: a float number representing the reward in this state
        """
        pass

    @abstractclassmethod
    def valid_actions(self, state=None):
        # return a list with all valid actions for a specific state,
        # if state == None, return all possible states
        pass

    @abstractclassmethod
    def get_states(self):
        """ return a list of values that represents the state of each AP

        """
        pass

    @abstractclassmethod
    def make_step(self, action):
        """must be implemented in descendent
           @param action: is a (list of) number (int) that represents the action to be taken

           @return: next_state: a (list of) number (int) that represents the next state (one for each AP)
           @return: reward: a real number (reward feedback)
        """
        next_states = self.get_states()
        reward = self.rewards()
        return next_states, reward
