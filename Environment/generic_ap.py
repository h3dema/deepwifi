#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Environment implementation (abstract class) that represents the experiment using Video
    This class implements the basic functions to control the APs,
    but it does not implement the QoE


"""
import time
import logging
import pickle
from abc import abstractmethod
import socket

import numpy as np

import http.client
import urllib.parse

from Environment.interface_env import Interface_Env
from Environment.common import kill_aps
from Environment.common import change_channel_hostapd
from Environment.common import start_hostapd


def decode_txpower(t):
    """ convert the data in info['txpower'] which is, for example, '15.00 dBm' into 15.0

        @return: the value of the tx power
        @rtype: float
    """
    r = float(t.split()[0].strip())
    return r


class Generic_AP(Interface_Env):

    NUM_CHANNELS = 11
    NUM_TXPOWER_LEVELS = 15
    DEFAULT_C = 0.4  # used in the reward hossfeld

    def __init__(self,
                 aps,   # List[AP_Config]
                 model_filename,  # filename that contains the trained model
                 mac_mapping={},  # {'hostname':'mac'}
                 log_level=logging.DEBUG,
                 log_name='AP Controller',
                 wait_for_states=10,
                 execute_action=False,
                 ):
        """
        initialize the environment
        @param aps: list of aps controlled in the experiment

        @param model_filename: name of the file that contains the trained model
        @type model_filename: str

        @param mac_mapping: a dictionary that maps the hostname to its mac address

        @param execute_action: if True send the selected actions to the devices
        """
        super().__init__(LOG_NAME=log_name, log_level=log_level)

        self.aps = aps
        # load model from json
        self.mos_model = self.get_model(model_filename=model_filename)

        # num_states is inf because there are continuous dimensions
        self.num_states = None
        self.dim_states = 20  # (None, 20)
        self.num_actions = self.NUM_CHANNELS * self.NUM_TXPOWER_LEVELS

        self.station_data = dict()
        # used to inform command_ap the mapping between the station name and its MACs
        self.mac_mapping = mac_mapping
        self.wait_for_states = wait_for_states

        self.execute_action = execute_action
        self.last_channnel = [1 for _ in range(len(aps))]

    def command_ap(self, server, port, iface, cmd, extra_params=None):
        """
            @return: returns true if receive the response,
                     also returns the data or an empty dict (if error)
            @rtype bool, dict
        """
        conn = http.client.HTTPConnection(server, port)
        params = {'iface': iface}
        if extra_params is not None:
            params.update(extra_params)
        q = urllib.parse.urlencode(params)
        url = "{}?{}".format(cmd, q)
        try:
            conn.request(method='GET', url=url)
        except (ConnectionRefusedError, http.client.RemoteDisconnected, socket.gaierror):
            return False, {}  # Got an error
        resp = conn.getresponse()
        self.log.info("cmd: {} @ {} status:{}".format(cmd, server, resp.status))
        try:
            data = pickle.loads(resp.read())
        except (EOFError, pickle.UnpicklingError):
            data = {}
        conn.close()
        return resp.status == 200, data

    def restart_aps(self, run_id):
        """ this is done because our ap sometimes crashes. the hostapd continues to run, but does not provide a channel
        """
        aps_to_change = []
        chans = []
        for ap, channel in zip(self.aps, self.last_channnel):
            _, data = self.command_ap(ap.name, ap.port, ap.iface, '/get_info')
            ch = data.get('channel', -1)
            if ch != -1:
                continue
            aps_to_change.append(ap)
            chans.append(ch)
        if len(aps_to_change) == 0:
            # nothing to do
            return

        # alter the hostapd.conf file
        change_channel_hostapd(aps_to_change, chans)
        # restart the hostapd
        kill_aps(aps_to_change)
        # start hostapd
        start_hostapd(aps_to_change, [run_id for i in len(aps_to_change)])

    def valid_actions(self, state=None):
        """ return a list with all valid actions for a specific state,
            if state == None, return all possible states
        @param state: current state
        @return: list(int)
        """
        # TODO: check for valid actions when states is not None
        valid = list(range(self.num_actions))  # now we always return all actions
        return valid

    def one_hot(self, channel):
        """ code the channel using one-hot encoding
        @param channel:
        @type channel: int
        @return: the channel hot encoded
        @rtype: list(int)
        """
        assert channel > 0 and channel <= self.NUM_CHANNELS, "Wrong channel = {}".format(channel)
        cs = [0 for i in range(self.NUM_CHANNELS)]
        cs[channel - 1] = 1
        self.log.debug("one-hot {} = {}".format(channel, cs))
        return cs

    def get_states(self):
        """ get the states, one for each AP
            the state contains:
            - ( #stations, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11,
                tx_power, #num_neighbors, ch_noise_max, perc_phy_busy_time,
                sta_signal_avg,
                rec_bitrate_min, tx_byte_avg, rx_byte_avg )
        @return: return the value that represent the state of all APs. Returns None if an error occurs.
        """
        known_macs = set([ap.mac for ap in self.aps])
        try:
            states = []
            for ap in self.aps:
                self.log.info("Data from {} @ {}".format(ap.name, ap.iface))
                _, info = self.command_ap(ap.name, ap.port, ap.iface, '/get_info')
                self.log.info("Info: {}".format(info))

                ch = int(info['channel'])
                self.log.info("Channel: {}".format(ch))

                _, stations = self.command_ap(ap.name, ap.port, ap.iface, '/get_stations')
                self.log.info("Stations: {}".format(stations))
                num_stations = len(stations)  # number of stations now
                self.log.info("n: {}".format(num_stations))

                # check #num_neighbors
                _, scan = self.command_ap(ap.name, ap.port, ap.iface, '/get_scan_mac')
                self.log.info("Scan: {}".format(scan))
                macs = set([k for k in scan])  # the dictionary key is the mac of the detected AP
                num_neighbors = len(macs.intersection(known_macs))
                self.log.info("num_neighbors: {}".format(num_neighbors))

                _, survey = self.command_ap(ap.name, ap.port, ap.iface, '/get_survey')
                self.log.info("survey: {}".format(survey))
                chann_in_use = [v for v in survey if survey[v].get('in use', False)][0]  # we need only the channel in use
                self.log.info("survey (in use): {}".format(chann_in_use))
                survey_in_use = survey[chann_in_use]

                ch_noise_max = survey_in_use['noise']
                perc_phy_busy_time = (survey_in_use['channel busy time'] + survey_in_use['channel receive time'] + survey_in_use['channel transmit time']) \
                                      / survey_in_use['channel active time']

                # obtain the state: one state per AP, so consolidate
                signal_avg = np.average([stations[s]['signal avg'] for s in stations])
                rx_bitrate = np.average([stations[s]['rx bitrate'] for s in stations])
                # detrend tx_bytes and rx_bytes
                tx_bytes = 0
                rx_bytes = 0
                for k in stations:
                    if k not in self.station_data:
                        self.station_data[k] = dict()
                        self.station_data[k]['tx bytes'] = stations[k]['tx bytes']
                        self.station_data[k]['rx bytes'] = stations[k]['rx bytes']

                    tx_bytes = stations[k]['tx bytes'] - self.station_data[k]['tx bytes']
                    rx_bytes = stations[k]['rx bytes'] - self.station_data[k]['rx bytes']

                    # save to use in the next round
                    self.station_data[k]['tx bytes'] = stations[k]['tx bytes']
                    self.station_data[k]['rx bytes'] = stations[k]['rx bytes']

                # save the AP's state
                state = [num_stations] + \
                        self.one_hot(ch) + \
                        [decode_txpower(info['txpower']),
                         num_neighbors,  # num_neighbors
                         ch_noise_max,
                         perc_phy_busy_time,
                         signal_avg,
                         rx_bitrate,
                         tx_bytes,
                         rx_bytes,
                         ]
                if np.any(np.isnan(state)):
                    # some reading got nan == error
                    states = None
                    break
                states.append(state)  # get the final state for the AP
        except (KeyError, ValueError, IndexError):
            # IndexError: can occur in chann_in_use
            # KeyError: can occur in ch, survey_in_use, ch_noise_max, perc_phy_busy_time
            states = None  # trigger an Error

        self.log.info("States: {}".format(states))
        return states

    def encode_action(self, txpower, channel):
        """
            @param action: an integer that represents the action
            @return: decoded values of txpower (1 to 15 dBm) and channel (1 to 11)
        """
        assert channel > 0 and txpower > 0

        action = (channel - 1) * self.NUM_TXPOWER_LEVELS + (txpower - 1)
        return action

    def decode_action(self, action):
        """
            @param action: an integer that represents the action
            @return: decoded values of txpower (1 to 15 dBm) and channel (1 to 11)
        """
        channel = action // self.NUM_TXPOWER_LEVELS + 1
        txpower = action % self.NUM_TXPOWER_LEVELS + 1
        return txpower, channel

    def setup_device(self, ap, txpower, channel):
        """ change the tx power and the ap's channel

            @param ap: the ap
            @param txpower: tx power (from 1 to 15 dBm)
            @param channel: the 2.4GHz channel number (1 to 11)
        """
        assert txpower in range(1, 16)
        assert channel in range(1, 12)

        _, data = self.command_ap(ap.name, ap.port, ap.iface, '/get_info')
        ch = data.get('channel', -1)
        if ch not in [-1, channel]:
            # send command to change channel, if the channel is different
            self.log.info("last_channnel {} ==> new channel {}".format(ch, channel))
            self.command_ap(ap.name, ap.port, ap.iface,
                            '/set_channel', extra_params={'new_channel': channel})
        else:
            return False

        self.command_ap(ap.name, ap.port, ap.iface,
                        '/set_power', extra_params={'new_power': txpower})

        self.log.info("setup_device ** ap {} txpower {} channel {}".format(ap.name, txpower, channel))
        return True

    def make_step(self, actions, retries=5):
        """send commands to aps
           @param actions: is a list of number (int) that represents the action to be taken for each AP
           @type actions: list(int)
           @param retries: number of times this function tries to get the next_state from the devices, if unsuccessful then return None in next_state
           @int retries: int

           @return: next_state: a (list of) number (int) that represents the next state
           @return: reward: a real number (reward feedback). Reward contains np.nan if an error occurs
           @rtype: list(int), float
        """
        assert retries > 0, "At least one try"
        self.log.info("make_step ** actions {} - type {}".format(actions, type(actions)))

        if self.execute_action:
            # make the move defined in action
            i = 0
            for ap, action in zip(self.aps, actions):
                # decode the number into the actual set of commands
                # send the commands to the ap
                txpower, channel = self.decode_action(action)
                self.setup_device(ap, txpower, channel)
                self.last_channnel[i] = channel
                i += 1
        else:
            # use this to just grab the data from a execution without the interference of the algorithm
            self.log.info("******************")
            self.log.info("******************")
            self.log.info("** NO STEP DONE **")
            self.log.info("******************")
            self.log.info("******************")

        # check the new state
        i = 0
        while i < retries:
            new_states = self.get_states()
            if new_states is None:
                i += 1
                time.sleep(self.wait_for_states)
            else:
                i = retries  # leave

        if new_states is None:
            # error
            return None, [np.nan]  # send back error values

        # get the reward
        reward = self.reward()
        return new_states, reward

    #
    #
    # the INTERFACE
    #
    @abstractmethod
    def get_model(self, model_filename):
        """ called in the init() code to read the model from a file
        @param model_filename: name of the file that contains the trained model
        @type model_filename: str
        @return: the model
        """
        return None
