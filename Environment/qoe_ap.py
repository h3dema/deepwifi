#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Environment implementation (concrete class) that represents the experiment using Video and QoE, where
    QoE is calculated using only AP parameters


    Uses a pre-trained RNN model to estimate the MOS, which consists of:
    * Bit Error Rate (BER): variation of the Bit Error Rate (BER) that can cause the MAC frame to be received with errors and trigger retransmissions that canimpact the overall performances of the system2.
    * frame aggregation: A-MPDU (MAC Protocol Data Unit) aggregation, allows many MAC frames to combine into one larger aggregated frame3.
    * number of competing stations: performance of the wireless network degrades withincreasing number of users, and
    * traffic load: percentage of traffic over the maximum throughput of the interface
    data needed: 'TX-Failed_*', 'TX-Pkts-All_*', 'AMPDUs Completed_*' --> xmit
                 'tx_bytes' --> ifconfig
                 'num_stations' --> iw station dump

    definitions:
    'FER' = 'txf_detrend' / ('txf_detrend' + 'txp_detrend')
    'AMPDU' = np.sum('AMPDUs Completed_*')
    'traffic_load' = 'tx_bytes_detrend' / 'tx_bytes'.max(iface)


"""
import numpy as np
import pickle

from Environment.generic_ap import Generic_AP
from Environment.hossfeld import reward_hossfeld


class QoE_AP(Generic_AP):
    """ defines the QoE as MOS_AP
    """
    def reward(self, **kwargs):
        """
            check the MOS of each station
            @param curr_state: current state
            @return: the reward
            @rtype: float
        """
        rewards = dict()  # saves the reward of each client from each ap
        for ap in self.aps:
            # 1. get parameters from ap
            _, data = self.command_ap(ap.name, ap.port, ap.iface, '/get_mos_ap')
            num_stations, BER, AMPDU, traffic_load = data

            # 2. calculate MOS
            mos = None
            rewards[ap.id] = mos  # save the reward for each AP in ap.id
            self.log.debug("num_stations: {} BER: {} AMPDU: {} traffic load: {} - Reward: {}".format(num_stations, BER, AMPDU, traffic_load, mos))

        # 3, take the average as the reward
        avgs = [rewards[i] for i in rewards]
        if np.any(np.isnan(avgs)):
            # if found a nan, that means an error (e.g. a disconnected station)
            r = [np.nan for _ in rewards]
        else:
            avgs = np.array(avgs)
            C = self.DEFAULT_C if 'C' not in kwargs else kwargs['C']
            r = reward_hossfeld(avgs, C=C)
            self.log.info("Hossfeld reward: {} C: {}".format(r, C))
        return r

    def get_model(self, model_filename='model-ap.p'):
        """ get the trained model from a file
        @param model_filename: the name of the file containing the trained model
        @type model_filename: str
        @return: the RNN model trained
        """
        __results = pickle.load(open(model_filename, 'rb'))
        idx = np.argmin([r['value'] for r in __results])
        clf = __results[idx]['model']
        self.log.debug(clf)
        return clf
