#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    this module uses two sources to calculate the MOS: from AP and Client

    * From client:
        * FR = reportedBitrate
        * Frame Loss = droppedFPS

        effectiveBitrate = (reportedBitrate * execution_time) / (execution_time + not_running_time')
        effectiveBitrate = effectiveBitrate / reportedBitrate']

    * From AP:
        * loss rate (PLR)
            packets = | rx_packets[t] - rx_packets[t-1] |
            PLR = rxdrop / (packets + rxdrop)

        * send bit rate (SBR)
            SBR = tx_bitrate / maximum tx_bitrate
"""
import numpy as np

from Environment.generic_ap import Generic_AP
from Environment.hossfeld import reward_hossfeld


class mos_hybrid(object):
    """codes the best regression obtained """
    def predict(self, X):
        """ finds the MOS for each entry (line) in X

            @param X: np.array[:, 3]. Contains three columns: fr, sbr, plr
            @return: a list of rewards, one for each line in X
        """
        assert len(X.shape) == 2, "Expected 2D array, got {}D array instead".format(len(X.shape))
        assert X.shape[1] == 3, "X should contain three columns: fr, sbr, plr"

        fr, sbr, plr = X[:, 0], X[:, 1], X[:, 2]
        d1 = 0.7845640778541565
        d2 = 2.8015859127044678
        d3 = 0.45656442642211914
        d4 = -0.31910404562950134
        d5 = -0.2297087013721466
        r = (d1 + d2 * fr + d3 * np.log(sbr)) / (1 + d4 * plr * d5 * np.square(plr))

        # round to the MOS limits
        if r < 1:
            r = 1
        elif r > 5:
            r = 5
        return r


class QoE_Hybrid(Generic_AP):
    """ defines the QoE as MOS_HYBRID
    """
    def reward(self, **kwargs):
        """ check the MOS of each station using command_ap module

            @param curr_state: current state
            @return: the reward
            @rtype: float
        """
        rewards = dict()  # saves the reward of each client from each ap
        for ap in self.aps:
            _, data = self.command_ap(ap.name, ap.port, ap.iface,
                                      "/get_mos_hybrid",
                                      extra_params={'macs': self.mac_mapping})
            rs = []
            for X in data:
                # X = (fr, sbr, plr)
                r = self.mos(X)
                rs.append(r)
            rewards[ap.id] = rs  # save the reward for each client in ap.id

        # obtain the global reward
        avgs = []
        for i in rewards:
            m = np.average(rewards[i])
            avgs.append(m)
        if np.any(np.isnan(avgs)):
            # if found a nan, that means an error (e.g. a disconnected station)
            r = [np.nan for _ in rewards]
        else:
            avgs = np.array(avgs)
            C = self.DEFAULT_C if 'C' not in kwargs else kwargs['C']
            r = reward_hossfeld(avgs, C=C)
            self.log.info("Hossfeld reward: {} C: {}".format(r, C))

        return r

    def get_model(self, **kwargs):
        """ get the module from the file

            @return: the model object that constains .fit() and .predict()
        """
        # don't use model. The model is hardcoded in mos_hybrid()
        return mos_hybrid()
