#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    this module calculates the MOS using only data from the client

    * rt_1, rt
        * r[t] = reportedBitrate in time [t] / max_bitrate

    * srt = not_running_time / (not_running_time + execution_time)

"""
from Environment.qoe_client import QoE_Client
# from Environment.hossfeld import reward_hossfeld


class QoE_PSNR(QoE_Client):
    """ defines the QoE using PSNR (MOS) received from the client
    """

    def get_mos_from_aps(self):
        """ it considers that each AP collects from the stations their data
        """
        rewards = dict()  # saves the reward of each client from each ap
        for ap in self.aps:
            _, data = self.command_ap(ap.name, ap.port, ap.iface, "/get_mos_client")
            self.log.debug("data for MOS @ {} => {}".format(ap.name, data))
            rs = [x[3] for x in data]
            # log information
            for x in data:
                self.log.debug("R_t: {} R_t1: {} SR: {} -> MOS: {} | {}".format(x[0], x[1], x[2], x[3], x[4]))
            rewards[ap.id] = rs  # save the reward for each client in ap.id
        return rewards

    def get_mos_from_localhost(self):
        """ it considers that the controller collects data from all the stations
        """
        rewards = dict()  # saves the reward of each client from each ap
        _, data = self.command_ap('localhost', 8080, '', "/get_mos_client")  # the interface (3rd param) does not matter
        self.log.debug("data for MOS @ {} => {}".format('all', data))
        stations = {'gnu-nb3': ['cloud'],
                    'fenrir': ['storm'],
                    }
        for ap in self.aps:
            rs = []
            for sta in stations[ap.name]:
                mos = [x[3] for x in data if x[4] == sta]
                rs.extend(mos)
                # log data
                for x in [y for y in data if y[4] == sta]:
                    self.log.debug("R_t: {} R_t1: {} SR: {} -> MOS: {} | {}".format(x[0], x[1], x[2], x[3], x[4]))
            rewards[ap.id] = rs  # used the returned MOS as the reward
        return rewards

    def get_model(self, **kwargs):
        """ Uses the MOS from the client, thus there is no model.

            @return: the model object
        """
        return None
