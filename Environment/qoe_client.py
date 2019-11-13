#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    this module calculates the MOS using only data from the client

    * rt_1, rt
        * r[t] = reportedBitrate in time [t] / max_bitrate

    * srt = not_running_time / (not_running_time + execution_time)

"""
import numpy as np
from Environment.generic_ap import Generic_AP
from Environment.hossfeld import reward_hossfeld
from abc import abstractmethod


class mos_client_abstract(object):
    @abstractmethod
    def predict(self, X):
        pass


class mos_client_local(mos_client_abstract):
    """ codes the best regression obtained
        see MOS_CLIENT/Generate QoE Metric -Log.ipynb for the results

        # data
        R_t = Selected bitrate for t-th chunk / Maximum bitrate
        R_t = Selected bitrate for (t-1)-th chunk / Maximum bitrate
        SR_t = Stalling length to play out the t-th chunk / (Stalling length to play out the t-th chunk + Time length of the t-th chunk)

        # Equation
        QoE( R_{t-1}, R_{t}, SR_t ) = a0 + a1 [log(R_t) + log(R_t1)] + a2 * SR_t + a3 | log(R_t) - log(R_t1) |
    """
    def __init__(self):
        self.a0 = 5.85993308
        self.a1 = -0.90426088
        self.a2 = -1.262913
        self.a3 = -1.6215644

    def predict(self, X):
        """ finds the MOS for each entry (line) in X

            @param X: np.array[:, 3]. Contains three columns: R_t, R_t1, SR
            @return: a list of rewards, one for each line in X
        """
        assert len(X.shape) == 2, "Expected 2D array, got {}D array instead".format(len(X.shape))
        assert X.shape[1] >= 3, "X should contain three columns: R_t, R_t1, SR"
        lR_t, lR_t1, SR = np.log(X[:, 0]), np.log(X[:, 1]), X[:, 2]
        r = self.a0 + self.a1 * (lR_t + lR_t1) + self.a2 * SR + self.a3 * np.abs(lR_t - lR_t1)

        # round to the MOS limits
        if r < 1:
            r = 1
        elif r > 5:
            r = 5
        return r


class mos_client(mos_client_abstract):
    """ # X
        R_t = Selected bitrate for t-th chunk / Maximum bitrate
        R_t = Selected bitrate for (t-1)-th chunk / Maximum bitrate
        SR_t = Stalling length to play out the t-th chunk / (Stalling length to play out the t-th chunk + Time length of the t-th chunk)
    """
    def __init__(self, model, kernel):
        self.model = model
        self.kernel = kernel

    def predict(self, X):
        """ finds the MOS for each entry (line) in X

            @param X: np.array[:, 3]. Contains three columns: R_t, R_t1, SR
            @return: a list of rewards, one for each line in X
        """
        assert len(X.shape) == 2, "Expected 2D array, got {}D array instead".format(len(X.shape))
        assert X.shape[1] >= 3, "X should contain three columns: R_t, R_t1, SR"

        Xt = self.kernel.transform(X)
        Y_pred = self.model.predict(Xt)

        # round to the MOS limits
        Y_pred[Y_pred < 1] = 1
        Y_pred[Y_pred > 5] = 5
        return r


class QoE_Client(Generic_AP):
    """ defines the QoE as MOS_CLIENT
    """

    def get_rs(self, data):
        rs = []
        for R_t, R_t1, SR, mos, *_ in data:
            X = np.array((R_t, R_t1, SR)).reshape(1, -1)
            self.log.info("X {}".format(X))
            r = self.mos_model.predict(X)
            if not isinstance(r, float):
                r = np.nanmean(r)
            self.log.debug("R_t: {} R_t1: {} SR: {} -> MOS: {} | {}".format(R_t, R_t1, SR, r, mos))
            rs.append(r)
        return rs

    def get_mos_from_aps(self):
        """ it considers that each AP collects from the stations their data
        """
        rewards = dict()  # saves the reward of each client from each ap
        for ap in self.aps:
            _, data = self.command_ap(ap.name, ap.port, ap.iface, "/get_mos_client")
            self.log.debug("data for MOS @ {} => {}".format(ap.name, data))
            rs = self.get_rs(data)
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
            d = []
            for sta in stations[ap.name]:
                entries = [x[:4] for x in data if x[4] == sta]
                d.extend(entries)
            rs = self.get_rs(d)
            rewards[ap.id] = rs
        return rewards

    def reward(self, **kwargs):
        """
            check the MOS of each station
            @param curr_state: current state
            @return: the reward
        """
        # rewards = self.get_mos_from_aps()
        rewards = self.get_mos_from_localhost()
        self.log.debug("Collect MOS: {}".format(rewards))

        # obtain the global reward
        avgs = []
        for i in rewards:
            m = np.nanmean(rewards[i])
            # trunc to bounds
            if m < 1:
                m = 1
            elif m > 5:
                m = 5
            avgs.append(m)
        self.log.debug("Avgs: {}".format(avgs))
        if np.any(np.isnan(avgs)):
            # if found a nan, that means an error (e.g. a disconnected station)
            r = [np.nan for _ in rewards]
        else:
            avgs = np.array(avgs)
            C = self.DEFAULT_C if 'C' not in kwargs else kwargs['C']
            r = reward_hossfeld(avgs, C=C)
            self.log.info("Hossfeld reward: {} C: {}".format(r, C))

        self.log.info("Rewards: {}".format(r))
        return r

    def get_model(self, **kwargs):
        """ The model is hard-coded in mos_client()

            @return: the model object
        """
        # use the following line to insert a local model
        # qos_model = mos_client_local()

        # read the model from a file
        if 'filename' not in kwargs:
            filename = 'qoe_client.p'
        else:
            filename = kwargs['filename']

        predictor = pickle.load(open(args.predictor, 'rb'))
        n_components, param, fit, mae, mse, rmse, amp, mapNys, y_test, y_pred = predictor
        qos_model = mos_client(fit, mapNys)

        return qos_model
