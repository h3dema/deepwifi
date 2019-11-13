#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Running
    =======
    There are two options:

    a) creates the configuration files (hostapd.conf, and wpa_supplicant.conf), create the script run.sh,
       copy the files to the APs and STAs
       and then runs the experiment
    python3 run_client.py --save-wpa-conf --save-hostapd-conf

    b) just runs. This pressuposes that the configuration files are copied to the devices
    python3 run_client.py  --qoe-model [client | ap | hybrid  | psnr ]

"""
import os
import sys
import argparse
import logging
import datetime
import time
import glob
from threading import Timer

from keras.models import model_from_json

from Environment.common import LOG as LOG2
from Environment.common import aps, stas
from Environment.common import conf_stas, conf_aps
from Environment.common import kill_aps, kill_stas
from Environment.common import start_devices
from Environment.common import reboot_devices
from Environment.common import run_get_set_server, kill_get_set_server, is_runnning_get_set_server

# Environment
from Environment.qoe_psnr import QoE_PSNR
from Environment.qoe_psnr2 import QoE_PSNR as QoE_PSNR2
from Environment.qoe_client import QoE_Client
from Environment.qoe_ap import QoE_AP
from Environment.qoe_hybrid import QoE_Hybrid
from Environment.testEnv import test_qoe  # used to debug

# agent
from Memory.replay_tuple import ReplayMemoryTuple
from DQL.ddql import DDQL
# model used in the Q-function
from TCN.tcnn import compiled_tcn
from TCN.tcnn import get_opt


"""create logging object to print RunClient information"""
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('RunClient')


def reboot(aps, stas):
    reboot_devices(aps)
    reboot_devices(stas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # MOS model file
    parser.add_argument('--model', type=str, default='model-client.p', help='the name of the file with the sklearn model')
    parser.add_argument("--timesteps", type=int, default=2, help="number of timesteps considered in the model")

    parser.add_argument('--keras-model', type=str, default=None, help='the name of the dir with the Keras model used for Q-function approximation')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')

    # learning parameteres
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.002, help="model learning rate")
    parser.add_argument("--dropout-rate", type=float, default=0.05, help="model dropout rate")
    parser.add_argument("--dilations", type=int, default=4, help="number of dilation levels")
    parser.add_argument("--kernel-size", type=int, default=2, help="kernel size")
    parser.add_argument("--filters", type=int, default=24, help="number of filters")

    # devices configuration
    parser.add_argument('--dont-kill-first', action='store_true', help='dont kill the applications in the devices first, before restarting the experiment')
    parser.add_argument('--save-wpa-conf', action='store_true', help='create the configuration file for the wpa_supplicant')
    parser.add_argument('--save-hostapd-conf', action='store_true', help='create the configuration file for the hostapd')

    # experiment parameters
    parser.add_argument("--initial-id", type=int, default=1, help="id of the first experiment")
    parser.add_argument("--repetitions", type=int, default=1, help="number of times the experiment should run")
    parser.add_argument("--retries", type=int, default=3, help="number of retries before reset the device")
    parser.add_argument("--timeout", type=int, default=None, help="how much time the learning agent should run (in minutes)")
    parser.add_argument('--info', action='store_true', help='log info')
    parser.add_argument("--log-dir", type=str, default='logs', help="output dir")

    parser.add_argument('--dont-run', action='store_true', help='skip the call to run the applications at the devices')
    parser.add_argument('--kill-all', action='store_true', help='kill all applications stopping the experiment')
    parser.add_argument('--reboot', action='store_true', help='reboot the devices')
    parser.add_argument('--runs-between-reboot', type=int, default=5, help='every "runs-between-reboot" runs, performs a reboot of the devices. If None does not reboot')
    parser.add_argument("--firefox-restart", type=int, default=5, help="restarts the firefox every x minutes")
    parser.add_argument("--wait-for-states", type=int, default=10, help="wait x seconds before retry calling get_states()")

    parser.add_argument("--episodes", type=int, default=10, help="number of runs the agent takes before replay")
    parser.add_argument("--interaction-interval", type=float, default=120.0, help="number of seconds between the agent interaction with the environment")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon in Q-Learning")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay in Q-Learning. None equals to no decay")

    # parser.add_argument('--debug', action='store_true', help='log debug')
    # parser.add_argument('--verbose', type=int, default=0, help='define the verbose level (0=less verbose)')

    parser.add_argument('--dont-act', action='store_true', help='prevent the environment from sending the selected actions to the devices')

    # select the QoE model
    parser.add_argument('--qoe-model', type=str, help='select the QoE model')

    parser.add_argument('--browser', type=str, default='firefox', help='select the browser used to download the video')

    args = parser.parse_args()
    assert args.browser in ['opera', 'firefox'], 'browser {} is not valid'.format(args.browser)

    if args.epsilon < 0 or args.epsilon > 1:
        print("Wrong epsilon")
        sys.exit(1)

    if args.reboot:
        reboot(aps, stas)
        print("Reboot command send to the devices")
        sys.exit(0)

    # Logging
    log_level = logging.INFO if args.info else logging.DEBUG
    logging.basicConfig(level=log_level,
                        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                        )
    # set local LOG variable
    LOG.setLevel(log_level)
    # set common module LOG variable
    LOG2.setLevel(log_level)

    LOG.debug("Configuration")
    LOG.debug('Qoe model           : {}'.format(args.qoe_model))
    LOG.debug("number of epochs    : {}".format(args.epochs))
    LOG.debug("model learning rate : {}".format(args.learning_rate))
    LOG.debug("model dropout rate  : {}".format(args.dropout_rate))
    LOG.debug("number of dilation levels: {}".format(args.dilations))
    LOG.debug("kernel size         : {}".format(args.kernel_size))
    LOG.debug("number of filters   : {}".format(args.filters))
    LOG.debug('Model               : {}'.format(args.model))
    LOG.debug("Timesteps           : {}".format(args.timesteps))
    LOG.debug("Episodes            : {}".format(args.episodes))
    LOG.debug('interaction interval: {} seconds'.format(args.interaction_interval))
    LOG.debug('Epsilon             : {}'.format(args.epsilon))
    LOG.debug('Epsilon decay       : {}'.format(args.epsilon_decay))
    LOG.debug('Environmental action is{} activated'.format(" ** NOT **" if args.dont_act else " "))
    LOG.debug('Browser             : {}'.format(args.browser))

    try:
        os.makedirs(args.log_dir)
    except FileExistsError:
        # do nothing, because dir exists
        pass
    if args.kill_all:
        # stop all applications
        LOG.info("Stopping APs")
        kill_aps(aps)
        LOG.info("Stopping Stations")
        kill_stas(stas)
        sys.exit(0)

    assert args.timesteps >= args.kernel_size, "the kernel size cannot be bigger than the number of timesteps in the input"

    valid_qoe_models = {'client': QoE_Client,
                        'ap': QoE_AP,
                        'hybrid': QoE_Hybrid,
                        'psnr': QoE_PSNR,
                        'psnr2': QoE_PSNR2,
                        'test': test_qoe}
    if args.qoe_model is None or args.qoe_model not in valid_qoe_models:
        print("The parameter --qoe-model is required, and it accepts only the following values: {}".format(", ".join(valid_qoe_models)))
        parser.print_help()
        sys.exit(0)

    # print ifo
    if not args.info:
        LOG.debug("APs:")
        for ap in aps:
            LOG.debug("#{} Conn: {}:{} MAC: {} Iface:{} SSID:{}".format(ap.id, ap.name, ap.port, ap.mac, ap.iface, ap.SSID))
        LOG.debug("STATIONS:")
        for sta in stas:
            LOG.debug("#{} Conn: {} Iface:{} SSID:{}".format(sta.id, sta.name, sta.iface, sta.SSID))

    # configure the devices -- aps and stations
    if args.save_hostapd_conf and args.qoe_model != 'test':
        conf_aps(aps)

    if args.save_wpa_conf and args.qoe_model != 'test':
        conf_stas(aps, stas, restart_ffox=args.firefox_restart, browser=args.browser)

    if not args.dont_run:

        rep_count = 0
        not_detected_errors = True
        while rep_count < args.repetitions:
            rep_id = args.initial_id + rep_count
            identification = "{:05}-{}".format(rep_id, datetime.datetime.now().strftime('%Y%m%dZ%H%M%S'))
            LOG.info("Experiment id: {}".format(identification))

            # kill devices, before try to start them
            if not args.dont_kill_first and args.qoe_model != 'test':
                kill_aps(aps)
                kill_stas(stas)
                kill_get_set_server()

            # start the devices -- controller, aps and stations
            if args.qoe_model != 'test':
                start_devices(aps, stas,
                              max_retries=args.retries,
                              _id=identification,
                              browser=args.browser)

            # create a nodejs server to collect browser data
            if not is_runnning_get_set_server():
                run_get_set_server(_id=identification, log_dir=os.path.abspath(args.log_dir))

            if not_detected_errors:
                # create a nodejs server to collect browser data
                if not is_runnning_get_set_server():
                    run_get_set_server(_id=identification, log_dir=os.path.abspath(args.log_dir))

                # TRUE: create new environment and model

                # create the environment
                mac_mapping = dict((sta.mac, sta.name) for sta in stas)

                qoe_environment = valid_qoe_models[args.qoe_model]
                env = qoe_environment(aps,
                                      mac_mapping=mac_mapping,
                                      model_filename=args.model,
                                      log_level=log_level,
                                      execute_action=not args.dont_act,
                                      )

                # the number of features corresponds to the size of the state
                num_feat = env.state_dim
                LOG.debug("num features: {}".format(num_feat))
                LOG.debug("num classes (actions): {}".format(env.action_size))

                if args.keras_model is None:
                    # create the model, using information from the environment
                    model = compiled_tcn(num_feat=num_feat,  # type: int
                                         num_classes=env.action_size,  # type: int --> number of actions
                                         nb_filters=args.filters,  # type: int
                                         kernel_size=args.kernel_size,  # type: int
                                         dilations=[2 ** i for i in range(args.dilations)],  # type: List[int]
                                         nb_stacks=1,  # type: int
                                         max_len=args.timesteps,  # type: int -- number of timesteps, None means a dynamic number of timesteps
                                         padding='causal',  # type: str
                                         use_skip_connections=False,  # type: bool
                                         return_sequences=False,  # False because we just want the last y from an input x_{1}..x_{timesteps}
                                         regression=True,  # type: bool --> we want to output the estimation of the Q-value for each action
                                         dropout_rate=args.dropout_rate,  # type: float
                                         name='tcn',  # type: str,
                                         kernel_initializer='he_normal',  # type: str,
                                         activation='linear',  # type:str,
                                         opt=args.optimizer,
                                         lr=args.learning_rate,
                                         use_batch_norm=False)
                else:
                    # recover the last model saved in the dir
                    model_filename = sorted(glob.glob(os.path.join(args.keras_model, 'model_final*.json')))[-1]
                    model = model_from_json(open(model_filename, 'r').read())
                    model.load_weights(model_filename.replace('.json', '.h5'))
                    optimizer = get_opt(opt=args.optimizer,
                                        lr=args.learning_rate,
                                        decay=0.0,
                                        )
                    model.compile(optimizer, loss='mean_squared_error')
                    LOG.debug("Using pre-saved model {}".format(os.path.basename(model_filename)))
                    model.summary(print_fn=LOG.info)

                # create the memory replay object
                # this implementation saves the timesteps
                memory = ReplayMemoryTuple(capacity=2000, timesteps=args.timesteps, num_devices=len(aps))
                # create RL agent
                agent = DDQL(env=env,
                             model=model,
                             memory=memory,
                             timesteps=args.timesteps,
                             episodes=args.episodes,
                             interaction_interval=args.interaction_interval,
                             wait_for_states=args.wait_for_states,
                             log_level=logging.DEBUG,  # log_level
                             epsilon=args.epsilon,
                             epsilon_decay=args.epsilon_decay,
                             )

                # save the initial model
                model_filename = os.path.join(args.log_dir, "model_initial_{}.json".format(identification))
                agent.save_model(model_filename)

            def stop():
                """ called by the timer to stop the agent
                """
                agent.stop_running()

            # a thread can call the following command to stop the experiment after some RUNNING TIME expiration
            if args.timeout is not None:
                Timer(args.timeout * 60, stop)

            # uncomment below to never stop on too much errors in a row
            # agent.TO_MUCH_ERROR_IN_A_ROW = None
            # execute forever, or until stop() is called
            not_detected_errors = agent.run(run_id=identification)

            # save last model
            model_filename = os.path.join(args.log_dir, "model_final_{}.json".format(identification))
            agent.save_model(model_filename)
            memory.save(filename=model_filename.replace('.json', '.p'))

            args.kill_first = True  # from the second repetition, it is necessary to kill the applications for a fresh start

            if args.qoe_model != 'test' and (args.runs_between_reboot is not None) and (rep_id % args.runs_between_reboot == 0):
                reboot(aps, stas)
                time.sleep(5 * 60)  # wait 5 minutes to the system to go up again

            if not_detected_errors:
                # if detected error, just restart all devices, and don't increment rep_count
                rep_count += 1
