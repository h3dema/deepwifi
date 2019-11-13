# Deepwifi


This repository provides the python classes (modules) necessary to implement reinforcement learning control loops with deep RL in Wi-Fi networks.
It is uses the [command ap](https://github.com/h3dema/command_ap) repository to send commands and read parameters from the APs.

This repository contains the TCN class that implements Temporal CNN classifier. We use this classifier to obtain the probabilities to perform an action in a state.
Because it is temporal, we should use a series of states s_i (i > 1) to train the TCN.


# Setup

## In the controller

```
git clone https://github.com/h3dema/deepwifi.git
cd deepwifi
```

Read the `experiments` section to see how each experiment is called.


## In the APs

Install the dependencies:
```
sudo apt-get install -y hostapd
sudo apt-get install -y ntpdate
sudo apt-get install -y python3
sudo apt-get install -y rfkill
sudo apt-get install -y git

```

Then install the `command ap` repository.

```
cd /home/winet
git clone https://github.com/h3dema/command_ap
git clone https://github.com/h3dema/server.js.git
```

Follow the instructions in `server.js` to install the packages and node.js dependencies necessary to run the server.


## In the stations

Install the dependencies:

```
sudo apt-get install -y wpasupplicant
sudo apt-get install -y rfkill
sudo apt-get install -y ntpdate
sudo apt-get install -y dnsutils   # dig
sudo apt-get install -y net-tools  # ifconfig
sudo apt-get install -y xvfb
sudo apt-get install -y nodejs npm
sudo apt-get install -y firefox
sudo apt-get install -y git
sudo apt-get install -y ubuntu-restricted-extras
```

## Conf. files

If you want to create only the configuration files, without running the experiment, call the following command:

```
cd deepwifi
python3 run_client.py --debug --dont-run --save-wpa-conf --save-hostapd-conf  --qoe-model client
```

Notice that "qoe-model" is a required parameter when running the program, thus you shouldn't forge it or you will receive an error.


# Command

```
usage: run_client.py [-h] [--model MODEL] [--timesteps TIMESTEPS]
                     [--epochs EPOCHS] [--learning-rate LEARNING_RATE]
                     [--dropout-rate DROPOUT_RATE] [--dilations DILATIONS]
                     [--kernel-size KERNEL_SIZE] [--filters FILTERS]
                     [--kill-first] [--save-wpa-conf] [--save-hostapd-conf]
                     [--dont-run] [--kill-all] [--repetitions REPETITIONS]
                     [--retries RETRIES] [--timeout TIMEOUT] [--debug]
                     [--verbose VERBOSE] [--qoe-model QOE_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         the name of the file with the sklearn model
  --timesteps TIMESTEPS
                        number of timesteps considered in the model
  --epochs EPOCHS       number of epochs
  --learning-rate LEARNING_RATE
                        model learning rate
  --dropout-rate DROPOUT_RATE
                        model dropout rate
  --dilations DILATIONS
                        number of dilation levels
  --kernel-size KERNEL_SIZE
                        kernel size
  --filters FILTERS     number of filters
  --kill-first          kill the applications in the devices first, before
                        restarting the experiment
  --save-wpa-conf       create the configuration file for the wpa_supplicant
  --save-hostapd-conf   create the configuration file for the hostapd
  --dont-run            skip the call to run the applications at the devices
  --kill-all            kill all applications stopping the experiment
  --repetitions REPETITIONS
                        number of times the experiment should run
  --retries RETRIES     number of retries before reset the device
  --timeout TIMEOUT     how much time the learning agent should run (in
                        minutes)
  --debug               log debug
  --verbose VERBOSE     define the verbose level (0=less verbose)
  --qoe-model QOE_MODEL
                        select the QoE model
```


# Experiments

1) Uses the MOS-client to obtain the reward.


```
cd deepwifi
python3 run_client.py --timeout 120 --debug  --qoe-model client --repetitions 30
```

2) Uses the MOS-ap to obtain the reward.

```
cd deepwifi
python3 run_client.py --timeout 120 --debug  --qoe-model ap --repetitions 30
```


3) Uses MOS-hybrid to obtain the reward.

```
cd deepwifi
python3 run_client.py --timeout 120 --debug  --qoe-model hybrid --repetitions 30
```

