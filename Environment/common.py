#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import logging
from collections import namedtuple


"""create a log variable for this module"""
LOG = logging.getLogger('common')
#
#
# definitions
#
#
#
AP_Config = namedtuple('AP', ['id', 'name', 'port', 'iface', 'mac', 'SSID', 'IP', 'initial_channel', 'initial_txpower'])
aps = [AP_Config(1, 'gnu-nb3', 8080, 'wlan0', 'aa:bb:cc:dd:ee:01', 'DQL1', '192.168.2.1', 1, 15),  # old mac = b0:aa:ab:ab:ac:11
       AP_Config(2, 'fenrir', 8080, 'wlan0', 'aa:bb:cc:dd:ee:02', 'DQL2', '192.168.2.2', 11, 15),  # old mac = 54:27:1e:f9:41:17
       ]

ClientsConfig = namedtuple('Sta', ['id', 'name', 'iface', 'mac', 'AP', 'SSID', 'IP', 'webpage'])
stas = [ClientsConfig(11, 'cloud', 'wlan0', '00:18:e7:7c:9c:cd', 'gnu-nb3', 'DQL1', '192.168.2.11', 'index4'),
        ClientsConfig(12, 'storm', 'wlan0', '54:e6:fc:da:ff:34', 'fenrir', 'DQL2', '192.168.2.12', 'index3')]


def exec_cmd(cmd):
    """ execute a shell command in the local computer
        @param cmd: command to be executed
    """
    with os.popen(cmd) as p:
        return p.readlines()


def exec_ssh(host, cmd):
    ssh_cmd = 'ssh winet@{}.winet.dcc.ufmg.br "{}"'.format(host.name, cmd)
    LOG.debug(ssh_cmd)
    with os.popen(ssh_cmd) as p:
        return p.read()


def kill_aps(aps, kill_file='kill.sh'):
    for ap in aps:
        cmd = "nohup bash {} 1>>start.log 2>&1 &".format(kill_file)
        LOG.debug(cmd)
        exec_ssh(ap, cmd)


def kill_stas(stas, kill_file='kill_sta.sh'):
    for sta in stas:
        cmd = "nohup bash {} 1>>start.log 2>&1 &".format(kill_file)
        LOG.debug(cmd)
        exec_ssh(sta, cmd)


def change_channel_hostapd(aps, channels):
    for ap, ch in zip(aps, channels):
        cmd = "sed -i '/channel/c\channel={}' hostapd.conf".format(ch)
        exec_ssh(ap, cmd)


TEMPLATE_AP_START = """echo "Starting hostapd"
T="`hostname`-{id}"
LOG="$OUTPUT_DIR/AP_$T.log"
sudo hostapd {config} 1>>$LOG 2>&1 &
"""


def start_hostapd(aps, ids, conf_file="hostapd.conf",):
    for ap, _id in zip(aps, ids):
        cmd = TEMPLATE_AP_START.format(**{'id': _id,
                                          'config': conf_file,
                                          })
        exec_ssh(ap, cmd)


HOSTAPD_FILE = """#This configuration file goes to {host}
interface={iface}
bssid={mac}
driver=nl80211
ignore_broadcast_ssid=0
channel={channel}

hw_mode=g
wmm_enabled=1
ieee80211n=1
ssid={ssid}

wpa=2
wpa_passphrase={passphrase}
wpa_pairwise=TKIP
rsn_pairwise=CCMP
auth_algs=1

macaddr_acl=0
ctrl_interface=/var/run/hostapd
logger_syslog=-1
logger_syslog_level=0
logger_stdout=-1
logger_stdout_level=0
"""

TEMPLATE_AP = """#!/bin/bash
#
# This scripts should run in {host}
#
if [ "$#" -ne 1 ]; then
    echo "using default format"
    id="`date +%Y%m%dZ%H%M%S`"
else
    id="$1"
fi

OUTPUT_DIR="/home/winet/logs"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR &>/dev/null
fi

echo "Altering the firewall rules"
sudo iptables --flush
sudo iptables --table nat --flush
sudo iptables --delete-chain
sudo iptables --table nat --delete-chain
# Set up IP FORWARDing and Masquerading
sudo iptables --table nat --append POSTROUTING --out-interface {default} -j MASQUERADE
sudo iptables --append FORWARD --in-interface {iface} -j ACCEPT
# Enables packet forwarding by kernel
sudo bash -c "echo 1 > /proc/sys/net/ipv4/ip_forward"

sudo ifconfig {iface} {ip} netmask 255.255.255.0
sudo ifconfig {iface} up

echo "Synchronizing time"
sudo pkill ntpd
sudo ntpdate pool.ntp.br &> /dev/null

echo "Starting hostapd"
T="`hostname`-$id"
LOG="$OUTPUT_DIR/AP_$T.log"
echo "HOSTAPD:$LOG"
sudo hostapd {config} 1>$LOG 2>&1 &

echo "Starting command_ap"
cd {cmd_ap_dir}
if [ {activate_get_set_server} ]; then
    LOG="$OUTPUT_DIR/SVR_$T.log"
    echo "GET_SET.SERVER:$LOG"
    sudo python3 -m get_set.server --collect-firefox-data 1>$LOG 2>&1 &
fi
"""

TEMPLATE_KILL_AP = """#!/bin/bash
sudo pkill hostapd
procs=`ps axf | grep nodejs | grep server.js | grep -v grep | awk '{print $1}'`
sudo kill -9 $procs 2>/dev/null

procs=`ps axf | grep python | grep get_set.server | grep -v grep | awk '{print $1}'`
sudo kill -9 $procs 2>/dev/null
"""


def save_hostapd_config(ap,
                        run_file='run.sh',
                        conf_file="hostapd.conf",
                        kill_file='kill.sh',
                        passphrase='winet3014atm',
                        activate_get_set_server=False
                        ):
    """ create hostapd.conf
        @param ap: list[ap_config] contains a list of the aps' configuration parameters
        @param run_file: the run.sh script filename
        @param conf_file: the hostapd.conf configuration file for the ap's SSID
        @param kill_file: the kill.sh script that stops all applications in the APs
    """
    conf = HOSTAPD_FILE.format(**{'ssid': ap.SSID,
                                  'mac': ap.mac,
                                  'iface': ap.iface,
                                  'ip': ap.IP,
                                  'channel': 1,
                                  'passphrase': 'winet3014atm',
                                  'host': ap.name,
                                  })
    with open(conf_file, 'w') as f:
        f.write(conf)
    # copy config to station
    cmd = 'scp {config} winet@{host}.winet.dcc.ufmg.br:{config}'.format(**{'config': conf_file,
                                                                           'host': ap.name
                                                                           })
    exec_cmd(cmd)
    LOG.debug(cmd)

    # create the file that executes the APs programs
    config = TEMPLATE_AP.format(**{'default': 'eth0',
                                   'iface': ap.iface,
                                   'ip': ap.IP,
                                   'config': conf_file,
                                   'cmd_ap_dir': '/home/winet/command_ap',
                                   'host': ap.name,
                                   'activate_get_set_server': 1 if activate_get_set_server else 0,  # 1: activate  0: deactivate
                                   })
    with open(run_file, 'w') as f:
        f.write(config)
    # copy to AP
    cmd = 'scp {config} winet@{host}.winet.dcc.ufmg.br:{config}'.format(**{'config': run_file,
                                                                           'host': ap.name,
                                                                           })
    exec_cmd(cmd)
    LOG.debug(cmd)

    # generate the script to kill all processes in the AP
    with open(kill_file, 'w') as f:
        f.write(TEMPLATE_KILL_AP)
    cmd = 'scp {kill_file} winet@{host}.winet.dcc.ufmg.br:{kill_file}'.format(**{'kill_file': kill_file,
                                                                                 'host': ap.name,
                                                                                 })
    exec_cmd(cmd)
    LOG.debug(cmd)

    # mark script as executable
    exec_ssh(ap, "chmod 755 {}".format(run_file))
    exec_ssh(ap, "chmod 755 {}".format(kill_file))
    return conf_file, run_file, kill_file


WPA_FILE = """# This configuration file run in {host}
ctrl_interface=/var/run/wpa_supplicant
network={{
    ssid="{ssid}"
    scan_ssid=1
    key_mgmt=WPA-PSK
    psk="{passphrase}"
}}
"""

TEMPLATE_STATION = """#!/bin/bash
#
# This configuration file belongs to {host}
#
if [ "$#" -ne 1 ]; then
    echo "using default format"
    id="`date +%Y%m%dZ%H%M%S`"
else
    id="$1"
fi

OUTPUT_DIR="/home/winet/logs"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR &>/dev/null
fi

# change server.js location of 'save.db':
# cd ~/server.js/server
# sed -i 's/tmp\/save.db/home\/winet\/save.db/g' server.js

# ubuntu 14
sudo nmcli nm wifi off &> /dev/null
# ubuntu 16/18
sudo nmcli radio wifi off &> /dev/null

sudo pkill wpa_supplicant
sudo rm -fr /var/run/wpa_supplicant/{iface}

sudo /usr/sbin/rfkill unblock wifi
sleep 2

sudo ifconfig {iface} {ip} netmask 255.255.255.0
sudo ifconfig {iface} up

sudo route del -net 0 dev eth0
sudo route add -net 0 netmask 0 gw {gw} metric 0
sudo route del -net 0 netmask 0 gw 150.164.10.1 dev eth1
# set route to dash server via wireless
sudo route add 150.164.10.51/32 gw {gw}

sudo pkill ntpd
sudo ntpdate pool.ntp.br &> /dev/null

T="`hostname`-$id"
sudo wpa_supplicant -Dnl80211 -i {iface} -c {wpafile} 1>"$OUTPUT_DIR/sta-$T.log" 2>&1 &

nohup nodejs /home/winet/server.js/server/server.js &> /dev/null &

#
# wait to connect
#
while [ "`iwconfig {iface} | grep {ssid} | wc -l`" -eq "0" ]; do
    sleep 1
done

# run firefox
bash {restart_sh} $id &>/dev/null &
"""

TEMPLATE_FFOX = """#!/bin/bash
BROWSER="{browser}"
if [ "$#" -ne 1 ]; then
    echo "using default format"
    id="`date +%Y%m%dZ%H%M%S`"
else
    id="$1"
fi
T="`hostname`-$id"

OUTPUT_DIR="/home/winet/logs"
mkdir -p $OUTPUT_DIR &>/dev/null

#
# criar o display virtual
#
# Xvfb :1 -screen 0 1920x1080x24+32 -fbdir /var/tmp &
# export DISPLAY=:1
#
# # RODAR FIREFOX
#
# DISPLAY=:0  nohup /usr/bin/firefox --private-window {site} 1>>$OUTPUT_DIR/ffox-$T.log 2>&1 &

# kill all browsers
procs=`ps axf | grep '{browser}.*html' | grep -v grep | awk '{print $1}'`
sudo kill -9 $procs 2>/dev/null

procs=`ps axf | grep 'firefox.*html' | grep -v grep | awk '{print $1}'`
sudo kill -9 $procs 2>/dev/null

if [ "$BROWSER" == "opera" ]; then
    DISPLAY=:0 {browser_path} {site} 1>>$OUTPUT_DIR/ffox-$T.log 2>&1 &
else
    # DISPLAY=:1  nohup /usr/bin/firefox --headless --private-window {site} 1>$OUTPUT_DIR/ffox-$T.log 2>&1 &
    DISPLAY=:0 /usr/bin/firefox --private-window http://$SITE/index3.html 1>>$OUTPUT_DIR/ffox-$T.log 2>&1 &
fi
"""

RESTART_FFOX = """#!/bin/bash
#
if [ "$#" -ne 1 ]; then
    echo "using default format"
    id="`date +%Y%m%dZ%H%M%S`"
else
    id="$1"
fi


while [ 1 ]; do
    bash {ffox_file} $id &>/dev/null &
    sleep {restart}m
done
"""

# SITE_DASH = 'http://dash.winet.dcc.ufmg.br'
SITE_DASH = 'http://150.164.10.51'

TEMPLATE_KILL_STA = """#!/bin/bash
sudo pkill wpa_supplicant
sudo pkill Xvfb

kill -9 `ps axf | grep 'restart.sh' | grep -v grep | awk '{print $1}'` &>/dev/null

kill -9 `ps axf | grep 'watch.*ffox' | grep -v grep | awk '{print $1}'` &>/dev/null
sudo pkill ffox.sh

procs=`ps axf | grep nodejs | grep server.js | grep -v grep | awk '{print $1}'`
sudo kill -9 $procs 2>/dev/null
sudo pkill nodejs

procs=`ps axf | grep 'firefox.*html' | grep -v grep | awk '{print $1}'`
sudo kill -9 $procs 2>/dev/null
sudo pkill firefox

procs=`ps axf | grep 'opera.*html' | grep -v grep | awk '{print $1}'`
sudo kill -9 $procs 2>/dev/null
sudo pkill opera
"""


def save_wpa_config(sta, ap,
                    run_file='run_sta.sh',
                    config_file="wpa_supplicant.conf",
                    kill_file='kill_sta.sh',
                    restart_file='restart.sh',
                    ffox_file='ffox.sh',
                    restart_ffox=5,
                    browser='opera',
                    passphrase='winet3014atm'):
    """ create the wpa_supplicant.conf file for the designated sta
        @param ap: list[sta_config] contains a list of each station's configuration parameters
        @param ap: list[ap_config] contains a list of each ap's configuration parameters
        @param run_file: the run.sh script filename
        @param conf_file: the wpa_supplicant.conf the create the connection to the correct AP
        @param kill_file: the kill.sh script that stops all applications in the stations
        @return:  the wpa_supplicant.conf name
    """
    #
    #
    #
    # create wpa_supplicant conf
    wpa = WPA_FILE.format(**{'ssid': sta.SSID,
                             'passphrase': passphrase,
                             'host': sta.name,
                             })
    with open(config_file, 'w') as f:
        f.write(wpa)
    # copy config to station
    cmd = 'scp {config} winet@{host}.winet.dcc.ufmg.br:{config}'.format(**{'config': config_file,
                                                                           'host': sta.name,
                                                                           })
    exec_cmd(cmd)
    LOG.debug(cmd)

    #
    #
    #
    # create the script that prepares the station and runs apps
    # this scripts calls 'restart_file', which runs firefox
    #
    config = TEMPLATE_STATION.format(**{'iface': sta.iface,
                                        'ip': sta.IP,
                                        'gw': ap.IP,
                                        'ssid': sta.SSID,
                                        'wpafile': config_file,
                                        'host': sta.name,
                                        'restart_sh': restart_file,
                                        })

    with open(run_file, 'w') as f:
        f.write(config)
    # copy to the station
    cmd = 'scp {config} winet@{host}.winet.dcc.ufmg.br:{config}'.format(**{'config': run_file,
                                                                           'host': sta.name,
                                                                           })
    exec_cmd(cmd)
    LOG.debug(cmd)

    #
    #
    #
    # creates the script that restarts firefox from time to time
    # notice that it calls "ffox_file" script
    #
    config = RESTART_FFOX.format(**{'restart': restart_ffox,  # restarts firefox every x minutes
                                    'ffox_file': ffox_file,
                                    })

    with open(restart_file, 'w') as f:
        f.write(config)
    # copy to the station
    cmd = 'scp {config} winet@{host}.winet.dcc.ufmg.br:{config}'.format(**{'config': run_file,
                                                                           'host': sta.name,
                                                                           })
    exec_cmd(cmd)
    LOG.debug(cmd)

    select_browser = {'opera': ['opera',
                                '/usr/lib/x86_64-linux-gnu/opera/opera --private'],
                      'firefox': ['firefox',
                                  '/usr/bin/firefox --private-window'
                                  ],
                      }
    #
    #
    #
    # creates the script that runs the firefox
    ffox = TEMPLATE_FFOX.format(**{'site': "{}/{}.html".format(SITE_DASH, sta.webpage),
                                   'browser': select_browser[browser][0],
                                   'browser_cmd': select_browser[browser][1],
                                   })
    with open(ffox_file, 'w') as f:
        f.write(ffox)
    cmd = 'scp {config} winet@{host}.winet.dcc.ufmg.br:{config}'.format(**{'config': ffox_file,
                                                                           'host': sta.name,
                                                                           })
    exec_cmd(cmd)
    LOG.debug(cmd)

    # kill sta apps
    with open(kill_file, 'w') as f:
        f.write(TEMPLATE_KILL_STA)
    # copy to the station
    cmd = 'scp {kill_file} winet@{host}.winet.dcc.ufmg.br:{kill_file}'.format(**{'kill_file': kill_file,
                                                                                 'host': sta.name,
                                                                                 })
    exec_cmd(cmd)
    LOG.debug(cmd)

    #
    #
    #
    # mark script as executable
    exec_ssh(sta, "chmod 755 {}".format(run_file))
    exec_ssh(sta, "chmod 755 {}".format(ffox_file))
    exec_ssh(sta, "chmod 755 {}".format(restart_file))
    exec_ssh(sta, "chmod 755 {}".format(kill_file))
    return config_file, run_file, kill_file


def run_station(sta, _id='', run_file='run_sta.sh'):
    """call the run.sh script to run the applications in the STA"""
    cmd = "nohup bash {} {} 1>>start.log 2>&1".format(run_file, _id)
    LOG.debug(cmd)
    ret = exec_ssh(sta, cmd)
    return ret


def run_hostapd(ap, _id='', run_file='run.sh'):
    """ calls the AP, and starts the hostapd
    """
    cmd = "nohup bash {} {} 1>>start.log 2>&1".format(run_file, _id)
    LOG.debug(cmd)
    ret = exec_ssh(ap, cmd)
    return ret


def ap_is_running(ap):
    """ calls the AP, and verifies if hostapd is running
    """
    cmd = "ps axf | grep hostapd | grep -v grep"
    loaded = exec_ssh(ap, cmd)
    loaded = len(loaded) > 0
    if not loaded:
        return False

    cmd = "iw {} info| grep channel | grep -v grep | wc -l".format(ap.iface)
    connected = exec_ssh(ap, cmd).strip()
    try:
        connected = int(connected) > 0
    except ValueError:
        connected = False
    LOG.debug("{} is up? {}".format(ap.iface, connected))
    if not connected:
        return False

    cmd = "ps axf | grep python | grep get_set | wc -l"
    server = exec_ssh(ap, cmd).strip()
    try:
        server = int(server) > 0
    except ValueError:
        server = False
    LOG.debug("get_set server {}".format(server))

    up = loaded and connected and server
    LOG.debug("AP {} is running? {}".format(ap.name, up))
    return up


def sta_is_running(sta, browser='opera'):
    """ calls the STA, and verifies if wpa_supplicant is running
    """
    # check if wpa_supplicant is running
    cmd = "ps axf | grep wpa_supplicant | grep -v grep"
    wpa = len(exec_ssh(sta, cmd)) > 0
    LOG.debug("STA: wpa_supplicant is{}running in {}".format(" " if wpa else " *NOT* ", sta.name))
    if not wpa:
        # don't need to test anymore because wpa_supplication is not running
        return False

    # check if it is connected
    cmd = "iwconfig {} | grep ESSID | grep {}".format(sta.iface, sta.SSID)
    connected = len(exec_ssh(sta, cmd)) > 0
    LOG.debug("STA: {} is {}".format(sta.name, "connected" if connected else "disconnected"))
    if not connected:
        # don't need to test anymore because the station is not connected
        return False

    # check if firefox is running
    cmd = {'firefox': "ps axf | grep 'firefox.*html' | grep -v grep",
           'opera': "ps axf | grep 'opera.*html' | grep -v grep",
           }
    browser_is_running = len(exec_ssh(sta, cmd[browser])) > 0
    LOG.debug("STA {}: browser is{}running".format(sta.name, " " if browser_is_running else " NOT "))

    ret = wpa and connected and browser_is_running
    return ret


# **********************************
#
# config the APs and stations
#
# **********************************
def conf_stas(aps, stas, restart_ffox, browser):
    # config STAs
    LOG.info("Configuring STAs: {}".format([sta.name for sta in stas]))
    for sta in stas:
        ap = [ap for ap in aps if ap.SSID == sta.SSID][0]
        save_wpa_config(sta, ap, restart_ffox=restart_ffox, browser=browser)


def conf_aps(aps):
    # config APs
    LOG.info("Configuring APs: {}".format([ap.name for ap in aps]))
    for ap in aps:
        save_hostapd_config(ap)


def start_devices(aps, stas,
                  max_retries=3,
                  sleep_interval=10,
                  _id='',
                  kill_ap='kill.sh',
                  kill_sta='kill_sta.sh',
                  browser='opera'):
    """
        @param max_retries: number of retries, before reseting the device
        @param sleep_interval: number of seconds the execution is suspended before retrying
    """
    retry = dict()

    # **********************************
    #
    # run the apps
    #
    # **********************************
    something_to_start = True

    for ap in aps:
        retry[ap] = 1
    for sta in stas:
        retry[sta] = 1
    while something_to_start:
        LOG.debug("Trying to start devices")
        # 1- start hostapd
        # 2- check if the hostapd is running
        #    if not, go to step 1
        # ap_is_running(ap)
        num_aps_running = 0  # count the number of APs running
        for ap in aps:
            if not ap_is_running(ap):
                retry[ap] -= 1
                if retry[ap] <= 0:
                    LOG.debug("Trying to start AP {}".format(ap.name))
                    exec_ssh(ap, 'nohup bash {}'.format(kill_ap))
                    run_hostapd(ap, _id)
                    # print("\n\n **  run_hostapd(ap) is not running yet  **")
                    retry[ap] = max_retries
                else:
                    # wait some time before checking again
                    time.sleep(sleep_interval)
            else:
                # increment the APs running
                num_aps_running += 1

        # start the clients
        # 1- run wpa_supplicant
        # 2- check if sta is running
        num_stas_running = 0  # count the number of stations running
        for sta in stas:
            if not sta_is_running(sta, browser=browser):
                # decrement retry
                retry[sta] -= 1
                # if retry is zero, run_wpa_supplicant again
                if retry[sta] <= 0:
                    LOG.debug("Trying to start STA {}".format(sta.name))
                    exec_ssh(sta, 'nohup bash {}'.format(kill_sta))
                    run_station(sta, _id)
                    retry[sta] = max_retries
                else:
                    time.sleep(sleep_interval)
            else:
                # increment the APs running
                num_stas_running += 1
        # check if there is something to start
        something_to_start = num_aps_running != len(aps) or num_stas_running != len(stas)


def reboot_devices(devices):
    """ reboot the devices to get a clean slate
        @param devices: contains the hostname of each device
        @type devices: list(namedtuple)
    """
    for d in devices:
        exec_ssh(d, 'sudo reboot')


def run_nodejs(dir_='/home/h3dema/Devel/server.js'):
    """ create server to collect browser data
        @param dir_: directory where the nodejs program is installed (from https://github.com/h3dema/server.js.git)
    """
    cmd = "nohup nodejs {} &> /dev/null &".format(os.path.join(dir_, "server/server.js"))
    exec_cmd(cmd)


def is_runnning_get_set_server():
    cmd = "ps axf | grep get_set.server | grep -v grep | wc -l"
    ret = exec_cmd(cmd)
    try:
        ret = int(ret[0].strip()) > 0
    except (ValueError, IndexError):
        ret = False
    return ret


def run_get_set_server(_id, dir_='/home/h3dema/Devel/command_ap', log_dir='/home/h3dema/Devel/deepwifi/logs'):
    filename = os.path.join(log_dir, "SVR_{}.log".format(_id))
    LOG.debug("Starting get_set server id={}".format(_id))
    cmd = "cd {};sudo python3 -m get_set.server --collect-firefox-data 1>>{} 2>&1 &".format(dir_, filename)
    LOG.debug(cmd)
    exec_cmd(cmd)


def kill_get_set_server():
    cmd = "ids=`ps axf | grep get_set.server | grep -v grep | awk '{print $1}'`;sudo kill -9 $ids 2>/dev/null"
    exec_cmd(cmd)
