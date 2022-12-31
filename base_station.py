import math
import random
from statistics import mean

import numpy as np
import simpy
from utils import find_target_user


class Base_station:
    def __init__(self, env, bs_id, network, location, bs_cn_dist=1, coords=(0, 0)):
        self.bs_id = bs_id
        self.location = location
        self.coords = coords
        self.server = None
        self.users = {}
        self.env = env
        self.in_store = simpy.Store(self.env)
        self.out_store = simpy.Store(self.env)
        self.network = network
        self.bs_cn_distance = bs_cn_dist
        self.wired_link_bandwidth = 1_000_000_000  # 1 Gbits
        self.wireless_bw = 20 * 1e6  # 20 megahertz
        self.total_res_blocks = 50
        self.coverage_radius = 500  # meters
        self.spectral_density = -174  # dBM/hz
        self.power_ctrl = 0.25
        self.path_loss = 3.75
        self.tx_power = 200  # milliWatt
        self.server = None
        self.service_times = []
        self.waits = []
        self.departs_t = []
        self.service_times = []
        self.inter_arrival_times = []
        self.arrivals = 0
        self.last_arrival = 0.0
        self.arrivals_t = []
        self.action = self.env.process(self.process())

    def __repr__(self):
        return "bs_id: {} | location: {} | coords: {} | Server: {}". \
            format(self.bs_id, self.location, self.coords, self.server)

    def get_distance_to_core_network(self):
        return self.bs_cn_distance

    def has_server(self):
        return False if self.server is None else True

    def connect_user(self, user):
        # Associates UE to current Base station
        user_id = user.uid
        self.users[user_id] = user

    def input(self, task):
        curr_arrival_t = self.env.now
        inter_arrival = curr_arrival_t - self.last_arrival
        self.inter_arrival_times.append(inter_arrival)
        self.last_arrival = curr_arrival_t
        task.t_server_arrival = curr_arrival_t
        self.arrivals_t.append(curr_arrival_t)
        self.in_store.put(task)
        self.arrivals += 1

    def process(self):
        while True:
            task = (yield self.in_store.get())
            # transmission_time = task.download_data_size
            t_depart = self.env.now
            waiting_time = t_depart - task.t_server_arrival
            self.waits.append(waiting_time)
            # processing delay
            proc_delay = task.upload_data_size / self.wired_link_bandwidth
            # print("Processing delay for base station is: ", proc_delay)
            # record processing delay for the processed task
            self.departs_t.append(proc_delay)
            # transmission time
            delay_time = random.expovariate(1. / proc_delay)
            # print("Base station service time is: ", delay_time)
            cpu_entry = self.env.now
            yield self.env.timeout(delay_time)
            service_time = self.env.now - cpu_entry
            print("Service time at base station is: ", service_time)
            self.service_times.append(service_time)
            self._send_result(task)

    def process_output_queue(self):
        print()

    def _send_result(self, task):
        dst_info = int(task.dst.split(':')[0])
        # check if server is attached to current base station
        if self.server is not None:
            # check if server in current location has same ID as destination server
            # In task dest field.
            if self.server.server_id == int(task.dst.split(':')[1]):
                self.server.put(task)
        else:
            dest_bs = self.network.all_bs[dst_info]
            dest_bs.server.put(task)

    def disconnect_user(self, user):
        del self.users[user.uid]

    def compute_MAN_delay(self):
        # =========================================================================
        # ESTIMATES COMMUNICATION DELAY FOR UPLOADING DATA
        # EMPLOYS QUEUEING THEORY
        # =========================================================================

        # print("bs_id: {} | {} users connected to this base station: ".format(self.bs_id, len(self.users)))
        avg_service_time = mean(self.service_times) if len(self.service_times) > 1 else 0
        avg_waiting_time = mean(self.waits) if len(self.waits) > 1 else 0

        avg_man_delay = (avg_waiting_time + avg_service_time)
        # if self.arrivals > 1:
        #     mu = 1. / mean(self.service_times)
        #     lambda_ = 1.0 / mean(self.inter_arrival_times)
        #     n_arrivals = self.arrivals
        #     t_waiting = sum(self.waits)
        #     t_delay = 1.0 / (mu - lambda_)
        #     # return n_arrivals, mu, lambda_, t_waiting, t_delay
        #     # return t_delay
        #     # print("Theoretical delay: ", t_delay)
        # else:
        #     # return 0.0
        #     pass

        return avg_man_delay

    def get_target_ue(self):
        return find_target_user(self.network.all_ue)

    def compute_wavelength(self):
        light_speed_v = 3 * np.power(10, 8)  # meters per second
        frequency = self.wireless_bw
        wavelength = light_speed_v / frequency

        return wavelength

    def euclid_distance(self):
        target_ue = self.get_target_ue()
        bs_x = bs_y = 0
        ue_x = target_ue.x_coord
        ue_y = target_ue.y_coord

        first_term = np.power((ue_x - bs_x), 2)
        second_term = np.power((ue_y - bs_y), 2)
        distance = np.sqrt(first_term + second_term)

        # maps distance to meters
        distance = self.eucl_to_metric(distance)

        if distance <= 0:
            distance = 10

        return distance

    def eucl_to_metric(self, euclid_dist):
        trans_range = 500  # meters
        refer_dist = np.sqrt(np.power((5-0), 2) + np.power((5-0), 2))
        meters = (euclid_dist * trans_range) / refer_dist

        return meters

    def channel_gain(self, wave_length, distance):
        inner_term = math.pow(wave_length, 2) / (math.pow(4 * math.pi, 2) * distance)
        gain = 10 * np.log10(inner_term) - (20 * np.log10(distance))

        return gain

    def snr(self):
        wave_length = self.compute_wavelength()
        distance = self.euclid_distance()
        channel_gain = self.channel_gain(wave_length, distance)
        snr = self.tx_power * channel_gain \
              * np.power(distance, self.path_loss * (self.power_ctrl - 1))\
              / self.spectral_density
        return snr

    def channel_capacity(self):
        snr = self.snr()
        res_alloc = np.floor(self.total_res_blocks / len(self.users))
        c = self.wireless_bw * res_alloc * (10 * np.log2(1 + snr))

        return c

    def get_RAN_delay(self):
        # =========================================================================
        # CALCULATES UPLOAD DELAY IN THE RADIO ACCESS NETWORK
        # SHANNON CAPACITY IS USED TO COMPUTE THE MAX ACHIEVABLE DATA RATE
        # =========================================================================
        user = self.get_target_ue()
        data_rate_achievable = self.channel_capacity()
        data_size = user.get_upload_data_size()
        print("Data size is: ", data_size, "Data rate is: ", data_rate_achievable)
        avg_trans_delay = data_size / data_rate_achievable
        return avg_trans_delay

