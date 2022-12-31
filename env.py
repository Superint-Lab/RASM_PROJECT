import copy
import math
import random
from statistics import mean

import simpy
import torch

from base_station import Base_station
from server import Server
from user import User
from utils import *


class Env(object):
    def __init__(self, env, env_height=6, env_width=6, n_users=50, n_servers=20, n_bs=36,
                 SIMULATION_DURATION=60, next_step=60):
        self.env = env
        self.SIMULATION_DURATION = SIMULATION_DURATION
        self.simulation_stop = next_step
        self.simpy_delta = 60
        self.height = env_height
        self.width = env_width
        self.n_users = n_users
        self.n_servers = n_servers
        self.n_base_stations = n_bs
        self.total_migrations = 0
        self.rejected_migrations = 0
        self.backhaul_bandwidth = 1_000_000_000
        self.virtual_machines = {}

        self.all_bs = {}
        self.all_ue = {}
        self.all_servers = {}

        self._deploy_eNodeB()
        self._create_server()
        self._deploy_server()
        self._create_user()

        self.epsde_man_delay = {}
        self.epsde_proc_delay = {}

    def _generate_bs_cn_dist(self, min_dist=0.5, max_dist=20):
        #  Returns distance in Kilometers
        #  The closest base station is assumed to be 500 Meters away from the Central office
        #  The farthest is location 20 Kilometers away
        distance = random.uniform(min_dist, max_dist)
        # print("The distance to base station is: ", distance)
        return distance

    def _instantiate_bs(self):
        if len(self.all_bs) > 0:
            # create dict key for new BS object
            # based on position index in dictionary
            bs_id = len(self.all_bs)
            coords = convert_scalar_to_pixel_coord(bs_id)
            base_station = Base_station(self.env, bs_id=bs_id, network=self,
                                        location=bs_id, coords=coords)
        else:
            bs_id = 0
            coords = convert_scalar_to_pixel_coord(bs_id)
            base_station = Base_station(self.env, bs_id=bs_id, network=self,
                                        location=bs_id, coords=coords)

        return base_station, bs_id

    def _deploy_eNodeB(self):
        # Instantiates and deploys base stations in the network environment
        # The number of deployed base stations depends on self.n_base_stations = 36
        for i in range(self.n_base_stations):
            bs_cn_dist = self._generate_bs_cn_dist()
            base_station, new_idx_key = self._instantiate_bs()
            base_station.bs_cn_distance = bs_cn_dist
            self.all_bs[new_idx_key] = base_station

    # Instantiates MEC servers and adds to pool
    def _create_server(self):
        server_ids = [i for i in range(self.n_servers)]
        for server_id in server_ids:
            server = Server(env=self.env, server_id=server_id, network=self, base_station=None)
            self.all_servers[server_id] = server

        print(self.all_servers)

    # Randomly attaches MEC servers to base stations
    def _deploy_server(self):
        # Attaches created servers to base stations
        selected_base_stations = random.sample(list(self.all_bs), k=self.n_servers)
        for bs, server in zip(selected_base_stations, self.all_servers):
            self.all_bs[bs].server = self.all_servers[server]  # associate server with BS
            self.all_servers[server].base_station = self.all_bs[bs].bs_id  # save associated bs in the server

    # Instantiates users
    def _create_user(self):
        ue_ids = [i for i in range(self.n_users)]
        for ue_id in ue_ids:
            bs_id = random.sample(list(self.all_bs), k=1)[0]
            server_ID = random.sample(list(self.all_servers), k=1)[0]
            user = User(self.env, self, ue_id, server=self.all_servers[server_ID], cell_id=self.all_bs[bs_id])
            print("User ", ue_id)
            # Generate coordinates to identify precise user location within service area
            print("User associated with: ", server_ID)
            user.sample_loc_coords()
            # Set the target property to True on User whose ID is 0
            if user.uid == 0:
                user.target = True
            # Adds current user to list of available users.
            self.all_ue[len(self.all_ue)] = user
            # Connects UE to target base station
            self.all_bs[bs_id].connect_user(user)

    # Moves all users to new base stations
    def user_mobility(self):
        # Calls the walk method on user to relocate the target UE to new destination
        for key, user in self.all_ue.items():
            # if user.is_target_user():
            user.walk()

    def simulate_task_generation(self):
        # Triggers UEs to generate tasks, send to ECs through BSs for processing
        while self.env.peek() < self.SIMULATION_DURATION:
            print("Still running...!")
            # Checks if task generation pause time is reached!
            if self.env.peek() <= self.simulation_stop:
                self.env.step()
            else:
                break

        # Adjust the task generations stop time
        # Users generate and schedule tasks until simulation_stop time
        # The tasks are sent to BSs and MEC servers for processing
        self.simulation_stop += self.simpy_delta

    # Migrates service instances belonging to other users
    def other_users_service_mobility(self):
        for key, user in self.all_ue.items():
            # Randomly select server ID (identified by key in dict) where VM should be migrated
            # Do not migrate the service for the target
            if not user.is_target_user():
                dst_server_id = random.choice(list(self.all_servers.keys()))
                # select the target server identified by key in dict
                self.all_servers[dst_server_id].migrate(user)

    def check_service_delay(self, target_user):
        self.experience_quality(target_user)

    def step(self, action):
        # Action = destination server ID
        target_user = find_target_user(self.all_ue)
        dest_server = self.all_servers[int(action)]
        vm_current_server = target_user.vm.edge_server.server_id
        if dest_server.server_id != vm_current_server:
            return self.attempt_migration(target_user, dest_server)
        else:
            return self.no_migration(target_user)

    def attempt_migration(self, target_user, dest_server):
        # check if destination server is not same as source
        vm_current_server = target_user.server
        if vm_current_server.server_id != dest_server.server_id:
            migration_cost = self._migration_cost(target_user, dest_server)
            # move services for other users
            self.other_users_service_mobility()
            # |************* clear queues in all server *************|?
            self.simulate_task_generation()
            # move service for target user
            dest_server.migrate(target_user)
            observation = self.get_state()
            # service_delay, delay_MAN, delay_RAN, processing_delay
            qoe, perf_metrics = self.gather_metrics(target_user)
            reward = self.compute_reward(qoe, migration_cost, metrics=perf_metrics)
            reward = torch.tensor([reward])
            return observation, reward, qoe, migration_cost, perf_metrics
        else:
            return self.no_migration(target_user)

    def no_migration(self, target_user):
        self.simulate_task_generation()
        print("Checking generated tasks.............")
        migration_cost = 0.0
        qoe, perf_metrics = self.gather_metrics(target_user)
        observation = self.get_state()
        reward = self.compute_reward(qoe, migration_cost, metrics=perf_metrics)
        reward = torch.tensor([reward])

        return observation, reward, qoe, migration_cost, perf_metrics

    def gather_metrics(self, target_user):
        # Holds delays values in current time step
        perf_metrics = {}
        qoe, d_man, d_ran, d_proc = self.experience_quality(target_user)
        # save metrics to dictionary
        perf_metrics['d_ran'] = d_ran
        perf_metrics['d_man'] = d_man
        perf_metrics['d_proc'] = d_proc

        return qoe, perf_metrics

    def compute_vm_trans_delay(self, vm_size):
        bits_factor = 8  # Number of bits in one byte
        # gigabits = 1_000_000_000  # 1 Gigabit per second
        megabits = 1_000_000  # 1 megabit per second
        data_in_bits = vm_size * bits_factor * megabits
        trans_delay = data_in_bits / self.backhaul_bandwidth

        print("bandwidth: ", self.backhaul_bandwidth, " Data size (Mbits):  ",
              data_in_bits, "Transmission delay(ms): ", trans_delay * 1000)

        return trans_delay  # * conversion_units['millisec']

    def compute_prop_delay(self, src_bs_id, dest_bs_id):
        # ****** Returns time in milliseconds ******
        speed_of_light = 300_000_000  # In meters per second
        src_bs = self.all_bs[src_bs_id]
        dest_bs = self.all_bs[dest_bs_id]
        # Current location of the virtual machine, base station
        src_coord_x, src_coord_y = src_bs.coords
        dst_coord_x, dst_coord_y = dest_bs.coords
        hop_dist = self.hop_distance(x1=src_coord_x, x2=dst_coord_x, y1=src_coord_y, y2=dst_coord_y)
        print("Hop distance is: ", hop_dist)
        d_propagation = 0.02 * hop_dist
        return d_propagation

    # Computes cost of migrating service VM between MEC servers
    def _migration_cost(self, target_user, dest_base_station):
        # Virtual machine of the target mobile user
        vm_size = target_user.vm.vm_size
        print("VM size is: ", vm_size)
        # Current location of the virtual machine, base station
        src_base_station_id = target_user.server.base_station
        dest_base_station_id = dest_base_station.base_station
        src_coord_x, src_coord_y = self.all_bs[int(src_base_station_id)].coords
        dst_coord_x, dst_coord_y = self.all_bs[int(dest_base_station_id)].coords
        hop_dist = self.hop_distance(x1=src_coord_x, x2=dst_coord_x, y1=src_coord_y, y2=dst_coord_y)
        print("Hop distance is: ", hop_dist)
        # Transmission delay in seconds
        trans_delay = self.compute_vm_trans_delay(vm_size=vm_size) * hop_dist
        # Calculate propagation time in milliseconds
        propagation_delay = self.compute_prop_delay(src_base_station_id, dest_base_station_id)
        # Total migration time
        migration_time = propagation_delay + trans_delay

        print("The migration cost is: ", migration_time)

        return migration_time

    # ================================== Quality of Experience ==================================
    def experience_quality(self, user):
        # Virtual machine of the target mobile user
        print("Checking this information ...")
        vm = user.vm
        service_server = vm.edge_server
        processing_delay = service_server.get_empirical_proc_delay()
        delay_RAN = user.base_station.get_RAN_delay()
        delay_MAN = user.base_station.compute_MAN_delay()
        network_delay = (delay_MAN + delay_RAN)

        service_delay = processing_delay + network_delay
        print("*** BS # of Users: {} | RAN Delay: {:.5f} ***".format(len(user.base_station.users), delay_RAN))
        print(
            "*** Server # of Users: {} | MAN Delay is: {} ***".format(len(service_server.virtual_machines), delay_MAN))
        print("*** Processing delay: {} ***".format(processing_delay))
        print("*** Theo. Proc delay: {} ***".format(service_server.get_theoretical_proc_delay()[0]))
        print("*** Server util     : {}".format(service_server.get_theoretical_proc_delay()[1]))
        print("*** Service delay: {} ***".format(network_delay + processing_delay))

        return service_delay, delay_MAN, delay_RAN, processing_delay

    # ================================== End of Quality of Experience ==================================

    def compute_reward(self, service_delay, mig_cost, metrics=None):
        delay_upper_bound = 0.35  # milliseconds = 0.35 seconds
        qoe = 1 - (service_delay / delay_upper_bound)
        reward = qoe - mig_cost

        return reward

    def reset(self):
        self.env = simpy.Environment()
        self.all_bs.clear()
        self.all_ue.clear()
        self.all_servers.clear()

        self._deploy_eNodeB()
        self._create_server()
        self._deploy_server()
        self._create_user()

        self.epsde_man_delay.clear()
        self.epsde_proc_delay.clear()

        self.total_migrations = 0
        self.rejected_migrations = 0

        return self.get_state()

    # The state is used as input to DQN
    def get_state(self):
        target_user = find_target_user(self.all_ue)
        ue_location = target_user.base_station.bs_id
        vm_location = target_user.vm.edge_server.server_id
        n_users_source = target_user.vm.edge_server.get_num_users()

        server_loads = []
        for i in range(self.n_servers):
            server_loads.append(self.all_servers[i].get_num_users())

        # server_res = target_user.vm.edge_server.get_resource_capacity()
        distance = math.fabs(ue_location - vm_location)
        # state = [distance, server_res, n_users_source]
        state = [distance, n_users_source, *server_loads]
        state = torch.tensor(state)
        print("State is: ", state)
        return state

    # adds neighboring servers to list
    # Best on resource capacity and distance to UE
    def identify_neighbouring_servers(self, user):
        neighbors_with_server = []
        neighbours = user.fetch_current_base_station_neighbours()
        for neighbor in neighbours:
            if self.all_bs[neighbor].has_server():
                neighbors_with_server.append(self.all_bs[neighbor].server)

        print("The vm is located at: ", user.vm.edge_server)
        return neighbors_with_server

    # Sorts servers based on current resource capacity
    def identify_highest_res_neighbor(self, neighbours_with_server):
        threshold = 30
        sorted_neighbors = []
        if neighbours_with_server:
            for neighbour in neighbours_with_server:
                if len(sorted_neighbors) > 0:
                    if threshold <= sorted_neighbors[-1].get_resource_capacity() < neighbour.get_resource_capacity():
                        temp = sorted_neighbors[-1]
                        sorted_neighbors.append(neighbour)
                        sorted_neighbors.append(temp)
                else:
                    sorted_neighbors.append(neighbour)
        else:
            print("The list is Empty")

        print("Highest res neighbours: ", sorted_neighbors)
        return sorted_neighbors

    def get_server_users_count(self):
        num_users = []
        for key, server in self.all_servers.items():
            num_users.append(server.get_num_users())

        return num_users

    def get_resource_capacity(self, neighbors):
        # Resource capacity of all candidate servers
        available_capacity = []
        max_capacity = 100  # 100%
        if neighbors:
            for server in neighbors:
                s_util = max_capacity - (server.get_theoretical_proc_delay()[1] * max_capacity)
                available_capacity.append(s_util)

        return available_capacity

    def get_dists_to_servers(self):
        distances = []
        target_user = find_target_user(self.all_ue)
        ue_bs_id = target_user.base_station.bs_id
        for key, server in self.all_servers.items():
            dist = math.fabs(server.base_station - ue_bs_id)
            distances.append(dist)

        return distances

    def hop_distance(self, x1, y1, x2, y2):
        return math.fabs(x1 // BLOCK_SIZE - x2 // BLOCK_SIZE) + math.fabs(y1 // BLOCK_SIZE - y2 // BLOCK_SIZE)
