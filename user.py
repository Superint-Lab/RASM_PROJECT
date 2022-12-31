import random

import numpy as np

from application import Application
from application_types import APPLICATION_TYPES
from task import Task
from utils import csv_writer

random.seed(123)
class User:
    def __init__(self, env, network, uid, server, cell_id):
        self.env = env
        self.network_env = network
        self.uid = uid
        self.target = False
        self.base_station = cell_id
        self.x_coord = 0
        self.y_coord = 0
        self.apps = {}
        self.server = server
        self.generated_tasks = []
        self.vm = None
        self.response_time = []
        self.network_delay = []
        self._deploy_app()
        self.request_vm_instantiation()
        self.env.process(self.generate_tasks())

    def __repr__(self):
        return "user id: {} | Coords: {} | target: {} | BS_ID: {}" \
            .format(self.uid, (self.x_coord, self.y_coord), self.target,
                    self.base_station.bs_id)

    # Instantiates an application on UE
    def _deploy_app(self):
        app_name = random.choice(list(APPLICATION_TYPES.keys()))
        upload_data_size = 5  # random.randint(2, 5)  # Megabytes of data
        # task_length = 100  # random.randint(50, 100)
        task_length = random.randint(70, 100)
        app = Application(
            app_name, task_length, upload_data_size * 1e3 * 8.0,
            APPLICATION_TYPES[app_name]['download_data_size'],
            APPLICATION_TYPES[app_name]['max_latency'],
            APPLICATION_TYPES[app_name]['lambda']
        )

        self.apps[app.app_type] = app

    # Requests VM instantiation on randomly selected MEC server
    def request_vm_instantiation(self):
        self.vm = self.server.instantiate_vm(self)
        return

    def sample_loc_coords(self):
        coords_space = [i for i in range(5)]
        x_coord = random.choice(coords_space)
        y_coord = random.choice(coords_space)

        self.x_coord = x_coord
        self.y_coord = y_coord
        # print("New coordinates for ", self, " are: ", x_coord, y_coord)

    # UE generates tasks and send to target MEC server through base station
    def generate_tasks(self):
        task_type_name = random.choice(list(self.apps.keys()))
        app = self.apps[task_type_name]
        lamda = app.arrival_rate
        while True:
            t_id_prefix = 'tid_' + str(self.uid) + "_" + str(len(self.generated_tasks) + 1)
            # Set dynamic task properties
            task_length = random.randint(50, 100)
            data_size = random.randint(2, 5)
            if len(list(self.apps.keys())) > 0:
                # Create source and destination addresses for the task
                src = str(self.uid) + ":" + str(self.base_station.bs_id)
                dst = str(self.server.base_station) + ":" + str(self.server.server_id)
                # Instantiate a task
                task = Task(t_id_prefix, self.uid, task_type_name,
                            task_length, data_size,
                            app.download_data_size, app.max_delay, src, dst)
                # Add task to list of generated tasks
                self.generated_tasks.append(task)
                # Upload task to base station for routing to MEC server
                self.base_station.input(task)
            yield self.env.timeout(random.expovariate(1. / lamda))

    def change_server(self, server):
        # Assigns a new MEC server to the user
        # Assigned server depends on location of VM of UE
        self.server = server

    def get_cell_id(self):
        return self.base_station.cell_id

    def is_target_user(self):
        # Check if current user is target user
        return True if self.target else False

    def _cell_residence_time(self):
        # Average time user spends in a cell: 5 seconds
        mean_residence_time = 5
        return random.expovariate(mean_residence_time)

    def walk(self):
        # ==================================================================
        # Moves user from one cell to another
        # ==================================================================
        # Fetches neighbor base stations of user's associated base station
        neighbor_cells = self.fetch_current_base_station_neighbours()
        neighbor_cells.append(self.base_station.bs_id)

        # Randomly select destination from neighboring base stations
        dest_cell_idx = random.choice(neighbor_cells)
        # get base station at index
        dest_cell = self.network_env.all_bs[int(dest_cell_idx)]
        # Connect user to new base station
        if dest_cell.bs_id == self.base_station.bs_id:
            pass
        else:
            print("UE_ID: {} | Source: {} | Dest: {}: ".format(self.uid, self.base_station.bs_id, dest_cell_idx))
            dest_cell.connect_user(self)
            # Disconnect user from previous base station
            self.base_station.disconnect_user(self)
            # Update the user's reference to base station
            self.base_station = dest_cell
            self.sample_loc_coords()

    def fetch_current_base_station_neighbours(self):
        net_height = 6
        net_width = 6
        current_location = self.base_station.bs_id
        totalLocations = net_width * net_height
        four_neighbor = [current_location - net_width, current_location + net_height,
                         current_location - 1, current_location + 1]  # four neighbors (up, down, left, right)

        enodeBValidNeighbors = list()
        for i in range(4):
            # Exclude neighbors that are not in square (Location_Length X Location_Width)
            if four_neighbor[i] < 0 or four_neighbor[i] > totalLocations - 1:
                four_neighbor[i] = -1

        # Exclude right neighbor of the most right side location and
        if current_location % 6 == net_width - 1:
            four_neighbor[3] = -1

        # Exclude left neighbor of the most left side location
        if current_location % 6 == 0:
            four_neighbor[2] = -1

        # print("Shortlisted Neighbours: ", four_neighbor)
        for n in four_neighbor:  # only save the valid neighbors
            if n != -1:
                enodeBValidNeighbors.append(n)

        enodeBValidNeighbors.append(current_location)
        return enodeBValidNeighbors

    def get_upload_data_size(self):
        first_app_key = list(self.apps.keys())[0]
        return self.apps[first_app_key].upload_data_size
