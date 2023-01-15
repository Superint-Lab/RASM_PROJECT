import random
from statistics import mean, StatisticsError
import simpy
from virtual_machine import VirtualMachine
from utils import find_target_user
import csv


class Server(object):
    def __init__(self, env, server_id, network, base_station=None):
        self.env = env
        self.network = network
        self.store = simpy.Store(self.env)
        self.virtual_machines = {}
        self.server_id = server_id
        self.base_station = base_station
        self.vms_hosting_cap = 10
        self.MIPS = 700_000_000
        self.memory_size = 16
        self.arrivals_t = []
        self.departs_t = []
        self.waits = []
        self.service_times = []
        self.inter_arrival_times = []
        self.arrivals = 0
        self.last_arrival = 0.0
        self.serviced_task = []
        self.logs = []
        self.delay = 0.0
        self.empirical = 0.0
        self.action_ = self.env.process(self.process_task())

    def __repr__(self):
        return "Server ID: {} | Server BS: {}".format(self.server_id, self.base_station)

    def instantiate_vm(self, user):
        vm_id = "v" + str(user.uid) + str(len(self.virtual_machines) + 1)
        app_type = list(user.apps.keys())[0]
        vm = VirtualMachine(vm_id, user.uid, app_type, user.server)
        self.virtual_machines[vm_id] = vm

        return vm

    def deploy_migrated_vm(self, vm):
        print("VM ID: ", vm.uid)
        if self.network.all_ue[vm.uid].is_target_user():
            self.network.total_migrations += 1
            if len(self.virtual_machines) != self.vms_hosting_cap:
                vm.set_new_server(self)
                self.virtual_machines[vm.vmid] = vm
                return True
            else:
                self.network.rejected_migrations += 1
                print(self.virtual_machines)
                print("Current VM count: ", len(self.virtual_machines))
                return False
        else:
            if len(self.virtual_machines) != self.vms_hosting_cap:
                vm.set_new_server(self)
                self.virtual_machines[vm.vmid] = vm
                return True
            else:
                return False

    def remove_vm(self, vm):
        if vm.vmid in self.virtual_machines:
            print("VMs before migration: ", len(self.virtual_machines))
            del self.virtual_machines[vm.vmid]
            print("VMs after migration: ", len(self.virtual_machines))
            return True
        else:
            return False

    # Moves service VM from current server to the destination server
    def migrate(self, user):
        # ---------- Remember to implement migr. rejection here -------------
        source_server = user.server
        if self.deploy_migrated_vm(user.vm):
            user.change_server(self)
            source_server.remove_vm(user.vm)

    def put(self, task):
        curr_arrival_t = self.env.now
        inter_arrival = curr_arrival_t - self.last_arrival
        self.inter_arrival_times.append(inter_arrival)
        self.last_arrival = curr_arrival_t
        task.t_server_arrival = curr_arrival_t
        self.arrivals_t.append(curr_arrival_t)
        self.store.put(task)
        self.arrivals += 1
        self.logs.append(task)

    def process_task(self):
        while True:
            task = (yield self.store.get())
            t_depart = self.env.now
            waiting_time = t_depart - task.t_server_arrival
            self.waits.append(waiting_time)
            # processing delay
            proc_delay = (task.task_length * 1e6) / self.MIPS
            # record processing delay for the processed task
            self.departs_t.append(proc_delay)
            delay_time = random.expovariate(1./proc_delay)
            print("General Delay: ", proc_delay, " Exponential delay: ", delay_time)
            cpu_entry = self.env.now
            yield self.env.timeout(delay_time)
            self.serviced_task.append(task)
            self.service_times.append(self.env.now - cpu_entry)

    def get_empirical_proc_delay(self):
        avg_service_time = mean(self.service_times) if len(self.service_times) > 1 else 0
        avg_waiting_time = mean(self.waits) if len(self.waits) > 1 else 0

        return avg_service_time + avg_waiting_time

    def get_theoretical_proc_delay(self):
        if self.arrivals > 1:
            mu = 1./mean(self.service_times)
            lamda = 1./mean(self.inter_arrival_times)
            t_service = 1./(mu - lamda)
            util = lamda / mu
            return t_service, util
        else:
            return 0, 0

    def get_resource_capacity(self):
        # Resource capacity of all candidate servers
        max_capacity = 100  # 100%
        available_capacity = max_capacity - (self.get_theoretical_proc_delay()[1] * max_capacity)
        return available_capacity

    def get_num_users(self):
        return len(self.virtual_machines)

