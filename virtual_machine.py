class VirtualMachine:
    def __init__(self, vm_id, user_id, app_type, edge_server):
        self.vmid = vm_id
        self.uid = user_id
        self.edge_server = edge_server
        self.vm_size = 500
        self.app_type = app_type

    def __repr__(self):
        return "{} | {}".format(self.vmid, self.app_type)

    def set_new_server(self, server):
        self.edge_server = server


