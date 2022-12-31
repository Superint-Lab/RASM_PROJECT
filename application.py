class Application:
    def __init__(self, _app_type, _task_length, _up_data_size, _dw_data_size, _max_latency, _arrival_rate):
        self.app_type = _app_type
        self.task_length = _task_length
        self.upload_data_size = _up_data_size
        self.download_data_size = _dw_data_size
        self.max_delay = _max_latency
        self.arrival_rate = _arrival_rate

    def __repr__(self):
        return "{}".format(self.app_type)

    def get_app_type(self):
        return self.app_type

    def get_task_length(self):
        return self.task_length

    # def set_upload_data_size(self):

