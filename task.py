from datetime import datetime


class Task(object):
    def __init__(self, tid, uid, app_type, task_length, up_data_size, dw_data_size, max_delay, src, dst):
        self.tid = tid
        self.uid = uid
        self.app_type = app_type
        self.task_length = task_length
        self.upload_data_size = up_data_size
        self.download_data_size = dw_data_size
        self.max_delay = max_delay
        self.src = src
        self.dst = dst
        self.t_server_arrival = None
        self.created_at = datetime.now().strftime("%H:%M:%S")

    def __repr__(self):
        return "Task ID: {} | User ID: {} | App Type: {} | Created_at: {}".format(
            self.tid, self.uid, self.app_type, self.created_at
        )


