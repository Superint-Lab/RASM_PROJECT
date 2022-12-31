import random

APPLICATION_TYPES = {
    "HEALTH_APP": {
        "task_length": random.randint(50, 100),
        "upload_data_size": random.randint(2, 5),
        "download_data_size": 124,
        "max_latency": 30,
        "lambda": random.uniform(1, 3),
    },
    #
    # "INFOTAINMENT": {
    #     "task_length": random.randint(100, 500),
    #     "upload_data_size": random.randint(300, 400),
    #     "download_data_size": 124,
    #     "max_latency": 30,
    #     "lambda": random.uniform(3, 5),
    # },
    #
    # "AUGMENTED_REALITY": {
    #     "task_length": random.randint(100, 500),
    #     "upload_data_size": random.randint(300, 400),
    #     "download_data_size": 124,
    #     "max_latency": 30,
    #     "lambda": random.uniform(5, 10),
    # },
    #
    # "HEAVY_COMP_APP": {
    #     "task_length": random.randint(100, 500),
    #     "upload_data_size": random.randint(300, 400),
    #     "download_data_size": 124,
    #     "max_latency": 30,
    #     "lambda": random.uniform(3, 5),
    # }
}
