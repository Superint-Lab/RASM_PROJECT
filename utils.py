import csv
import matplotlib.pyplot as plt
import numpy as np

env_width = env_height = 6
BLOCK_SIZE = 1

# Units conversion
conversion_units = {
    "millisec": 1e3,
    "microsec": 1e6,
    "nanosec": 1e9,
    "meters": 1e3,
}

def convert_scalar_to_pixel_coord(location):
    x = int((location // env_height) * BLOCK_SIZE)
    y = int(((location % env_width) / env_width) * (BLOCK_SIZE * env_width))
    return x, y


def find_target_user(users_dict):
    global user
    user = None
    for key in users_dict.keys():
        if key == 0 and users_dict[key].target is True:
            user = users_dict[key]

    return user


def csv_writer(data):
    with open("arrivals.csv", "a", newline='') as csvfile:
        task_writer = csv.writer(csvfile)
        task_writer.writerow(
            ["tid", "uid", "app_type", "task_length", "upload_data_size", "download_data_size", "max_delay", "src",
             "dst"])
        for d in data:
            task_writer.writerow(
                [d.tid, d.uid, d.app_type, d.task_length, d.upload_data_size, d.download_data_size, d.max_delay, d.src,
                 d.dst])


def csv_metrics_writer(metrics, file_name):
    # field_names = ["Ep.#", "RAN Delay", "MAN Delay", "Proc. Delay"]
    # file_name = "metrics_log_n_servers" + str(n_users) + ".csv"
    with open(file_name, "a", newline='') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(metrics)


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    # ax.plot(x, epsilons)
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.plot(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)
    plt.show()
    plt.savefig(filename)
