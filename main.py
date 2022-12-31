import random
from dqn_agent import Agent
from env import Env
from utils import *
import simpy


# simulation scenarios based on the number of
# Users i.e., starts with 10, 20, 30, 40, 50
simulation_scenarios = [20]
server_scenarios = [6, 12, 18, 24, 30, 36]
users_scenarios = [20, 40, 60, 80, 100, 120]
random.seed(123)
for n_servers in server_scenarios:
    n_users = users_scenarios[2]          # The number of users in the system
    n_episodes = 500                      # The number of training episodes
    n_steps = 80                          # The number of steps per episode
    input_dims = 2 + n_servers            # Number of input features for DQN agent
    batch_size = 64                       # Number of experience samples per batch size
    # The number of episodes after which the target net is updated
    target_update = 10

    # Initialize DQN agent
    server_keys = [i for i in range(n_servers)]
    num_actions = len(server_keys)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=batch_size,
                  n_actions=num_actions, eps_end=0.003,
                  input_dims=[input_dims], lr=0.001)

    # Data structures for holding simulation results
    scores, eps_history, losses, qoe, mig_cost = [], [], [], [], []
    ran_delay, man_delay, processing_delay = [], [], []

    # Initializing and setting simulation environment parameters
    simpy_sim_time = 60
    sim_time = n_steps * simpy_sim_time
    env = Env(simpy.Environment(), n_users=n_users, n_servers=n_servers, SIMULATION_DURATION=sim_time)

    for episode in range(1, n_episodes+1):
        score = 0
        done = False
        n_plays = 1
        total_loss = 0.0
        total_qoe = 0.0
        total_mig_cost = 0.0
        observation = env.reset()
        target_user = find_target_user(env.all_ue)

        # perf metrics variables
        d_RAN = d_MAN = d_proc = 0.0

        print("\n+++++++++++++++ Episode #: {} +++++++++++++++\n".format(episode))
        while n_plays < n_steps:
            print("\n************* Step #: {} *************".format(n_plays))
            if n_plays == (n_steps - 1):
                env.done = True
            else:
                env.done = False

            # move the users
            env.user_mobility()
            current_edge_id = target_user.server.server_id
            action = agent.select_action(observation)
            observation_, reward, qoe_, mig_cost_, perf_metrics = env.step(action)
            score += reward.item()
            agent.store_transition(observation, action, reward, observation_, done)
            loss = agent.learn()
            total_qoe += qoe_
            total_mig_cost += mig_cost_

            # Save perf. metrics to list data structure
            d_RAN += perf_metrics['d_ran']
            d_MAN += perf_metrics['d_man']
            d_proc += perf_metrics['d_proc']

            if loss is None:
                pass
            else:
                total_loss += loss

            observation = observation_

            n_plays += 1

        if episode % target_update == 0:
            print("\n+++++Updating the target network........\n")
            agent.Q_target.load_state_dict(agent.Q_val.state_dict())

        # ================================== Log metrics ==================================
        # Save the metrics to a log file (.xlx format)
        # metrics = {"d_ran": d_RAN, "d_man": d_MAN, "d_proc": d_proc}
        d_RAN = d_RAN / n_steps
        d_MAN = d_MAN / n_steps
        d_proc = d_proc / n_steps
        total_mig_cost = total_mig_cost / n_steps
        final_reward = np.mean(scores)
        n_migrations = env.total_migrations
        n_failed_migrations = env.rejected_migrations

        metrics = [episode, d_RAN, d_MAN, d_proc, total_mig_cost, final_reward, n_migrations, n_failed_migrations]

        csv_file_name_MECs = "metrics_log_" + str(n_servers) + "_MECv9_RASM.csv"
        csv_metrics_writer(metrics, csv_file_name_MECs)
        # ================================== End of Log metrics ==================================
        qoe.append(total_qoe)
        mig_cost.append(total_mig_cost)
        losses.append(total_loss)
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores)
        avg_loss = np.mean(losses)
        print('Episode ', episode, 'reward %.2f' % score, 'Average reward %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon, 'Loss %.2f' % avg_loss)

        print("Attempted Migrations: ", env.total_migrations, "Failed Migrations: ", env.rejected_migrations)

    file_name = 'test.png'
    x = [i + 1 for i in range(n_episodes)]
    plot_learning_curve(x, scores, eps_history, file_name)

    plt.plot(losses)
    plt.xlabel("Episodes")
    plt.ylabel("Total Loss")
    plt.show()

    plt.plot(qoe)
    plt.xlabel("Episodes")
    plt.ylabel("Quality of Service")
    plt.show()

    plt.plot(mig_cost)
    plt.xlabel("Episodes")
    plt.ylabel("Migration cost")
    plt.show()

    print("Average QoE :%.2f" % np.mean(qoe))
    print("Migration Cost :%.2f" % np.mean(mig_cost))
    print("Reward :%.2f" % np.mean(scores))
