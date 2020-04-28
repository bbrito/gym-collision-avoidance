import os
import numpy as np
import pickle
from tqdm import tqdm

from gym_collision_avoidance.envs.config import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy

np.random.seed(1)

Config.EVALUATE_MODE = True
Config.SAVE_EPISODE_PLOTS = True
Config.SHOW_EPISODE_PLOTS = False
Config.ANIMATE_EPISODES = False
Config.DT = 0.1
start_from_last_configuration = False

results_subdir = 'non_cooperative_dataset'

test_case_fn = tc.get_traincase_2agents_swap
#test_case_fn = tc.get_testcase_random
policies = {
            'RVO': {
                'policy': RVOPolicy,
                },
            'NonCooperative': {
                'policy': NonCooperativePolicy
            }
            #'GA3C-CADRL-10': {
            #     'policy': GA3CCADRLPolicy,
            #     'checkpt_dir': 'IROS18',
            #     'checkpt_name': 'network_01900000'
            #     },
            }

num_agents_to_test = [2]
num_test_cases = 1000
test_case_args = {}
Config.PLOT_CIRCLES_ALONG_TRAJ = True
Config.NUM_TEST_CASES = num_test_cases
Config.EVALUATE_MODE =  True
Config.TRAIN_SINGLE_AGENT = False


def add_traj(agents, trajs, dt, traj_i, max_ts,last_time):
    agent_i = 0
    other_agent_i = (agent_i + 1) % 2
    #agent = agents[agent_i]
    #other_agent = agents[other_agent_i]
    #max_t = int(max_ts[agent_i])
    future_plan_horizon_secs = 3.0
    future_plan_horizon_steps = int(future_plan_horizon_secs / dt)
    for i, agent in enumerate(agents):
        for t in range(max_ts):
            robot_linear_speed = agent.global_state_history[t, 9]
            robot_angular_speed = agent.global_state_history[t, 10] / dt

            t_horizon = min(max_ts, t+future_plan_horizon_steps)
            future_linear_speeds = agent.global_state_history[t:t_horizon, 9]
            future_angular_speeds = agent.global_state_history[t:t_horizon, 10] / dt
            predicted_cmd = np.dstack([future_linear_speeds, future_angular_speeds])

            future_positions = agent.global_state_history[t:t_horizon, 1:3]

            d = {'time': np.round(agent.global_state_history[t, 0]+last_time,decimals=1),
                'pedestrian_state': {
                    'position': np.array([
                        agent.global_state_history[t, 1],
                        agent.global_state_history[t, 2],
                        ]),
                    'velocity': np.array([
                        agent.global_state_history[t, 7],
                        agent.global_state_history[t, 8],
                        ])
                },
                'pedestrian_goal_position': np.array([
                    agent.goal_global_frame[0],
                    agent.goal_global_frame[1],
                ])
            }
            trajs[traj_i+i].append(d)
    last_time += np.round(agent.global_state_history[-1, 0],decimals=1) +1
#     global_state = np.array([self.t,
#                                  self.pos_global_frame[0],
#                                  self.pos_global_frame[1],
#                                  self.goal_global_frame[0],
#                                  self.goal_global_frame[1],
#                                  self.radius,
#                                  self.pref_speed,
#                                  self.vel_global_frame[0],
#                                  self.vel_global_frame[1],
#                                  self.speed_global_frame,
#                                  self.heading_global_frame])

    return trajs, last_time


def main():
    env, one_env = create_env()
    dt = one_env.dt_nominal
    file_dir_template = os.path.dirname(os.path.realpath(__file__)) + '/../results/{results_subdir}/{num_agents}_agents'
    last_time = 0.0
    trajs = [[] for _ in range(num_test_cases*num_agents_to_test[0]*len(policies))]

    for num_agents in num_agents_to_test:

        file_dir = file_dir_template.format(num_agents=num_agents, results_subdir=results_subdir)
        plot_save_dir = file_dir + '/figs/'
        os.makedirs(plot_save_dir, exist_ok=True)
        one_env.plot_save_dir = plot_save_dir

        #test_case_args['num_agents'] = num_agents
        # What is this?
        #test_case_args['side_length'] = 7
        for test_case in tqdm(range(num_test_cases)):
            test_case_args['test_case_index'] = test_case % 10
            # test_case_args['num_test_cases'] = num_test_cases
            for j,policy in enumerate(policies):
                one_env.plot_policy_name = policy
                policy_class = policies[policy]['policy']
                test_case_args['agents_policy'] = policy_class
                agents = test_case_fn(**test_case_args)
                for agent in agents:
                    if 'checkpt_name' in policies[policy]:
                        agent.policy.env = env
                        agent.policy.initialize_network(**policies[policy])
                one_env.set_agents(agents)
                one_env.test_case_index = test_case
                init_obs = env.reset()

                times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck, agents = run_episode(env, one_env)

                max_ts = [t / dt for t in times_to_goal]
                # Change the global state history according with the number of steps required to finish the episode

                if not collision:
                    for agent in agents:
                        agent.global_state_history = agent.global_state_history[:agent.step_num]
                    trajs,last_time = add_traj(agents, trajs, dt, test_case*num_agents_to_test[0]*len(policies)+j*num_agents_to_test[0], agent.step_num,last_time)

        # print(trajs)
                
        one_env.reset()

        pkl_dir = file_dir + '/trajs/'
        os.makedirs(pkl_dir, exist_ok=True)
        fname = pkl_dir+policy+'.pkl'
        # Protocol 2 makes it compatible for Python 2 and 3
        pickle.dump(trajs, open(fname,'wb'), protocol=2)
        print('dumped {}'.format(fname))

    print("Experiment over.")

if __name__ == '__main__':
    main()