'''
Collision Avoidance Environment
Author: Michael Everett
MIT Aerospace Controls Lab
'''
import gc
import gym
import gym.spaces
import numpy as np
import itertools
import copy
import os
import matplotlib.pyplot as plt

from gym_collision_avoidance.envs.config import Config
#from gym_collision_avoidance.envs.utils import DataHandlerLSTM
from gym_collision_avoidance.envs.util import find_nearest, rgba2rgb
from gym_collision_avoidance.envs.visualize import plot_episode, animate_episode
from gym_collision_avoidance.envs.agent import Agent
from gym_collision_avoidance.envs.Map import Map
from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxAcc import UnicycleDynamicsMaxAcc
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxTurnRate import UnicycleDynamicsMaxTurnRate
from gym_collision_avoidance.envs.dynamics.UnicycleSecondOrderEulerDynamics import UnicycleSecondOrderEulerDynamics
from mpc_rl_collision_avoidance.policies.MPCPolicy import MPCPolicy
from mpc_rl_collision_avoidance.policies.SecondOrderMPCPolicy import SecondOrderMPCPolicy
from mpc_rl_collision_avoidance.policies.SecondOrderMPCRLPolicy import SecondOrderMPCRLPolicy
from mpc_rl_collision_avoidance.policies.FirstOrderMPCPolicy import FirstOrderMPCPolicy
from mpc_rl_collision_avoidance.policies.FirstOrderMPCRLPolicy import FirstOrderMPCRLPolicy
from mpc_rl_collision_avoidance.policies.MultiAgentMPCPolicy import MultiAgentMPCPolicy
from mpc_rl_collision_avoidance.policies.MPCStaticObsPolicy import MPCStaticObsPolicy
from mpc_rl_collision_avoidance.policies.MPCRLStaticObsPolicy import MPCRLStaticObsPolicy
from mpc_rl_collision_avoidance.policies.SocialMPCPolicy import SocialMPCPolicy
from mpc_rl_collision_avoidance.policies.SociallyGuidedMPCPolicy import SociallyGuidedMPCPolicy
from mpc_rl_collision_avoidance.policies.FirstOrderMPCPolicy import FirstOrderMPCPolicy
from mpc_rl_collision_avoidance.policies.SecondOrderMPCPolicy import SecondOrderMPCPolicy
from mpc_rl_collision_avoidance.policies.MPCRLPolicy import MPCRLPolicy
from mpc_rl_collision_avoidance.policies.LearningMPCPolicy import LearningMPCPolicy
from mpc_rl_collision_avoidance.policies.OtherAgentMPCPolicy import OtherAgentMPCPolicy
from timeit import default_timer as timer
import random

class CollisionAvoidanceEnv(gym.Env):
    metadata = {
        # UNUSED !!
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.id = 0

        # Initialize Rewards
        self._initialize_rewards()

        # Simulation Parameters
        self.num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.dt_nominal = Config.DT

        # Collision Parameters
        self.collision_dist = Config.COLLISION_DIST
        self.getting_close_range = Config.GETTING_CLOSE_RANGE

        # Plotting Parameters
        self.evaluate = Config.EVALUATE_MODE

        self.plot_episodes = Config.SHOW_EPISODE_PLOTS or Config.SAVE_EPISODE_PLOTS
        self.plt_limits = Config.PLT_LIMITS
        self.plt_fig_size = Config.PLT_FIG_SIZE
        self.test_case_index = 0

        self.animation_period_steps = Config.ANIMATION_PERIOD_STEPS

        self.number_of_agents = 2
        self.scenario = Config.SCENARIOS_FOR_TRAINING

        #self.ego_policy = "SecondOrderMPCRLPolicy"

        self.ego_policy = "MPCRLStaticObsPolicy"

        self.ego_agent_dynamics = "UnicycleSecondOrderEulerDynamics"

        self.other_agents_policy = "RVOPolicy"
        self.other_agents_dynamics = "UnicycleDynamics"

        self.max_heading_change = 4
        self.min_heading_change = -4
        self.min_speed = -4
        self.max_speed = 4

        ### The gym.spaces library doesn't support Python2.7 (syntax of Super().__init__())
        self.action_space_type = Config.ACTION_SPACE_TYPE
        
        if self.action_space_type == Config.discrete:
            self.action_space = gym.spaces.Discrete(self.actions.num_actions, dtype=np.float32)
        elif self.action_space_type == Config.continuous:
            self.low_action = np.array([self.min_speed,
                                        self.min_heading_change])
            self.high_action = np.array([self.max_speed,
                                         self.max_heading_change])
            self.action_space = gym.spaces.Box(self.low_action, self.high_action, dtype=np.float32)
        

        # original observation space
        # self.observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        
        # not used...
        # self.observation_space = np.array([gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
                                           # for _ in range(self.num_agents)])
        # observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        
        # single agent dict obs
        self.observation = {}
        for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
            self.observation[agent] = {}

        self.observation_space = gym.spaces.Dict({})
        for state in Config.STATES_IN_OBS:
            self.observation_space.spaces[state] = gym.spaces.Box(Config.STATE_INFO_DICT[state]['bounds'][0]*np.ones((Config.STATE_INFO_DICT[state]['size'])),
                Config.STATE_INFO_DICT[state]['bounds'][1]*np.ones((Config.STATE_INFO_DICT[state]['size'])),
                dtype=Config.STATE_INFO_DICT[state]['dtype'])
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']), dtype=Config.STATE_INFO_DICT[state]['dtype'])

        self.agents = None
        self.default_agents = None
        self.prev_episode_agents = None

        self.static_map_filename = None
        self.map = None

        self.episode_step_number = 0
        self.episode_number = 0
        self.total_number_of_steps = 1

        self.n_collisions = np.zeros([100])
        self.n_timeouts = np.zeros([100])

        self.plot_save_dir = None
        self.plot_policy_name = None

        self.perturbed_obs = None

        self.obstacles = None

        self.prediction_model = None
        self.online_training = None

        self.track_statistic_performance = False

        self.prev_scenario_index = 0
        self.scenario_index = 0

        self.avg_step_time = 0
        self.avg_reset_time = 0

        gc.enable()

    def step(self, actions, dt=None):
        ###############################
        # This is the main function. An external process will compute an action for every agent
        # then call env.step(actions). The agents take those actions,
        # then we check if any agents have earned a reward (collision/goal/...).
        # Then agents take an observation of the new world state. We compute whether each agent is done
        # (collided/reached goal/ran out of time) and if everyone's done, the episode ends.
        # We return the relevant info back to the process that called env.step(actions).
        #
        # Inputs
        # - actions: list of [delta heading angle, speed] commands (1 per agent in env)
        # Outputs
        # - next_observations: (obs_length x num_agents) np array with each agent's observation
        # - rewards: list with 1 scalar reward per agent in self.agents
        # - game_over: boolean, true if every agent is done
        # - info_dict: metadata (more details) that help in training, for example
        ###############################
        start = timer()

        if dt is None:
            dt = self.dt_nominal

        self.episode_step_number += 1
        self.total_number_of_steps += 1

        # Generate Predictions
        if self.prediction_model:
            self._prediction_step()

        # Take action
        self._take_action(actions, dt)

        # Collect rewards
        rewards = self._compute_rewards()

        # Take observation
        next_observations = self._get_obs()

        if (self.episode_number % Config.PLOT_EVERY_N_EPISODES == 1 or Config.EVALUATE_MODE) and Config.ANIMATE_EPISODES and self.episode_number >= 1 and self.episode_step_number % self.animation_period_steps == 0:
            plot_episode(self.agents, self.obstacles, False, self.map, self.episode_number,
                circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                plot_save_dir=self.plot_save_dir,
                plot_policy_name=self.plot_policy_name,
                save_for_animation=True,
                limits=self.plt_limits,
                fig_size=self.plt_fig_size,
                perturbed_obs=self.perturbed_obs,
                show=False,
                save=True)

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()

        if game_over and self.track_statistic_performance:
            self.n_collisions = np.roll(self.n_collisions, 1)
            self.n_collisions[0] = self.agents[0].in_collision
            self.n_timeouts = np.roll(self.n_timeouts, 1)
            self.n_timeouts[0] = self.agents[0].ran_out_of_time

        if self.online_training:
            self.prediction_model.data_handler.addEpisodeData(self.agents)
            if (self.episode_number >= 100) and (self.episode_number%100==0) and (len(self.prediction_model.data_handler.trajectory_set)>=100):
                self.prediction_model.train_step(self.episode_number,np.mean(self.n_collisions),np.mean(self.n_timeouts))

        which_agents_done_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]

        return next_observations, rewards, game_over, \
            {'which_agents_done': which_agents_done_dict}

        end = timer()
        self.avg_step_time = (end - start + self.total_number_of_steps*self.avg_step_time)/(self.total_number_of_steps+1)

    def reset(self):
        start = timer()

        if (self.episode_number % Config.PLOT_EVERY_N_EPISODES == 1 or Config.EVALUATE_MODE) and Config.ANIMATE_EPISODES and self.episode_number >= 1 and self.episode_step_number > 10:
            plot_episode(self.agents, self.obstacles, Config.TRAIN_MODE, self.map, self.episode_number,
                         self.id, circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                         plot_save_dir=self.plot_save_dir,
                         plot_policy_name=self.plot_policy_name,
                         limits=self.plt_limits,
                         fig_size=self.plt_fig_size,
                         show=Config.SHOW_EPISODE_PLOTS,
                         save=Config.SAVE_EPISODE_PLOTS)
            animate_episode(num_agents=len(self.agents), plot_save_dir=self.plot_save_dir,
                            plot_policy_name=self.plot_policy_name, test_case_index=self.episode_number,
                            agents=self.agents)

        # Collect trash
        gc.collect()

        self.episode_number += 1
        self.begin_episode = True
        self.episode_step_number = 0
        self._init_agents()

        if self.prediction_model:
            self._init_prediction_model()

        self._init_static_map()

        for state in Config.STATES_IN_OBS:
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']), dtype=Config.STATE_INFO_DICT[state]['dtype'])

        end = timer()

        self.avg_reset_time = (end - start + self.total_number_of_steps*self.avg_reset_time)/(self.total_number_of_steps+1)

        return self._get_obs()

    def close(self):
        print("--- Closing CollisionAvoidanceEnv! ---")
        return

    def _prediction_step(self):
        # TODO: move initial CV predictions to predictiton model class
        if self.episode_number >= 100:
            self.predicted_trajectory = self.prediction_model.query(self.agents)[0]
        else:
            # For the first time step Use CV model
            self.predicted_trajectory = np.zeros((len(self.agents),1, Config.FORCES_N, 6))
            for ag_id, agent in enumerate(self.agents):
                for t in range(Config.FORCES_N):
                    self.predicted_trajectory[ag_id,0, t,:2] = agent.pos_global_frame + agent.vel_global_frame * Config.FORCES_DT
                    self.predicted_trajectory[ag_id,0, t, 4:6] = agent.vel_global_frame
        indices = np.arange(len(self.agents))
        for id, agent in enumerate(self.agents):
            agent.policy.predicted_trajectory =  self.predicted_trajectory[indices != id]
            agent.policy.all_predicted_trajectory = self.predicted_trajectory

    def _take_action(self, actions, dt):
        num_actions_per_agent = 2  # speed, delta heading angle
        all_actions = np.zeros((len(self.agents), num_actions_per_agent), dtype=np.float32)

        # Agents set their action (either from external or w/ find_next_action)
        for agent_index, agent in enumerate(self.agents):
            if agent.is_done:
                continue
            if agent.policy.is_external:
                all_actions[agent_index, :] = agent.policy.convert_to_action(actions[agent_index])
            elif agent.policy.is_still_learning:
                all_actions[agent_index, :] = agent.policy.network_output_to_action(agent_index,self.agents, actions)
            else:
                dict_obs = self.observation[agent_index]
                all_actions[agent_index, :] = agent.policy.find_next_action(dict_obs, self.agents, agent_index, self.obstacles) #obstacle is added by Sant

        # After all agents have selected actions, run one dynamics update
        for i, agent in enumerate(self.agents):
            agent.take_action(all_actions[i,:], dt)

        # If agents follow a Multi-Agent MPC Policy Update their predicted plans
        for i, agent in enumerate(self.agents):
            if str(agent.policy) == 'MultiAgentMPCPolicy':
                agent.policy.update_predicted_trajectory()

    def update_top_down_map(self):
        print("Not adding agents to map")
        #self.map.add_agents_to_map(self.agents)
        #plt.imshow(self.map.map)
        #plt.pause(0.1)

    def set_agents(self, agents):
        self.default_agents = agents

    def _init_prediction_model(self):
        """
        Initialize prediction model function
        :return: None
        """
        if self.prediction_model:
            if self.episode_number > 1:
               self.prediction_model.reset_states(len(self.agents))
            else:
               self.prediction_model.load_model(len(self.agents))
            self.plot_policy_name = self.agents[0].policy.str + '_' + str(self.prediction_model)

        else:
           self.plot_policy_name = self.agents[0].policy.str + '_CV'

        self._prediction_step()

    def get_new_scenario(self):
        scenario_index = np.random.randint(0, len(self.scenario))
        if self.evaluate:
            number_of_agents = self.number_of_agents
        else:
            # if self.total_number_of_steps < 100000:
            #     # Supervised learning step
            #     number_of_agents = random.randint(2, 6) # Maximum no. of agents
            # # RL steps:
            # elif self.total_number_of_steps < 2e6:
            #     number_of_agents = 2
            # elif self.total_number_of_steps < 4e6:
            #     number_of_agents = 4
            # elif self.total_number_of_steps < 6e6:
            #     number_of_agents = 6
            # elif self.total_number_of_steps < 10e6:
            #     number_of_agents = 8
            # elif self.total_number_of_steps >= 10e6:
            #     number_of_agents = 10
            if self.total_number_of_steps < 100000:
                # Supervised learning step
                number_of_agents = 6 # Maximum no. of agents
            # RL steps:
            elif self.total_number_of_steps < 1e6:
                number_of_agents = 2
            elif self.total_number_of_steps < 2e6:
                number_of_agents = 4
            elif self.total_number_of_steps < 4e6:
                number_of_agents = 6
            elif self.total_number_of_steps >= 4e6:
                number_of_agents = 6
        return scenario_index, number_of_agents

    def _init_agents(self):
        if self.agents is not None:
            self.prev_episode_agents = copy.deepcopy(self.agents)

        self.scenario_index, self.number_of_agents = self.get_new_scenario()
        print(self.scenario[self.scenario_index] + "(number_of_agents=" + str(self.number_of_agents))
        if self.evaluate:
            self.prev_scenario_index = self.scenario_index
            self.scenario_index = np.random.randint(0, len(self.scenario))
            if Config.ANIMATE_EPISODES or Config.PERFORMANCE_TEST:
                self.agents, self.obstacles = eval("tc." + self.scenario[self.scenario_index] + "(number_of_agents=" + str(self.number_of_agents) + ", ego_agent_policy=" + self.ego_policy+ ", other_agents_policy=" + self.other_agents_policy + ", seed="+str(self.episode_number) +
                               ", ego_agent_dynamics=" + self.ego_agent_dynamics +", other_agents_dynamics=" + self.other_agents_dynamics  +")")
            else:
                self.agents, self.obstacles= eval("tc." + self.scenario[self.scenario_index] + "(number_of_agents=" + str(
                    self.number_of_agents) + ", ego_agent_policy=" + self.ego_policy  + ", other_agents_policy=" + self.other_agents_policy +
                               ", ego_agent_dynamics=" + self.ego_agent_dynamics +", other_agents_dynamics=" + self.other_agents_dynamics
                                                   + ")")
        else:
            self.agents, self.obstacles = eval("tc." + self.scenario[self.scenario_index] + "(number_of_agents=" + str(self.number_of_agents) + ", ego_agent_policy=" + self.ego_policy +
                                               ", ego_agent_dynamics=" + self.ego_agent_dynamics + ", other_agents_dynamics=" + self.other_agents_dynamics + ", other_agents_policy=" + self.other_agents_policy + ")")

        # TODO: This is used to avoid loading the network parameters over and over for this specific model which takes a lot of time
        print("2")
        if self.episode_number == 1:
            self.policies=[]
            ga3c_params = {
                'policy': GA3CCADRLPolicy,
                'checkpt_dir': 'IROS18',
                'checkpt_name': 'network_01900000'
            }
            for ag in self.agents:
                if "GA3C" in str(ag.policy):
                    self.policies.append(GA3CCADRLPolicy())
                    self.policies[-1].initialize_network(**ga3c_params)
                    ag.policy = self.policies[-1]
        else:
            i = 0
            for ag in self.agents:
                if "GA3C" in str(ag.policy):
                    ag.policy = self.policies[i]
                    i += 1

        if self.prediction_model:
            self.prediction_model.reset_states(len(self.agents))

        for agent in self.agents:
            agent.max_heading_change = self.max_heading_change
            agent.max_speed = self.max_speed

        # TODO: MOVE THIS TO THE POLICY CLASS for the multi agent case
        for agent in self.agents:
            if str(agent.policy) == 'MultiAgentMPCPolicy':
                agent.policy.current_state_[0] = agent.pos_global_frame[0]
                agent.policy.current_state_[1] = agent.pos_global_frame[1]
                agent.policy.update_predicted_trajectory()
            #if str(agent.policy) == 'PedestrianDatasetPolicy':
            #    agent.policy.trajectory = self.data_prep.getRandomTrajectory()

    def set_static_map(self, map_filename):
        self.static_map_filename = map_filename

    def _init_static_map(self):
        #self.set_static_map('../gym-collision-avoidance/gym_collision_avoidance/envs/world_maps/002.png')
        '''
        ## Michael everetts version:
        if isinstance(self.static_map_filename, list):
            static_map_filename = np.random.choice(self.static_map_filename)
        else:
            static_map_filename = self.static_map_filename
        '''
        ## Sants version:
        # Check if there are obstacles given
        if len(self.obstacles) == 0:
            static_map_filename = None
        else: 
            static_map_filename = self.obstacles

        self.map = Map(Config.MAP_WIDTH, Config.MAP_HEIGHT, Config.SUBMAP_RESOLUTION, static_map_filename)

        for agent in self.agents:
            agent.policy.map = self.map

    def _compute_rewards(self):
        ###############################
        # Check for collisions and reaching of the goal here, and also assign
        # the corresponding rewards based on those calculations.
        #
        # Outputs
        #   - rewards: is a scalar if we are only training on a single agent, or
        #               is a list of scalars if we are training on mult agents
        ###############################

        # if nothing noteworthy happened in that timestep, reward = -0.01
        rewards = self.reward_time_step*np.ones(len(self.agents))
        collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent = \
            self._check_for_collisions()

        for i, agent in enumerate(self.agents):
            if agent.is_at_goal:
                if agent.was_at_goal_already is False:
                    # agents should only receive the goal reward once
                    rewards[i] = self.reward_at_goal #- np.linalg.norm(agent.past_actions[0,:])
                    print("Agent %i: Arrived at goal!"% agent.id)
            else:
                # collision with other agent
                if agent.was_in_collision_already is False:
                    if collision_with_agent[i]:
                        rewards[i] = self.reward_collision_with_agent
                        agent.in_collision = True
                        print("Agent %i: Collision with another agent!"
                               % agent.id)
                    #collision with a static obstacle
                    elif collision_with_wall[i]:
                        rewards[i] = self.reward_collision_with_wall
                        agent.in_collision = True
                        # print("Agent %i: Collision with wall!"
                              # % agent.id)
                    # There was no collision
                    else:
                        # Penalty for getting close to agents
                        if dist_btwn_nearest_agent[i] <= Config.GETTING_CLOSE_RANGE:
                            rewards[i] += -0.1 - dist_btwn_nearest_agent[i]/2.
                            # print("Agent %i: Got close to another agent!"
                            #       % agent.id)
                        # Penalty for wiggly behavior
                        if np.linalg.norm(agent.past_actions[-1,:]-agent.past_actions[0,:]) > self.wiggly_behavior_threshold:
                            # Slightly penalize wiggly behavior
                            rewards[i] += self.reward_wiggly_behavior
                        # elif entered_norm_zone[i]:
                        #     rewards[i] = self.reward_entered_norm_zone

                elif agent.ran_out_of_time:
                    if i ==0:
                        print("Agent 0 is out of time.")
                    rewards[i] += Config.REWARD_TIMEOUT

                # If action is infeasible
                if agent.is_infeasible:
                    rewards[i] += Config.REWARD_INFEASIBLE

                # if gets close to goal
                rewards[i] += Config.REWARD_DISTANCE_TO_GOAL * (agent.past_dist_to_goal - agent.dist_to_goal)

        rewards = np.clip(rewards, self.min_possible_reward,
                          self.max_possible_reward)/(self.max_possible_reward - self.min_possible_reward)
        if Config.TRAIN_SINGLE_AGENT:
            rewards = rewards[0]
        return rewards

    def _compute_action_reward(self,action,agents):
        ###############################
        # Check for collisions and reaching of the goal here, and also assign
        # the corresponding rewards based on those calculations.
        #
        # Outputs
        #   - rewards: is a scalar if we are only training on a single agent, or
        #               is a list of scalars if we are training on mult agents
        ###############################

        # if nothing noteworthy happened in that timestep, reward = -0.01
        rewards = self.reward_time_step
        ego_agent = agents[0]
        other_agents = agents[1:]

        collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent = \
            self.check_action_for_collisions(action,ego_agent,other_agents)

        is_in_goal_direction = (ego_agent.pos_global_frame[0] + action[0,0] - ego_agent.goal_global_frame[0]) ** 2 + (
                    ego_agent.pos_global_frame[1] + action[0,1] - ego_agent.goal_global_frame[1]) ** 2 <= ego_agent.near_goal_threshold ** 2

        if is_in_goal_direction:
            if ego_agent.was_at_goal_already is False:
                # agents should only receive the goal reward once
                rewards = self.reward_at_goal  # - np.linalg.norm(agent.past_actions[0,:])
                print("Agent %i: Is going to the goal!" % ego_agent.id)
        else:
            for i, agent in enumerate(other_agents):
                # collision with other agent
                if ego_agent.was_in_collision_already is False:
                    if collision_with_agent[i]:
                        rewards = self.reward_collision_with_agent
                        agent.in_collision = True
                        print("\32 Agent %i: Collision with another agent!"
                               % agent.id)
                    #collision with a static obstacle
                    elif collision_with_wall[i]:
                        rewards = self.reward_collision_with_wall
                        agent.in_collision = True
                        # print("Agent %i: Collision with wall!"
                              # % agent.id)
                    # There was no collision
                    else:
                        # Penalty for getting close to agents
                        if dist_btwn_nearest_agent[i] <= Config.GETTING_CLOSE_RANGE:
                            rewards = -0.1 - dist_btwn_nearest_agent[i]/2.
                            # print("Agent %i: Got close to another agent!"
                            #       % agent.id)
                        # Penalty for wiggly behavior
                        if np.linalg.norm(ego_agent.past_actions[-1,:]-ego_agent.past_actions[0,:]) > self.wiggly_behavior_threshold:
                            # Slightly penalize wiggly behavior
                            rewards += self.reward_wiggly_behavior
                        # elif entered_norm_zone[i]:
                        #     rewards[i] = self.reward_entered_norm_zone
            # if gets close to goal
            rewards -= Config.REWARD_DISTANCE_TO_GOAL * np.linalg.norm(ego_agent.goal_global_frame - ego_agent.pos_global_frame - action[0])

        rewards = np.clip(rewards, self.min_possible_reward,
                          self.max_possible_reward)/(self.max_possible_reward - self.min_possible_reward)
        return rewards

    def _check_for_collisions(self):
        # NOTE: This method doesn't compute social zones!!!!!
        collision_with_agent = [False for _ in self.agents]
        collision_with_wall = [False for _ in self.agents]
        entered_norm_zone = [False for _ in self.agents]
        dist_btwn_nearest_agent = [np.inf for _ in self.agents]
        agent_shapes = []
        agent_front_zones = []
        agent_inds = list(range(len(self.agents)))
        agent_pairs = list(itertools.combinations(agent_inds, 2))
        for i, j in agent_pairs:
            agent = self.agents[i]
            other_agent = self.agents[j]
            dist_btwn = np.linalg.norm(
                agent.pos_global_frame - other_agent.pos_global_frame)
            combined_radius = agent.radius + other_agent.radius
            dist_btwn_nearest_agent[i] = min(dist_btwn_nearest_agent[i], dist_btwn - combined_radius)
            if dist_btwn <= combined_radius:
                # Collision with another agent!
                collision_with_agent[i] = True
                collision_with_agent[j] = True
                if i == 0 and collision_with_agent[i]:
                    print("Ego-agent collided")
        if self.obstacles:
            for i in agent_inds:
                agent = self.agents[i]
                [pi, pj], in_map = self.map.world_coordinates_to_map_indices(agent.pos_global_frame)
                mask = self.map.get_agent_map_indices([pi, pj], agent.radius)
                # plt.figure('static map')
                # plt.imshow(self.map.static_map + mask)
                # plt.pause(0.1)
                if in_map and np.any(self.map.static_map[mask]):
                    print("Collision with wall!")
                    collision_with_wall[i] = True
        else:
            for i in agent_inds:
                collision_with_wall[i] = False

        return collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent

    def check_action_for_collisions(self,action,ego_agent,other_agents):
        # NOTE: This method doesn't compute social zones!!!!!
        collision_with_agent = [False for _ in other_agents]
        collision_with_wall = [False for _ in other_agents]
        entered_norm_zone = [False for _ in other_agents]
        dist_btwn_nearest_agent = [np.inf for _ in other_agents]
        agent_shapes = []
        agent_front_zones = []
        i = 0
        for other_agent in other_agents:
            dist_btwn = np.linalg.norm(
                ego_agent.pos_global_frame + action - other_agent.pos_global_frame)
            combined_radius = ego_agent.radius + other_agent.radius
            dist_btwn_nearest_agent[i] = min(dist_btwn_nearest_agent[i], dist_btwn - combined_radius)
            if dist_btwn <= combined_radius:
                # Collision with another agent!
                collision_with_agent[i] = True
            i += 1
        #TODO: Static Collision Avoidance check
        if self.obstacles:
            for i in agent_inds:
                agent = self.agents[i]
                [pi, pj], in_map = self.map.world_coordinates_to_map_indices(agent.pos_global_frame)
                mask = self.map.get_agent_map_indices([pi, pj], agent.radius)
                # plt.figure('static map')
                # plt.imshow(self.map.static_map + mask)
                # plt.pause(0.1)
                if in_map and np.any(self.map.static_map[mask]):
                    # Collision with wall!
                    collision_with_wall[i] = True



        return collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent

    def _check_which_agents_done(self):
        at_goal_condition = np.array(
                [a.is_at_goal for a in self.agents])
        ran_out_of_time_condition = np.array(
                [a.ran_out_of_time for a in self.agents])
        in_collision_condition = np.array(
                [a.in_collision for a in self.agents])
        which_agents_done = np.logical_or.reduce((at_goal_condition, ran_out_of_time_condition, in_collision_condition))
        for agent_index, agent in enumerate(self.agents):
            agent.is_done = which_agents_done[agent_index]
        
        if Config.EVALUATE_MODE:
            # Episode ends when every agent is done
            if Config.HOMOGENEOUS_TESTING:
                game_over = np.all(which_agents_done)
            else:
                game_over = which_agents_done[0]
                # hack just to get the plots with all agents finishing at same time
                #game_over = np.all(which_agents_done)
        elif Config.TRAIN_SINGLE_AGENT:
            # Episode ends when ego agent is done
            game_over = which_agents_done[0]
        else:
            # Episode is done when all *learning* agents are done
            learning_agent_inds = [i for i in range(len(self.agents)) if self.agents[i].policy.is_still_learning]
            game_over = np.all(which_agents_done[learning_agent_inds])
        
        return which_agents_done, game_over

    def _get_obs(self):

        # Agents have moved (states have changed), so update the map view
        #self.update_top_down_map()

        # Agents collect a reading from their map-based sensors
        for i, agent in enumerate(self.agents):
            agent.sense(self.agents, i, self.map)

        # Agents fill in their element of the multiagent observation vector
        for i, agent in enumerate(self.agents):
            self.observation[i] = agent.get_observation_dict(self.agents)

        return self.observation

    def _initialize_rewards(self):
        self.reward_at_goal = Config.REWARD_AT_GOAL
        self.reward_collision_with_agent = Config.REWARD_COLLISION_WITH_AGENT
        self.reward_collision_with_wall = Config.REWARD_COLLISION_WITH_WALL
        self.reward_getting_close = Config.REWARD_GETTING_CLOSE
        self.reward_entered_norm_zone = Config.REWARD_ENTERED_NORM_ZONE
        self.reward_time_step = Config.REWARD_TIME_STEP

        self.reward_wiggly_behavior = Config.REWARD_WIGGLY_BEHAVIOR
        self.wiggly_behavior_threshold = Config.WIGGLY_BEHAVIOR_THRESHOLD

        self.possible_reward_values = \
            np.array([self.reward_at_goal,
                      self.reward_collision_with_agent,
                      self.reward_time_step,
                      self.reward_collision_with_wall,
                      self.reward_wiggly_behavior
                      ])
        self.min_possible_reward = np.min(self.possible_reward_values)
        self.max_possible_reward = np.max(self.possible_reward_values)

    def set_plot_save_dir(self, plot_save_dir):
        os.makedirs(plot_save_dir, exist_ok=True)
        self.plot_save_dir = plot_save_dir

    def set_perturbed_info(self, perturbed_obs):
        self.perturbed_obs = perturbed_obs

if __name__ == '__main__':
    print("See example.py for a minimum working example.")
