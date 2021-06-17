'''

There are a lot of ways to define a test case.
At the end of the day, you just need to provide a list of Agent objects to the environment.

- For simple testing of a particular configuration, consider defining a function like `get_testcase_two_agents`.
- If you want some final position configuration, consider something like `formation`.
- For a large, static test suite, consider creating a pickle file of [start_x, start_y, goal_x, goal_y, radius, pref_speed] tuples and use our code to convert that into a list of Agents, as in `preset_testCases`.

After defining a test case function that returns a list of Agents, you can select that test case fn in the evaluation code (see example.py)

'''

import numpy as np
import random
np.random.seed(1)
import sys
sys.path.append('/home/bdebrito/code/mpc-rl-collision-avoidance')
from gym_collision_avoidance.envs.agent import Agent
#from gym_collision_avoidance.envs.Obstacle import Obstacle
#from gym_collision_avoidance.envs.utils import DataHandlerLSTM
from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
#from gym_collision_avoidance.envs.policies.PedestrianDatasetPolicy import PedestrianDatasetPolicy
# from gym_collision_avoidance.envs.policies.DRLLongPolicy import DRLLongPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.CARRLPolicy import CARRLPolicy
from mpc_rl_collision_avoidance.policies.MPCPolicy import MPCPolicy
from mpc_rl_collision_avoidance.policies.MultiAgentMPCPolicy import MultiAgentMPCPolicy
from mpc_rl_collision_avoidance.policies.OtherAgentMPCPolicy import OtherAgentMPCPolicy
from mpc_rl_collision_avoidance.policies.SocialMPCPolicy import SocialMPCPolicy
from mpc_rl_collision_avoidance.policies.MPCRLPolicy import MPCRLPolicy
from mpc_rl_collision_avoidance.policies.LearningMPCPolicy import LearningMPCPolicy
#from mpc_rl_collision_avoidance.policies.ROSMPCPolicy import ROSMPCPolicy
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.dynamics.FirstOrderDynamics import FirstOrderDynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxTurnRate import UnicycleDynamicsMaxTurnRate
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxAcc import UnicycleDynamicsMaxAcc
from gym_collision_avoidance.envs.dynamics.UnicycleSecondOrderEulerDynamics import UnicycleSecondOrderEulerDynamics
from gym_collision_avoidance.envs.dynamics.ExternalDynamics import ExternalDynamics
from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
from gym_collision_avoidance.envs.sensors.AngularMapSensor import AngularMapSensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.utils import end_conditions as ec
#from gym_collision_avoidance.envs.dataset import Dataset

import os
import pickle
import random

from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import gen_rand_testcases as tc

test_case_filename = "{dir}/test_cases/{pref_speed_string}{num_agents}_agents_{num_test_cases}_cases.p"

policy_dict = {
    'rvo': RVOPolicy,
    'noncoop': NonCooperativePolicy,
    'carrl': CARRLPolicy,
    'external': ExternalPolicy,
    'GA3C': GA3CCADRLPolicy
}

policy_train_dict = {
    '0': RVOPolicy,
    '2': NonCooperativePolicy,
    '1': GA3CCADRLPolicy
}

def get_testcase_two_agents():
    goal_x = 3
    goal_y = 3
    agents = [
        Agent(-goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.5, LearningPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
        Agent(goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1)
        ]
    return agents

def get_testcase_two_agents_external_rvo():
    goal_x = 3
    goal_y = 0
    agents = [
        # Agent(0.735, -0.568, -0.254, 0.798, 0.567, 1.444, -2.313, CARRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
        # Agent(0.105, -1.83, 0.342, 1.935, 0.236, 1.17, 1.36, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1)
        Agent(-goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.0, CARRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
        Agent(goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, np.pi, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1)
        ]
    return agents

def get_testcase_two_agents_laserscanners():
    goal_x = 3
    goal_y = 3
    agents = [
        Agent(-goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.5, PPOPolicy, UnicycleDynamics, [LaserScanSensor], 0),
        Agent(goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, 0.5, PPOPolicy, UnicycleDynamics, [LaserScanSensor], 1)
        ]
    return agents

def get_testcase_random(num_agents=None, side_length=None, speed_bnds=None, radius_bnds=None, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[]):
    if num_agents is None:
        num_agents = np.random.randint(2, Config.MAX_NUM_AGENTS_IN_ENVIRONMENT+1)

    if side_length is None:
        side_length = 4

    if speed_bnds is None:
        speed_bnds = [0.5, 1.5]

    if radius_bnds is None:
        radius_bnds = [0.2, 0.8]

    cadrl_test_case = tc.generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds)

    agents = cadrl_test_case_to_agents(cadrl_test_case,
        agents_policy=agents_policy,
        agents_dynamics=agents_dynamics,
        agents_sensors=agents_sensors)
    return agents

def is_pose_valid(new_pose, position_list,distance=1.5):
    for pose in position_list:
        if np.linalg.norm(new_pose[:2] - pose[:2]) < distance:
            return False
    return True

def is_pose_valid_with_obstacles(new_pose, obstacles):
    if len(obstacles) == 0:
        return True
    marge = 1 # This is the distance that needs to be between the agent and the obstacle.
    for corners in obstacles:
        l1 = np.add(corners[1], [-marge, marge])  # left upper corner
        r1 = np.add(corners[3], [marge, -marge])  # right lower corner
        if new_pose[0] >= r1[0] or new_pose[1] >= l1[1] or new_pose[0] <= l1[0] or new_pose[1] <= r1[1]:
            value = 1
        else:
            value = 0
        if value == 0:
            return False
    return True

def is_shape_valid(new_corners, obstacle):
    '''
    Checks if shapes overlap
    If one of the 2 'if' conditions is true, the shapes DO NOT overlap
    No overlap = True
    Overlap = False
    '''
    if len(obstacle) == 0:
        return True
    for corners in obstacle:
        l1 = corners[1] # left upper corner
        r1 = corners[3] # right lower corner
        l2 = new_corners[1] # left upper corner
        r2 = new_corners[3] # left lower corner
        if l1[0] >= r2[0] or l2[0] >= r1[0] or l1[1] <= r2[1] or l2[1] <= r1[1]:
            value = 1
        else:
            value = 0
        if value == 0:
            return False
    return True

def go_to_goal(test_case_index, agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[]):
    pref_speed = 1.0
    radius = 0.5
    # Swap x-axis
    x0_agent_1 = np.random.uniform(-10, 10.0)
    y0_agent_1 = np.random.uniform(-10, 10.0)
    goal_x_1 = np.random.uniform(-10, 10.0)
    goal_y_1 = np.random.uniform(-10, 10.0)

    agents = [Agent(x0_agent_1, y0_agent_1, goal_x_1, goal_y_1, radius, pref_speed, None, agents_policy,
                    ExternalDynamics,[OtherAgentsStatesSensor], 0),
              Agent(goal_x_1, goal_y_1,x0_agent_1, y0_agent_1,  radius, pref_speed, None, RVOPolicy,
                    UnicycleDynamicsMaxAcc, [OtherAgentsStatesSensor], 1)
              ]
    return agents

def get_train_cases(test_case_index, agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[]):
    pref_speed = 1.0
    radius = 0.5
    # Swap x-axis
    if test_case_index == 0:
        x0_agent_1 = np.random.uniform(-10, 10.0)
        y0_agent_1 = np.random.uniform(-10, 10.0)
        goal_x_1 = np.random.uniform(-10, 10.0)
        goal_y_1 = np.random.uniform(-10, 10.0)
        if np.linalg.norm(np.array([goal_x_1, goal_y_1]) - np.array([x0_agent_1, y0_agent_1])) < 5.0:
            goal_x_1 = np.random.uniform(-10, 10.0)
            goal_y_1 = np.random.uniform(-10, 10.0)
        x0_agent_2 = goal_x_1
        y0_agent_2 = goal_y_1
        goal_x_2 = x0_agent_1
        goal_y_2 = y0_agent_1
    # Swap y-axis
    elif test_case_index == 1:
        x0_agent_1 = np.random.normal(0, 1.0)
        y0_agent_1 = np.random.normal(-10, 1.0)
        goal_x_1 = np.random.normal(0, 1.0)
        goal_y_1 = np.random.normal(10.0, 1.0)
        x0_agent_2 = np.random.normal(0, 1.0)
        y0_agent_2 = np.random.normal(10, 1.0)
        goal_x_2 = np.random.normal(0, 1.0)
        goal_y_2 = np.random.normal(-10.0, 1.0)
    # Move behind
    elif test_case_index == 2:
        x0_agent_1 = np.random.uniform(-10.0, 10.0)
        y0_agent_1 = np.random.uniform(-10.0, 10.0)
        goal_x_1 = np.random.normal(10, 1.0)
        goal_y_1 = np.random.normal(0.0, 1.0)
        if np.linalg.norm(np.array([goal_x_1, goal_y_1]) - np.array([x0_agent_1, y0_agent_1])) < 5.0:
            goal_x_1 = np.random.normal(10, 1.0)
            goal_y_1 = np.random.normal(0.0, 1.0)
        x0_agent_2 = goal_x_1
        y0_agent_2 = goal_y_1
        goal_x_2 = 2.0 * goal_x_1 - x0_agent_1
        goal_y_2 = 2.0 * goal_y_1 - y0_agent_1
    else:
        x0_agent_1 = np.random.normal(0, 5.0)
        y0_agent_1 = np.random.normal(0, 5.0)
        goal_x_1 = np.random.normal(0, 5.0)
        goal_y_1 = np.random.normal(0.0, 5.0)
        if np.linalg.norm(np.array([goal_x_1, goal_y_1]) - np.array([x0_agent_1, y0_agent_1])) < 5.0:
            goal_x_1 = np.random.normal(0, 5.0)
            goal_y_1 = np.random.normal(0.0, 5.0)
        x0_agent_2 = np.random.normal(x0_agent_1, 4.0)
        y0_agent_2 = np.random.normal(y0_agent_1, 4.0)
        # If the goal is the same random sample again
        if np.linalg.norm(np.array([x0_agent_2, y0_agent_2]) - np.array([x0_agent_1, y0_agent_1])) < 1.0:
            x0_agent_2 = np.random.normal(x0_agent_1, 4.0)
            y0_agent_2 = np.random.normal(y0_agent_1, 4.0)
        goal_x_2 = np.random.normal(goal_x_1, 4.0)
        goal_y_2 = np.random.normal(goal_y_1, 4.0)
        # If the goal is the same random sample again
        if np.linalg.norm(np.array([goal_x_2, goal_y_2]) - np.array([goal_x_1, goal_y_1])) < 1.0:
            goal_x_2 = np.random.normal(goal_x_1, 4.0)
            goal_y_2 = np.random.normal(goal_y_1, 4.0)

    agents = [Agent(x0_agent_1, y0_agent_1, goal_x_1, goal_y_1, radius, pref_speed, None, agents_policy,
                    UnicycleDynamicsMaxAcc,
                    [OtherAgentsStatesSensor], 0),
              Agent(x0_agent_2, y0_agent_2, goal_x_2, goal_y_2, radius, pref_speed, None, RVOPolicy,
                    UnicycleDynamicsMaxAcc,
                    [OtherAgentsStatesSensor], 1)
              ]
    return agents

def get_traincase_2agents_swap(test_case_index, num_test_cases=10, agents_policy=LearningPolicy, agents_dynamics=ExternalDynamics, agents_sensors=[]):
    pref_speed = 1.0
    radius = 0.5
    # swap random point in space
    if test_case_index == 0:
        x0_agent_1 = np.random.uniform(-7, 7.0)
        y0_agent_1 = np.random.uniform(-7, 7.0)
        goal_x_1 = np.random.uniform(-7, 7.0)
        goal_y_1 = np.random.uniform(-7, 7.0)
        while np.linalg.norm(np.array([goal_x_1, goal_y_1]) - np.array([x0_agent_1, y0_agent_1])) < 7.0:
            goal_x_1 = np.random.uniform(-7, 7.0)
            goal_y_1 = np.random.uniform(-7, 7.0)
        x0_agent_2 = goal_x_1
        y0_agent_2 = goal_y_1
        goal_x_2 = x0_agent_1
        goal_y_2 = y0_agent_1
    # Crossing
    elif test_case_index == 1:
        x0_agent_1 = np.random.normal(-7, 1.0)
        y0_agent_1 = np.random.normal(0, 1.0)
        goal_x_1 = np.random.normal(7.0, 1.0)
        goal_y_1 = np.random.normal(0.0, 1.0)
        x0_agent_2 = np.random.normal(0.0, 1.0)
        y0_agent_2 = np.random.normal(-7.0, 1.0)
        goal_x_2 = np.random.normal(0.0, 1.0)
        goal_y_2 = np.random.normal(7.0, 1.0)
        # Swap y-axis
    # Swap x-axis
    elif test_case_index == 2:
        x0_agent_1 = np.random.normal(-4, 1.0)
        y0_agent_1 = np.random.normal(0, 0.2)
        goal_x_1 = np.random.normal(4.0,1.0)
        goal_y_1 = np.random.normal(0.0,0.2)
        x0_agent_2 = np.random.normal(4.0, 1.0)
        y0_agent_2 = np.random.normal(0, 0.2)
        goal_x_2 = np.random.normal(-4.0, 1.0)
        goal_y_2 = np.random.normal(0.0, 0.2)
        # Swap y-axis
    elif test_case_index == 3:
        x0_agent_1 = np.random.normal(0, 0.2)
        y0_agent_1 = np.random.normal(-4, 1.0)
        goal_x_1 = np.random.normal(0, 0.2)
        goal_y_1 = np.random.normal(4.0, 1.0)
        x0_agent_2 = np.random.normal(0, 0.2)
        y0_agent_2 = np.random.normal(4, 1.0)
        goal_x_2 = np.random.normal(0, 0.2)
        goal_y_2 = np.random.normal(-4.0, 1.0)
    # Move behind
    elif test_case_index == 4:
        x0_agent_1 = np.random.uniform(-10.0, 10.0)
        y0_agent_1 = np.random.uniform(-10.0, 10.0)
        goal_x_1 = np.random.normal(10,1.0)
        goal_y_1 = np.random.normal(0.0,1.0)
        while np.linalg.norm(np.array([goal_x_1,goal_y_1])-np.array([x0_agent_1,y0_agent_1])) < 5.0:
            goal_x_1 = np.random.normal(10, 1.0)
            goal_y_1 = np.random.normal(0.0, 1.0)
        x0_agent_2 = goal_x_1
        y0_agent_2 = goal_y_1
        goal_x_2 = 2.0*goal_x_1 - x0_agent_1
        goal_y_2 = 2.0*goal_y_1 - y0_agent_1
    # AGent stoped in the middle of the path
    elif test_case_index == 5:
        x0_agent_1 = np.random.uniform(-10.0, 10.0)
        y0_agent_1 = np.random.uniform(-10.0, 10.0)
        goal_x_1 = np.random.normal(10,1.0)
        goal_y_1 = np.random.normal(0.0,1.0)
        while np.linalg.norm(np.array([goal_x_1,goal_y_1])-np.array([x0_agent_1,y0_agent_1])) < 5.0:
            goal_x_1 = np.random.normal(10, 1.0)
            goal_y_1 = np.random.normal(0.0, 1.0)
        x0_agent_2 = (goal_x_1 + x0_agent_1)/2.0
        y0_agent_2 = (goal_y_1 + y0_agent_1)/2.0
        goal_x_2 = x0_agent_2 +np.random.uniform(-1.0,1.0)
        goal_y_2 = y0_agent_2 +np.random.uniform(-1.0,1.0)
    # Random motion in space
    elif test_case_index == 6:
        x0_agent_1 = np.random.normal(0, 7.0)
        y0_agent_1 = np.random.normal(0, 7.0)
        goal_x_1 = np.random.normal(0, 7.0)
        goal_y_1 = np.random.normal(0.0, 7.0)

        x0_agent_2 = np.random.normal(x0_agent_1, 7.0)
        y0_agent_2 = np.random.normal(y0_agent_1, 7.0)
        # If the goal is the same random sample again
        while np.linalg.norm(np.array([x0_agent_2, y0_agent_2]) - np.array([x0_agent_1, y0_agent_1])) < 1.0:
            x0_agent_2 = np.random.normal(x0_agent_1, 7.0)
            y0_agent_2 = np.random.normal(y0_agent_1, 7.0)
        goal_x_2 = np.random.normal(goal_x_1, 7.0)
        goal_y_2 = np.random.normal(goal_y_1, 7.0)
        # If the goal is the same random sample again
        while np.linalg.norm(np.array([goal_x_2, goal_y_2]) - np.array([goal_x_1, goal_y_1])) < 1.0:
            goal_x_2 = np.random.normal(goal_x_1, 7.0)
            goal_y_2 = np.random.normal(goal_y_1, 7.0)
    else:
        x0_agent_1 = np.random.uniform(-5, 5.0)
        y0_agent_1 = np.random.uniform(-5, 5.0)
        goal_x_1 = np.random.uniform(-5, 5.0)
        goal_y_1 = np.random.uniform(-5, 5.0)
        while np.linalg.norm(np.array([goal_x_1, goal_y_1]) - np.array([x0_agent_1, y0_agent_1])) < 4.0:
            goal_x_1 = np.random.uniform(-5, 5.0)
            goal_y_1 = np.random.uniform(-5, 5.0)
        x0_agent_2 = goal_x_1
        y0_agent_2 = goal_y_1
        goal_x_2 = x0_agent_1
        goal_y_2 = y0_agent_1


    agents = [Agent(x0_agent_1, y0_agent_1,goal_x_1, goal_y_1, radius, pref_speed, None, RVOPolicy, UnicycleDynamicsMaxAcc,
                  [OtherAgentsStatesSensor], 0),
              Agent(x0_agent_2, y0_agent_2, goal_x_2, goal_y_2, radius, pref_speed, None, agents_policy, UnicycleDynamicsMaxAcc,
                    [OtherAgentsStatesSensor], 1)
        ]
    return agents

def get_testcase_2agents_swap(test_case_index, num_test_cases=10, agents_policy=LearningPolicy, agents_dynamics=ExternalDynamics, agents_sensors=[]):
    pref_speed = np.random.uniform(0.9, 1.2)
    radius = np.random.uniform(0.4, 0.6)

    # swap random point in space
    if test_case_index == 0:
        x0_agent_1 = np.random.uniform(-7, 7.0)
        y0_agent_1 = np.random.uniform(-7, 7.0)
        goal_x_1 = np.random.uniform(-7, 7.0)
        goal_y_1 = np.random.uniform(-7, 7.0)
        while np.linalg.norm(np.array([goal_x_1, goal_y_1]) - np.array([x0_agent_1, y0_agent_1])) < 7.0:
            goal_x_1 = np.random.uniform(-7, 7.0)
            goal_y_1 = np.random.uniform(-7, 7.0)
        x0_agent_2 = goal_x_1
        y0_agent_2 = goal_y_1
        goal_x_2 = x0_agent_1
        goal_y_2 = y0_agent_1
    # Crossing
    elif test_case_index == 1:
        x0_agent_1 = np.random.normal(-7, 1.0)
        y0_agent_1 = np.random.normal(0, 1.0)
        goal_x_1 = np.random.normal(7.0, 1.0)
        goal_y_1 = np.random.normal(0.0, 1.0)
        x0_agent_2 = np.random.normal(0.0, 1.0)
        y0_agent_2 = np.random.normal(-7.0, 1.0)
        goal_x_2 = np.random.normal(0.0, 1.0)
        goal_y_2 = np.random.normal(7.0, 1.0)
        # Swap y-axis
    # Swap x-axis
    elif test_case_index == 2:
        x0_agent_1 = np.random.normal(-4, 1.0)
        y0_agent_1 = np.random.normal(0, 0.2)
        goal_x_1 = np.random.normal(4.0,1.0)
        goal_y_1 = np.random.normal(0.0,0.2)
        x0_agent_2 = np.random.normal(4.0, 1.0)
        y0_agent_2 = np.random.normal(0, 0.2)
        goal_x_2 = np.random.normal(-4.0, 1.0)
        goal_y_2 = np.random.normal(0.0, 0.2)
        # Swap y-axis
    elif test_case_index == 3:
        x0_agent_1 = np.random.normal(0, 0.2)
        y0_agent_1 = np.random.normal(-4, 1.0)
        goal_x_1 = np.random.normal(0, 0.2)
        goal_y_1 = np.random.normal(4.0, 1.0)
        x0_agent_2 = np.random.normal(0, 0.2)
        y0_agent_2 = np.random.normal(4, 1.0)
        goal_x_2 = np.random.normal(0, 0.2)
        goal_y_2 = np.random.normal(-4.0, 1.0)
    # Move behind
    elif test_case_index == 4:
        x0_agent_1 = np.random.uniform(-10.0, 10.0)
        y0_agent_1 = np.random.uniform(-10.0, 10.0)
        goal_x_1 = np.random.normal(10,1.0)
        goal_y_1 = np.random.normal(0.0,1.0)
        while np.linalg.norm(np.array([goal_x_1,goal_y_1])-np.array([x0_agent_1,y0_agent_1])) < 5.0:
            goal_x_1 = np.random.normal(10, 1.0)
            goal_y_1 = np.random.normal(0.0, 1.0)
        x0_agent_2 = goal_x_1
        y0_agent_2 = goal_y_1
        goal_x_2 = 2.0*goal_x_1 - x0_agent_1
        goal_y_2 = 2.0*goal_y_1 - y0_agent_1
    # AGent stoped in the middle of the path
    elif test_case_index == 5:
        x0_agent_1 = np.random.uniform(-10.0, 10.0)
        y0_agent_1 = np.random.uniform(-10.0, 10.0)
        goal_x_1 = np.random.normal(10,1.0)
        goal_y_1 = np.random.normal(0.0,1.0)
        while np.linalg.norm(np.array([goal_x_1,goal_y_1])-np.array([x0_agent_1,y0_agent_1])) < 5.0:
            goal_x_1 = np.random.normal(10, 1.0)
            goal_y_1 = np.random.normal(0.0, 1.0)
        x0_agent_2 = (goal_x_1 + x0_agent_1)/2.0
        y0_agent_2 = (goal_y_1 + y0_agent_1)/2.0
        goal_x_2 = x0_agent_2 +np.random.uniform(-1.0,1.0)
        goal_y_2 = y0_agent_2 +np.random.uniform(-1.0,1.0)
    # Random motion in space
    elif test_case_index == 6:
        x0_agent_1 = np.random.normal(0, 7.0)
        y0_agent_1 = np.random.normal(0, 7.0)
        goal_x_1 = np.random.normal(0, 7.0)
        goal_y_1 = np.random.normal(0.0, 7.0)

        x0_agent_2 = np.random.normal(x0_agent_1, 7.0)
        y0_agent_2 = np.random.normal(y0_agent_1, 7.0)
        # If the goal is the same random sample again
        while np.linalg.norm(np.array([x0_agent_2, y0_agent_2]) - np.array([x0_agent_1, y0_agent_1])) < 1.0:
            x0_agent_2 = np.random.normal(x0_agent_1, 7.0)
            y0_agent_2 = np.random.normal(y0_agent_1, 7.0)
        goal_x_2 = np.random.normal(goal_x_1, 7.0)
        goal_y_2 = np.random.normal(goal_y_1, 7.0)
        # If the goal is the same random sample again
        while np.linalg.norm(np.array([goal_x_2, goal_y_2]) - np.array([goal_x_1, goal_y_1])) < 1.0:
            goal_x_2 = np.random.normal(goal_x_1, 7.0)
            goal_y_2 = np.random.normal(goal_y_1, 7.0)
    else:
        x0_agent_1 = np.random.uniform(-5, 5.0)
        y0_agent_1 = np.random.uniform(-5, 5.0)
        goal_x_1 = np.random.uniform(-5, 5.0)
        goal_y_1 = np.random.uniform(-5, 5.0)
        while np.linalg.norm(np.array([goal_x_1, goal_y_1]) - np.array([x0_agent_1, y0_agent_1])) < 4.0:
            goal_x_1 = np.random.uniform(-5, 5.0)
            goal_y_1 = np.random.uniform(-5, 5.0)
        x0_agent_2 = goal_x_1
        y0_agent_2 = goal_y_1
        goal_x_2 = x0_agent_1
        goal_y_2 = y0_agent_1


    agents = [Agent(x0_agent_1, y0_agent_1,goal_x_1, goal_y_1, radius, pref_speed, None, RVOPolicy, UnicycleDynamicsMaxAcc,
                  [OtherAgentsStatesSensor], 0),
              Agent(x0_agent_2, y0_agent_2, goal_x_2, goal_y_2, radius, pref_speed, None, agents_policy, UnicycleDynamicsMaxAcc,
                    [OtherAgentsStatesSensor], 1)
        ]
    return agents

def agents_swap(number_of_agents=2, agents_policy=RVOPolicy, agents_dynamics=ExternalDynamics, agents_sensors=[],seed=0):
    pref_speed = 1.0
    radius = 0.5
    agents = []

    if seed:
        random.seed(seed)
        np.random.seed(seed)
    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }
    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }


    policies = [RVOPolicy] # GA3CCADRLPolicy NonCooperativePolicy
    positions_list = []
    for ag_id in range(number_of_agents):
        in_collision = False
        while not in_collision:
            distance = np.random.uniform(4.0, 6.0)
            angle = np.random.uniform(-np.pi, np.pi)
            x0_agent_1 = distance*np.cos(angle)
            y0_agent_1 = distance*np.sin(angle)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(goal, positions_list) or is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(number_of_agents):
        policy = random.choice(policies)  # RVOPolicy #
        cooperation_coef = 0.5
        distance = np.random.uniform(5.0, 6.0)
        angle = np.random.uniform(-np.pi, np.pi)
        x0_agent_1 = distance*np.cos(angle)
        y0_agent_1 = distance*np.sin(angle)
        goal_x_1 = -x0_agent_1
        goal_y_1 = -y0_agent_1
        agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius, pref_speed, None, agents_policy, UnicycleDynamicsMaxAcc,
                  [OtherAgentsStatesSensor], 2*ag_id,cooperation_coef))
        if str(policy) == 'NonCooperativePolicy':
            cooperation_coef = 0.0
        agents.append(
            Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed, None, policy, UnicycleDynamicsMaxAcc,
                  [OtherAgentsStatesSensor], 2*ag_id+1,cooperation_coef))
        if 'GA3CCADRLPolicy' in str(agents_policy):
            agents[-1].policy.initialize_network(**ga3c_params)

    return agents

def single_agents_swap(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamicsMaxAcc, agents_sensors=[],seed=None):
    print("single_agents_swap")
    pref_speed = 1.0 #np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        n_agents = np.maximum(number_of_agents, 2)
    else:
        n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }

    positions_list = []
    """
    positions_list.append(np.array([-2.5, 0]))
    positions_list.append(np.array([2.5, 0]))
    positions_list.append(np.array([0, -2.5]))
    positions_list.append(np.array([0, 2.5]))
    positions_list.append(np.array([-3.5, 3.5]))
    positions_list.append(np.array([3.5, -3.5]))
    positions_list.append(np.array([-3.5, -3.5]))
    positions_list.append(np.array([3.5, 3.5]))
    #positions_list.append(np.array([-4.5, -4.5]))
    #positions_list.append(np.array([4.5, 4.5]))
    """
    distance = np.random.uniform(4.0, 10.0)
    angle = np.random.uniform(-np.pi, np.pi)
    x0_agent_1 = distance * np.cos(angle)
    y0_agent_1 = distance * np.sin(angle)
    goal_x_1 = -x0_agent_1
    goal_y_1 = -y0_agent_1
    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2-1)):
        is_valid = False
        while not is_valid:
            distance = np.random.uniform(4.0, 10.0)
            angle = np.random.uniform(-np.pi, np.pi)
            x0_agent_1 = distance*np.cos(angle)
            y0_agent_1 = distance*np.sin(angle)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            is_valid = is_pose_valid(goal, positions_list) and is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2)):
        if ag_id == 0:
            agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                                    positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed,
                                    None, ego_agent_policy, UnicycleSecondOrderEulerDynamics,
                                    [OtherAgentsStatesSensor], 0))

            agents.append(Agent(positions_list[2 * ag_id][0], positions_list[2 * ag_id][1],
                                positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1], radius, pref_speed,
                                None, other_agents_policy, agents_dynamics,
                                [OtherAgentsStatesSensor], 2 * ag_id + 1))
        else:
            agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                                positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed,
                                None, other_agents_policy, agents_dynamics,
                                [OtherAgentsStatesSensor], 2*ag_id))
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius, pref_speed, None, other_agents_policy, agents_dynamics,
                              [OtherAgentsStatesSensor], 2*ag_id+1))

    return agents, []

def single_agents_random_swap(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamicsMaxAcc, agents_sensors=[],seed=None):
    print("single_agents_pairwise_swap")
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        n_agents = np.maximum(number_of_agents, 2)
    else:
        n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }


    positions_list = []
    x0_agent_1 = np.random.uniform(-10, 10)
    y0_agent_1 = np.random.uniform(-10, 10)

    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents -1):
        is_valid = False
        while not is_valid:
            x0_agent_1 = np.random.uniform(-10, 10)
            y0_agent_1 = np.random.uniform(-10, 10)
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            is_valid = is_pose_valid(initial_pose, positions_list,4.0)
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2)):
        if ag_id == 0:
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                    positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius, pref_speed,
                                    None, ego_agent_policy, UnicycleSecondOrderEulerDynamics,
                                    [OtherAgentsStatesSensor], 2*ag_id))
        else:
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius,
                                pref_speed, None, other_agents_policy, agents_dynamics,
                                [OtherAgentsStatesSensor], 2*ag_id))
        agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                            positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius,
                            pref_speed, None, other_agents_policy, agents_dynamics,
                            [OtherAgentsStatesSensor], 2*ag_id+1))

    return agents, []

def single_agents_random_positions(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamicsMaxAcc, agents_sensors=[],seed=None):

    print("single_agents_random_positions")

    pref_speed = 1.0
    radius = 0.5
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        n_agents = np.maximum(number_of_agents, 2)
    else:
        n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }


    init_positions_list = []
    goal_positions_list = []
    in_collision = False
    while not in_collision:
        x0_agent_1 = np.random.uniform(-10, 10)
        y0_agent_1 = np.random.uniform(-10, 10)
        goal_x_1 = np.random.uniform(-10, 10)
        goal_y_1 = np.random.uniform(-10, 10)
        goal = np.array([goal_x_1, goal_y_1])
        initial_pose = np.array([x0_agent_1, y0_agent_1])
        in_collision = is_pose_valid(initial_pose, [goal],4.0)

    goal_positions_list.append(np.array([goal_x_1, goal_y_1]))
    init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents-1):
        is_valid = False
        while not is_valid:
            x0_agent_1 = np.random.uniform(-10, 10)
            y0_agent_1 = np.random.uniform(-10, 10)
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            is_valid_1 = is_pose_valid(initial_pose, init_positions_list)

            goal_x_1 = np.random.uniform(-10, 10)
            goal_y_1 = np.random.uniform(-10, 10)
            goal = np.array([goal_x_1, goal_y_1])
            is_valid_2 = is_pose_valid(goal, goal_positions_list)

            is_valid_3 = is_pose_valid(goal, [initial_pose],3.0)
            is_valid = is_valid_1 and is_valid_2 and is_valid_3

        init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
        goal_positions_list.append(np.array([goal_x_1, goal_y_1]))

    for ag_id in range(n_agents):
        if ag_id == 0:
            agents.append(Agent(init_positions_list[ag_id][0], init_positions_list[ag_id][1],
                                    goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                                    None, ego_agent_policy, UnicycleSecondOrderEulerDynamics,
                                    [OtherAgentsStatesSensor], ag_id))
        else:
            agents.append(Agent(init_positions_list[ag_id][0], init_positions_list[ag_id][1],
                                    goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed, None,other_agents_policy , agents_dynamics,
                                    [OtherAgentsStatesSensor], ag_id))
    return agents, []

def single_corridor_scenario(number_of_agents=5, ego_agent_policy=MPCPolicy,other_agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamicsMaxAcc, agents_sensors=[],seed=None):
    print("single_corridor_scenario")
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []

    positions_list = []
    side = [-1,1]
    horizontal = random.choice([True,False])

    if horizontal:
        x0_agent_1 = np.random.uniform(-10.0, -10.0)*random.choice(side)
        y0_agent_1 = np.random.uniform(-10.0, 10.0)
    else:
        y0_agent_1 = np.random.uniform(-10.0, -10.0)*random.choice(side)
        x0_agent_1 = np.random.uniform(-10.0, 10.0)

    goal_x_1 = -x0_agent_1
    goal_y_1 = y0_agent_1
    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    if seed:
        random.seed(seed)
        np.random.seed(seed)
        n_agents = np.maximum(number_of_agents, 2)
    else:
        n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }

    for ag_id in range(int(n_agents/2)-1):
        in_collision = False
        while not in_collision:
            if horizontal:
                x0_agent_1 = np.random.uniform(-10.0, -10.0) * random.choice(side)
                y0_agent_1 = np.random.uniform(-10.0, 10.0)
            else:
                y0_agent_1 = np.random.uniform(-10.0, -10.0) * random.choice(side)
                x0_agent_1 = np.random.uniform(-10.0, 10.0)
            goal_x_1 = -x0_agent_1
            goal_y_1 = y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(goal, positions_list) or is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2)):
        if ag_id == 0:
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                    positions_list[2*ag_id+1][0], positions_list[2*ag_id+1 ][1], radius, pref_speed,
                                    None, ego_agent_policy, UnicycleSecondOrderEulerDynamics,
                                    [OtherAgentsStatesSensor], 2*ag_id))
        else:
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius,
                                pref_speed, None, other_agents_policy, agents_dynamics,
                                [OtherAgentsStatesSensor], 2 * ag_id))
        agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                                    positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed, None,other_agents_policy , agents_dynamics,
                                    [OtherAgentsStatesSensor], 2*ag_id+1))
    return agents, []

def homogeneous_agents_swap(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None):
    print("homogeneous_agents_swap")
    pref_speed = 1.0 #np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }
    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }

    positions_list = []
    """
    positions_list.append(np.array([-2.5, 0]))
    positions_list.append(np.array([2.5, 0]))
    positions_list.append(np.array([0, -2.5]))
    positions_list.append(np.array([0, 2.5]))
    positions_list.append(np.array([-3.5, 3.5]))
    positions_list.append(np.array([3.5, -3.5]))
    positions_list.append(np.array([-3.5, -3.5]))
    positions_list.append(np.array([3.5, 3.5]))
    #positions_list.append(np.array([-4.5, -4.5]))
    #positions_list.append(np.array([4.5, 4.5]))
    """
    distance = np.random.uniform(4.0, 8.0)
    angle = np.random.uniform(-np.pi, np.pi)
    x0_agent_1 = distance * np.cos(angle)
    y0_agent_1 = distance * np.sin(angle)
    goal_x_1 = -x0_agent_1
    goal_y_1 = -y0_agent_1
    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    n_agents = random.randint(0,np.maximum(number_of_agents-1,0))
    if not seed:
        n_agents = number_of_agents - 1

    #n_agents = number_of_agents - 1
    for ag_id in range(n_agents):
        in_collision = False
        while not in_collision:
            #distance = np.random.uniform(4.0, 6.0)
            angle = np.random.uniform(-np.pi, np.pi)
            x0_agent_1 = distance*np.cos(angle)
            y0_agent_1 = distance*np.sin(angle)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(goal, positions_list) or is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    #n_agents = number_of_agents-1
    for ag_id in range(n_agents+1):
        if 'GA3CCADRLPolicy' in str(ego_agent_policy):
            agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                                positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed,
                                None, 'GA3CCADRLPolicy', UnicycleDynamicsMaxAcc,
                                [OtherAgentsStatesSensor], 2*ag_id))
            #agents[2*ag_id].policy.initialize_network(**ga3c_params)
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id + 1][0], positions_list[2*ag_id + 1][1], radius, pref_speed,
                                None, 'GA3CCADRLPolicy', UnicycleDynamicsMaxAcc,
                                [OtherAgentsStatesSensor], 2*ag_id+1))
            #agents[2*ag_id+1].policy.initialize_network(**ga3c_params)
        else:
            agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                                positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed,
                                None, ego_agent_policy, agents_dynamics,
                                [OtherAgentsStatesSensor], 2*ag_id))
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius, pref_speed, None, ego_agent_policy, agents_dynamics,
                          [OtherAgentsStatesSensor], 2*ag_id+1))

    return agents, []

def homogeneous_agents_pairwise_swap(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None):
    print("homogeneous_agents_pairwise_swap")
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }


    init_positions_list = []
    x0_agent_1 = np.random.uniform(-7.5, 7.5)
    y0_agent_1 = np.random.uniform(-7.5, 7.5)

    init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    n_agents = random.randint(1, np.maximum(number_of_agents, 1))

    for ag_id in range(2* n_agents -1):
        in_collision = False
        while not in_collision:
            x0_agent_1 = np.random.uniform(-7.5, 7.5)
            y0_agent_1 = np.random.uniform(-7.5, 7.5)
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(initial_pose, init_positions_list,4.0)
        init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents):

            if 'GA3CCADRLPolicy' in str(ego_agent_policy):
                agents.append(Agent(init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1],
                                    init_positions_list[2*ag_id + 1][0], init_positions_list[2*ag_id + 1][1], radius, pref_speed,
                                    None, 'GA3CCADRLPolicy', UnicycleDynamicsMaxAcc,
                                    [OtherAgentsStatesSensor], 2 * ag_id))
                #agents[2*ag_id].policy.initialize_network(**ga3c_params)
                agents.append(Agent(init_positions_list[2*ag_id+1][0], init_positions_list[2*ag_id+1][1],
                                    init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1], radius, pref_speed,
                                    None, 'GA3CCADRLPolicy', UnicycleDynamicsMaxAcc,
                                    [OtherAgentsStatesSensor], 2 * ag_id+1))
                #agents[2*ag_id+1].policy.initialize_network(**ga3c_params)
            else:
                agents.append(Agent(init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1],
                                    init_positions_list[2*ag_id + 1][0], init_positions_list[2*ag_id + 1][1], radius, pref_speed,
                                    None, ego_agent_policy, agents_dynamics,
                                    [OtherAgentsStatesSensor], 2 * ag_id))
                agents.append(
                    Agent(init_positions_list[2 * ag_id + 1][0], init_positions_list[2 * ag_id + 1][1],
                          init_positions_list[2 * ag_id][0], init_positions_list[2 * ag_id][1], radius, pref_speed,
                          None, ego_agent_policy, UnicycleDynamics,
                          [OtherAgentsStatesSensor], 2 * ag_id+1))

    return agents, []

def homogeneous_agents_random_positions(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None):

    print("homogeneous_agents_random_positions")

    pref_speed = 1.0
    radius = 0.5
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }


    init_positions_list = []
    goal_positions_list = []
    in_collision = False
    while not in_collision:
        x0_agent_1 = np.random.uniform(-7.5, 7.5)
        y0_agent_1 = np.random.uniform(-7.5, 7.5)
        goal_x_1 = np.random.uniform(-7.5, 7.5)
        goal_y_1 = np.random.uniform(-7.5, 7.5)
        goal = np.array([goal_x_1, goal_y_1])
        initial_pose = np.array([x0_agent_1, y0_agent_1])
        in_collision = is_pose_valid(initial_pose, [goal],4.0)

    goal_positions_list.append(np.array([goal_x_1, goal_y_1]))
    init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    n_agents = random.randint(1, np.maximum(number_of_agents, 1))

    for ag_id in range(n_agents*2-1):
        is_valid = False
        while not is_valid:
            x0_agent_1 = np.random.uniform(-7.5, 7.5)
            y0_agent_1 = np.random.uniform(-7.5, 7.5)
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            is_valid_1 = is_pose_valid(initial_pose, init_positions_list)

            goal_x_1 = np.random.uniform(-7.5, 7.5)
            goal_y_1 = np.random.uniform(-7.5, 7.5)
            goal = np.array([goal_x_1, goal_y_1])
            is_valid_2 = is_pose_valid(goal, goal_positions_list)

            is_valid_3 = is_pose_valid(goal, [initial_pose],3.0)
            is_valid = is_valid_1 and is_valid_2 and is_valid_3

        init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
        goal_positions_list.append(np.array([goal_x_1, goal_y_1]))

    for ag_id in range(n_agents):

        if 'GA3CCADRLPolicy' in str(ego_agent_policy):
            agents.append(Agent(init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1],
                                goal_positions_list[2*ag_id][0], goal_positions_list[2*ag_id][1], radius, pref_speed,
                                None, 'GA3CCADRLPolicy', UnicycleDynamicsMaxAcc,
                                [OtherAgentsStatesSensor], 2*ag_id))
            #agents[2*ag_id].policy.initialize_network(**ga3c_params)
            agents.append(Agent(init_positions_list[2*ag_id+1][0], init_positions_list[2*ag_id+1][1],
                                goal_positions_list[2*ag_id+1][0], goal_positions_list[2*ag_id+1][1], radius, pref_speed,
                                None, 'GA3CCADRLPolicy', UnicycleDynamicsMaxAcc,
                                [OtherAgentsStatesSensor], 2*ag_id+1))
            #agents[2*ag_id+1].policy.initialize_network(**ga3c_params)
        else:
            agents.append(Agent(init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1],
                                goal_positions_list[2*ag_id][0], goal_positions_list[2*ag_id][1], radius, pref_speed,
                                None, ego_agent_policy, agents_dynamics,
                                [OtherAgentsStatesSensor], 2*ag_id))
            agents.append(Agent(init_positions_list[2*ag_id+1][0], init_positions_list[2*ag_id+1][1],
                                goal_positions_list[2*ag_id+1][0], goal_positions_list[2*ag_id+1][1], radius, pref_speed, None,ego_agent_policy , UnicycleDynamics,
                                [OtherAgentsStatesSensor], 2*ag_id+1))
    return agents, []

def homogeneous_corridor_scenario(number_of_agents=5, ego_agent_policy=MPCPolicy,other_agents_policy=MPCPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None):
    print("homogeneous_corridor_scenario")
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []

    positions_list = []
    side = [-1,1]

    x0_agent_1 = np.random.uniform(-8.0, -6.0)*random.choice(side)
    y0_agent_1 = np.random.uniform(-4.0, 4.0)
    goal_x_1 = -x0_agent_1
    goal_y_1 = y0_agent_1
    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    n_agents = random.randint(0, np.maximum(number_of_agents - 1, 0))

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }

    for ag_id in range(n_agents):
        in_collision = False
        while not in_collision:
            x0_agent_1 = np.random.uniform(-8.0, -6.0) * random.choice(side)
            y0_agent_1 = np.random.uniform(-4.0, 4.0)
            goal_x_1 = -x0_agent_1
            goal_y_1 = y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(goal, positions_list) or is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents+1):
        if 'GA3CCADRLPolicy' in str(ego_agent_policy):
            agents.append(Agent(positions_list[2 * ag_id][0], positions_list[2 * ag_id][1],
                                positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1], radius, pref_speed,
                                None, 'GA3CCADRLPolicy', UnicycleDynamicsMaxAcc,
                                [OtherAgentsStatesSensor], 2 * ag_id))
            #agents[2 * ag_id].policy.initialize_network(**ga3c_params)
            agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                                positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed, None,'GA3CCADRLPolicy' , UnicycleDynamicsMaxAcc,
                                [OtherAgentsStatesSensor], 2*ag_id+1))
            #agents[2 * ag_id+1].policy.initialize_network(**ga3c_params)
        else:
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius, pref_speed, None, ego_agent_policy, UnicycleDynamics,
                      [OtherAgentsStatesSensor], 2*ag_id))
            agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                                positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed, None,ego_agent_policy , UnicycleDynamics,
                                [OtherAgentsStatesSensor], 2*ag_id+1))
    return agents, []

def change_side(number_of_agents=2, agents_policy=MPCPolicy, agents_dynamics=ExternalDynamics, agents_sensors=[],seed=None):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }
    """"""
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }

    positions_list = []

    y = np.random.uniform(-1.0, 1.0)
    x = np.random.uniform(-7.0, -5.0)
    x0_agent_1 = x
    y0_agent_1 = y
    goal_x_1 = -x0_agent_1
    goal_y_1 = y0_agent_1
    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    n_agents = random.randint(0,np.maximum(number_of_agents-1,0))
    if not seed:
        n_agents = number_of_agents - 1

    for ag_id in range(n_agents):
        in_collision = False
        while not in_collision:
            #distance = np.random.uniform(4.0, 6.0)
            delta = np.random.uniform(-10, 10)
            x0_agent_1 = x
            y0_agent_1 = y+delta
            goal_x_1 = -x
            goal_y_1 = y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(goal, positions_list) or is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents+1):
        if 'GA3CCADRLPolicy' in str(agents_policy):
            agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                                positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed,
                                None, agents_policy, UnicycleDynamicsMaxAcc,
                                [OtherAgentsStatesSensor], 0))
            agents[ag_id].policy.initialize_network(**ga3c_params)
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id + 1][0], positions_list[2*ag_id + 1][1], radius, pref_speed,
                                None, agents_policy, UnicycleDynamicsMaxAcc,
                                [OtherAgentsStatesSensor], 0))
            agents[ag_id].policy.initialize_network(**ga3c_params)
        else:
            agents.append(Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                                positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed,
                                None, agents_policy, agents_dynamics,
                                [OtherAgentsStatesSensor], 2*ag_id))
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius, pref_speed, None, agents_policy, agents_dynamics,
                          [OtherAgentsStatesSensor], 2*ag_id+1))

    return agents

def train_agents_swap_circle(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=[MPCPolicy], ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None):
    print("train_agents_swap_circle")
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    obstacle = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        n_agents = np.maximum(number_of_agents, 2)
    else:
        n_agents = number_of_agents #random.randint(2, np.maximum(number_of_agents, 2))

    positions_list = []
    #other_agents_policy = [RVOPolicy, NonCooperativePolicy]

    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }
    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }

    distance = np.random.uniform(4.0, 8.0)
    angle = np.random.uniform(-np.pi, np.pi)
    x0_agent_1 = distance * np.cos(angle)
    y0_agent_1 = distance * np.sin(angle)
    goal_x_1 = -x0_agent_1
    goal_y_1 = -y0_agent_1
    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2)-1):
        is_valid = False
        while not is_valid:
            distance = np.random.uniform(4.0, 8.0)
            angle = np.random.uniform(-np.pi, np.pi)
            x0_agent_1 = distance*np.cos(angle)
            y0_agent_1 = distance*np.sin(angle)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            is_valid = is_pose_valid(goal, positions_list) and is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2)):
        #policy = random.choice(other_agents_policy) #RVOPolicy #
        if np.random.uniform(0,1)>0.8:
           other_agents_policy = NonCooperativePolicy
        else:
           other_agents_policy = RVOPolicy
        #cooperation_coef = 0.5
        cooperation_coef = np.random.uniform(0.1, 1.0)
        if ag_id == 0:
            if 'GA3CCADRLPolicy' in str(ego_agent_policy):
                agents.append(Agent(positions_list[ag_id][0], positions_list[ag_id][1],
                                    positions_list[ag_id + 1][0], positions_list[ag_id + 1][1], radius, pref_speed,
                                    None, ego_agent_policy, UnicycleDynamicsMaxAcc,
                                    [OtherAgentsStatesSensor,OccupancyGridSensor], 0))
                agents[0].policy.initialize_network(**ga3c_params)
            else:
                agents.append(Agent(positions_list[0][0], positions_list[0][1],
                                    positions_list[1][0], positions_list[1][1], radius, pref_speed,
                                    None, ego_agent_policy, ego_agent_dynamics,
                                    [OtherAgentsStatesSensor,OccupancyGridSensor], 0))
        else:
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius, pref_speed, None, other_agents_policy, other_agents_dynamics,
                      [OtherAgentsStatesSensor,OccupancyGridSensor], 2*ag_id,cooperation_coef))
        cooperation_coef = np.random.uniform(0.1, 1.0)
        #policy = random.choice(other_agents_policy)  # RVOPolicy #
        if np.random.uniform(0,1)>0.8:
           other_agents_policy = NonCooperativePolicy
        else:
           other_agents_policy = RVOPolicy

        agents.append(
            Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                  positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed, None,other_agents_policy , other_agents_dynamics,
                  [OtherAgentsStatesSensor], 2*ag_id+1,cooperation_coef))

    return agents, obstacle

def train_agents_swap_circle_ewc(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=[MPCPolicy], ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None):
    print("train_agents_swap_circle_ewc")
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    obstacle = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    n_agents = number_of_agents#random.randint(2, np.maximum(number_of_agents, 2)) #number_of_agents

    positions_list = []
    #other_agents_policy = [RVOPolicy, NonCooperativePolicy]
    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }
    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }

    distance = np.random.uniform(4.0, 8.0)
    angle = np.random.uniform(-np.pi, np.pi)
    x0_agent_1 = distance * np.cos(angle)
    y0_agent_1 = distance * np.sin(angle)
    goal_x_1 = -x0_agent_1
    goal_y_1 = -y0_agent_1
    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2)-1):
        is_valid = False
        while not is_valid:
            distance = np.random.uniform(4.0, 8.0)
            angle = np.random.uniform(-np.pi, np.pi)
            x0_agent_1 = distance*np.cos(angle)
            y0_agent_1 = distance*np.sin(angle)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            is_valid = is_pose_valid(goal, positions_list) and is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2)):
        cooperation_coef = np.random.uniform(0.1, 1.0)
        if np.random.uniform(0,1)>0.8:
            other_agents_policy = NonCooperativePolicy
        else:
            other_agents_policy = RVOPolicy

        if ag_id == 0:
            if 'GA3CCADRLPolicy' in str(ego_agent_policy):
                agents.append(Agent(positions_list[ag_id][0], positions_list[ag_id][1],
                                    positions_list[ag_id + 1][0], positions_list[ag_id + 1][1], radius, pref_speed,
                                    None, ego_agent_policy, UnicycleDynamicsMaxAcc,
                                    [OtherAgentsStatesSensor], 0))
                agents[0].policy.initialize_network(**ga3c_params)
            else:
                agents.append(Agent(positions_list[0][0], positions_list[0][1],
                                    positions_list[1][0], positions_list[1][1], radius, pref_speed,
                                    None, ego_agent_policy, ego_agent_dynamics,
                                    [OtherAgentsStatesSensor], 0))
        else:
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius, pref_speed, None, other_agents_policy, other_agents_dynamics,
                      [OtherAgentsStatesSensor], 2*ag_id,cooperation_coef))
        cooperation_coef = np.random.uniform(0.1, 1.0)
        if np.random.uniform(0,1)>0.8:
            other_agents_policy = NonCooperativePolicy
        else:
            other_agents_policy = RVOPolicy
        agents.append(
            Agent(positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1],
                  positions_list[2 * ag_id][0], positions_list[2 * ag_id][1], radius, pref_speed, None, other_agents_policy,
                  other_agents_dynamics,
                  [OtherAgentsStatesSensor], 2 * ag_id + 1, cooperation_coef))

    return agents, obstacle

def train_agents_pairwise_swap(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=[MPCPolicy], ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None):
    print("train_agents_pairwise_swap")
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    n_agents = number_of_agents#random.randint(2, np.maximum(number_of_agents, 2))

    #other_agents_policy = [RVOPolicy, NonCooperativePolicy]

    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }
    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }

    init_positions_list = []
    x0_agent_1 = np.random.uniform(-7.5, 7.5)
    y0_agent_1 = np.random.uniform(-7.5, 7.5)

    init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents - 1):
        is_valid = False
        while not is_valid:
            x0_agent_1 = np.random.uniform(-7.5, 7.5)
            y0_agent_1 = np.random.uniform(-7.5, 7.5)
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            is_valid = is_pose_valid(initial_pose, init_positions_list,2.0)
        init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    random.shuffle(init_positions_list)

    for ag_id in range(int(n_agents/2)):
        #policy = random.choice(other_agents_policy) #RVOPolicy #
        if np.random.uniform(0,1)>0.8:
            policy = NonCooperativePolicy
        else:
            policy = RVOPolicy

        #cooperation_coef = 0.5
        cooperation_coef = np.random.uniform(0.1, 1.0)
        if ag_id == 0:
            if 'GA3CCADRLPolicy' in str(ego_agent_policy):
                agents.append(Agent(init_positions_list[ag_id][0], init_positions_list[ag_id][1],
                                    init_positions_list[ag_id + 1][0], init_positions_list[ag_id + 1][1], radius, pref_speed,
                                    None, ego_agent_policy, UnicycleDynamicsMaxAcc,
                                    [OtherAgentsStatesSensor], 0))
                agents[0].policy.initialize_network(**ga3c_params)
            else:
                agents.append(Agent(init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1],
                                    init_positions_list[2*ag_id + 1][0], init_positions_list[2*ag_id + 1][1], radius, pref_speed,
                                    None, ego_agent_policy, ego_agent_dynamics,
                                    [OtherAgentsStatesSensor], 0))
        else:
            agents.append(Agent(init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1],
                                init_positions_list[2*ag_id+1][0], init_positions_list[2*ag_id+1][1], radius, pref_speed, None, policy, other_agents_dynamics,
                      [OtherAgentsStatesSensor], 2*ag_id,cooperation_coef))
        cooperation_coef = np.random.uniform(0.1, 1.0)
        #policy = random.choice(other_agents_policy)  # RVOPolicy #
        if np.random.uniform(0,1)>0.8:
            policy = NonCooperativePolicy
        else:
            policy = RVOPolicy

        agents.append(
            Agent(init_positions_list[2*ag_id+1][0], init_positions_list[2*ag_id+1][1],
                  init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1], radius, pref_speed, None,policy , other_agents_dynamics,
                  [OtherAgentsStatesSensor], 2*ag_id+1,cooperation_coef))
    return agents, []

def train_agents_random_positions(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=[MPCPolicy], ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None):
    print("train_agents_random_positions")
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    n_agents = number_of_agents#random.randint(2, np.maximum(number_of_agents, 2))

    #other_agents_policy = [RVOPolicy, NonCooperativePolicy]

    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'IROS18',
         'checkpt_name': 'network_01900000'
         }
    """
    ga3c_params =  {
         'policy': GA3CCADRLPolicy,
         'checkpt_dir': 'ICRA21',
         'checkpt_name': 'network_01990000'
         }

    init_positions_list = []
    goal_positions_list = []
    is_valid = False
    while not is_valid:
        x0_agent_1 = np.random.uniform(-7.5, 7.5)
        y0_agent_1 = np.random.uniform(-7.5, 7.5)
        goal_x_1 = np.random.uniform(-7.5, 7.5)
        goal_y_1 = np.random.uniform(-7.5, 7.5)
        goal = np.array([goal_x_1, goal_y_1])
        initial_pose = np.array([x0_agent_1, y0_agent_1])
        is_valid = is_pose_valid(initial_pose, [goal],4.0)

    goal_positions_list.append(np.array([goal_x_1, goal_y_1]))
    init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents - 1):
        is_valid = False
        while not is_valid:
            x0_agent_1 = np.random.uniform(-7.5, 7.5)
            y0_agent_1 = np.random.uniform(-7.5, 7.5)
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            is_valid_1 = is_pose_valid(initial_pose, init_positions_list)

            goal_x_1 = np.random.uniform(-7.5, 7.5)
            goal_y_1 = np.random.uniform(-7.5, 7.5)
            goal = np.array([goal_x_1, goal_y_1])
            is_valid_2 = is_pose_valid(goal, goal_positions_list)

            is_valid_3 = is_pose_valid(goal, [initial_pose],4.0)
            is_valid = is_valid_1 and is_valid_2 and is_valid_3

        init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
        goal_positions_list.append(np.array([goal_x_1, goal_y_1]))

    for ag_id in range(int(n_agents/2)):
        #policy = random.choice(other_agents_policy) #RVOPolicy #
        if np.random.uniform(0,1)>0.8:
           policy = NonCooperativePolicy
        else:
           policy = RVOPolicy

        #cooperation_coef = 0.5
        cooperation_coef = np.random.uniform(0.1, 1.0)
        if ag_id == 0:
            if 'GA3CCADRLPolicy' in str(ego_agent_policy):
                agents.append(Agent(init_positions_list[ag_id][0], init_positions_list[ag_id][1],
                                    goal_positions_list[ag_id + 1][0], goal_positions_list[ag_id + 1][1], radius, pref_speed,
                                    None, ego_agent_policy, UnicycleDynamicsMaxAcc,
                                    [OtherAgentsStatesSensor], 0))
                agents[0].policy.initialize_network(**ga3c_params)
            else:
                agents.append(Agent(init_positions_list[ag_id][0], init_positions_list[ag_id][1],
                                    goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                                    None, ego_agent_policy, ego_agent_dynamics,
                                    [OtherAgentsStatesSensor], 0))
        else:
            agents.append(Agent(init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1],
                                goal_positions_list[2*ag_id][0], goal_positions_list[2*ag_id][1], radius, pref_speed, None, policy, other_agents_dynamics,
                      [OtherAgentsStatesSensor], 2*ag_id,cooperation_coef))
        cooperation_coef = np.random.uniform(0.1, 1.0)
        #policy = random.choice(other_agents_policy)  # RVOPolicy #
        if np.random.uniform(0,1)>0.8:
           policy = NonCooperativePolicy
        else:
           policy = RVOPolicy

        agents.append(
            Agent(init_positions_list[2*ag_id+1][0], init_positions_list[2*ag_id+1][1],
                  goal_positions_list[2*ag_id+1][0], goal_positions_list[2*ag_id+1][1], radius, pref_speed, None,policy , other_agents_dynamics,
                  [OtherAgentsStatesSensor], 2*ag_id+1,cooperation_coef))

    return agents, []

def pedestrian_scenario(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=[MPCPolicy], ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    obstacle = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        n_agents = np.maximum(number_of_agents, 2)
    else:
        n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    other_agents_policy = [RVOPolicy, NonCooperativePolicy,PedestrianDatasetPolicy]

    init_positions_list = []
    goal_positions_list = []
    is_valid = False
    while not is_valid:
        x0_agent_1 = np.random.uniform(-7.5, 7.5)
        y0_agent_1 = np.random.uniform(-7.5, 7.5)
        goal_x_1 = np.random.uniform(-7.5, 7.5)
        goal_y_1 = np.random.uniform(-7.5, 7.5)
        goal = np.array([goal_x_1, goal_y_1])
        initial_pose = np.array([x0_agent_1, y0_agent_1])
        is_valid = is_pose_valid(initial_pose, [goal], 4.0)

    goal_positions_list.append(np.array([goal_x_1, goal_y_1]))
    init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents - 1):
        is_valid = False
        while not is_valid:
            x0_agent_1 = np.random.uniform(-7.5, 7.5)
            y0_agent_1 = np.random.uniform(-7.5, 7.5)
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            is_valid_1 = is_pose_valid(initial_pose, init_positions_list)

            goal_x_1 = np.random.uniform(-7.5, 7.5)
            goal_y_1 = np.random.uniform(-7.5, 7.5)
            goal = np.array([goal_x_1, goal_y_1])
            is_valid_2 = is_pose_valid(goal, goal_positions_list)

            is_valid_3 = is_pose_valid(goal, [initial_pose], 4.0)
            is_valid = is_valid_1 and is_valid_2 and is_valid_3

        init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
        goal_positions_list.append(np.array([goal_x_1, goal_y_1]))

    for ag_id in range(int(n_agents/2)):
        policy = random.choice(other_agents_policy) #RVOPolicy #
        #if np.random.uniform(0,1)>0.8:
        #    policy = NonCooperativePolicy
        #else:
        #    policy = RVOPolicy
        cooperation_coef = 0.5
        #cooperation_coef = np.random.uniform(0.0, 1.0)
        if ag_id == 0:
                agents.append(Agent(init_positions_list[0][0], init_positions_list[0][1],
                                    goal_positions_list[0][0], goal_positions_list[0][1], radius, pref_speed,
                                    None, ego_agent_policy, ego_agent_dynamics,
                                    [OtherAgentsStatesSensor], 0))
        else:
            agents.append(Agent(init_positions_list[2*ag_id][0], init_positions_list[2*ag_id][1],
                                goal_positions_list[2*ag_id][0], goal_positions_list[2*ag_id][1], radius, pref_speed, None, policy, other_agents_dynamics,
                      [OtherAgentsStatesSensor], 2*ag_id,cooperation_coef))

        #cooperation_coef = np.random.uniform(0.0, 1.0)
        policy = random.choice(other_agents_policy)  # RVOPolicy #
        #if np.random.uniform(0,1)>0.8:
        #    policy = NonCooperativePolicy
        #else:
        #    policy = RVOPolicy

        agents.append(
            Agent(init_positions_list[2*ag_id+1][0], init_positions_list[2*ag_id+1][1],
                  goal_positions_list[2*ag_id+1][0], goal_positions_list[2*ag_id+1][1], radius, pref_speed, None,policy , other_agents_dynamics,
                  [OtherAgentsStatesSensor], 2*ag_id+1,cooperation_coef))

    return agents, obstacle

def dataset_scenario(number_of_agents=2, ego_agent_policy=MPCPolicy,other_agents_policy=[MPCPolicy], ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[],seed=None,dataset=None):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    obstacle = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        n_agents = np.maximum(number_of_agents, 2)
    else:
        n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    other_agents_policy = [RVOPolicy, NonCooperativePolicy]

    init_positions_list = []
    goal_positions_list = []

    traj = Dataset.data_prep.getRandomTrajectory()
    all_trajs = Dataset.data_prep.agent_container.getTrajectorySetForTime(traj[1].time_vec[0])

    # First select a number of random peds
    #n_of_agents = np.minimum(len(all_trajs),n_agents)
    # Pick agents from dataset

    for key in all_trajs:
        if len(init_positions_list) == 0:
            goal_positions_list.append(all_trajs[key].pose_vec[-1, :2])
            init_positions_list.append(all_trajs[key].pose_vec[0, :2])
        else:
            x0_agent_1 = all_trajs[key].pose_vec[0,0]
            y0_agent_1 = all_trajs[key].pose_vec[0,1]
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            is_valid_1 = is_pose_valid(initial_pose, init_positions_list)

            goal_x_1 = all_trajs[key].pose_vec[-1,0]
            goal_y_1 = all_trajs[key].pose_vec[-1,0]
            goal = np.array([goal_x_1, goal_y_1])
            is_valid_2 = is_pose_valid(goal, goal_positions_list)

            is_valid_3 = is_pose_valid(goal, [initial_pose], 1.0)
            is_valid = is_valid_1 and is_valid_2 and is_valid_3
            if is_valid:
                init_positions_list.append(np.array([x0_agent_1, y0_agent_1,key]))
                goal_positions_list.append(np.array([goal_x_1, goal_y_1,key]))

        if len(init_positions_list) == n_agents:
            break
    n_of_agents = len(init_positions_list)
    for _ in range(len(init_positions_list),n_agents):
        is_valid = False
        while not is_valid:
            x0_agent_1 = np.random.uniform(Dataset.data_prep.min_pos_x, Dataset.data_prep.max_pos_x) - Dataset.data_prep.mean_pos_x
            y0_agent_1 = np.random.uniform(Dataset.data_prep.min_pos_y, Dataset.data_prep.max_pos_y) - Dataset.data_prep.mean_pos_y
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            is_valid_1 = is_pose_valid(initial_pose, init_positions_list)

            goal_x_1 = np.random.uniform(Dataset.data_prep.min_pos_x, Dataset.data_prep.max_pos_x) - Dataset.data_prep.mean_pos_x
            goal_y_1 = np.random.uniform(Dataset.data_prep.min_pos_y, Dataset.data_prep.max_pos_y) - Dataset.data_prep.mean_pos_y
            goal = np.array([goal_x_1, goal_y_1])
            is_valid_2 = is_pose_valid(goal, goal_positions_list)

            is_valid_3 = is_pose_valid(goal, [initial_pose], 4.0)
            is_valid = is_valid_1 and is_valid_2 and is_valid_3

        init_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
        goal_positions_list.append(np.array([goal_x_1, goal_y_1]))

    # Ego Agent
    agents.append(Agent(init_positions_list[0][0], init_positions_list[0][1],
                        goal_positions_list[0][0], goal_positions_list[0][1], radius, pref_speed,
                        None, ego_agent_policy, ego_agent_dynamics,
                        [OtherAgentsStatesSensor], 0))

    cooperation_coef = 0.5
    for ag_id in range(1,n_of_agents):
        policy = PedestrianDatasetPolicy
        agents.append(Agent(init_positions_list[ag_id][0], init_positions_list[ag_id][1],
                            goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed, None,
                            policy, ExternalDynamics,
                            [OtherAgentsStatesSensor], ag_id, cooperation_coef))

        other_agent_id = ag_id
        agents[-1].policy.trajectory = all_trajs[init_positions_list[ag_id][2]]
        agents[-1].policy.agent_id = other_agent_id

    for ag_id in range(n_of_agents,n_agents):
        policy = random.choice(other_agents_policy) #RVOPolicy #
        #if np.random.uniform(0,1)>0.8:
        #    policy = NonCooperativePolicy
        #else:
        #    policy = RVOPolicy
        cooperation_coef = 0.5

        #cooperation_coef = np.random.uniform(0.0, 1.0)
        agents.append(Agent(init_positions_list[ag_id][0], init_positions_list[ag_id][1],
                            goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed, None, policy, other_agents_dynamics,
                      [OtherAgentsStatesSensor], ag_id,cooperation_coef))

    return agents, obstacle

def corridor_scenario(test_case_index, number_of_agents=5, agents_policy=MPCPolicy, agents_dynamics=UnicycleSecondOrderEulerDynamics, agents_sensors=[]):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []

    policies = [RVOPolicy, NonCooperativePolicy]
    positions_list = []
    side = [-1,1]

    x0_agent_1 = np.random.uniform(-8.0, -6.0)*random.choice(side)
    y0_agent_1 = np.random.uniform(-4.0, 4.0)
    goal_x_1 = -x0_agent_1
    goal_y_1 = y0_agent_1
    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    n_agents = random.randint(0, np.maximum(number_of_agents - 1, 0))
    # n_agents = number_of_agents-1
    #n_agents = number_of_agents - 1

    for ag_id in range(n_agents):
        in_collision = True
        while in_collision:
            x0_agent_1 = np.random.uniform(-8.0, -6.0) * random.choice(side)
            y0_agent_1 = np.random.uniform(-4.0, 4.0)
            goal_x_1 = -x0_agent_1
            goal_y_1 = y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(goal, positions_list) and is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents+1):
        policy = RVOPolicy #random.choice(policies) #
        cooperation_coef = 0.5
        cooperation_coef = np.random.uniform(0.0, 1.0)
        if ag_id == 0:
            agents.append(Agent(positions_list[ag_id][0], positions_list[ag_id][1],
                                positions_list[ag_id+1][0], positions_list[ag_id+1][1], radius, pref_speed, None, agents_policy, agents_dynamics,
                      [OtherAgentsStatesSensor], 0))
        else:
            agents.append(Agent(positions_list[2*ag_id][0], positions_list[2*ag_id][1],
                                positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1], radius, pref_speed, None, policy, UnicycleDynamicsMaxAcc,
                      [OtherAgentsStatesSensor], 2*ag_id,cooperation_coef))
        cooperation_coef = np.random.uniform(0.0, 1.0)
        policy = RVOPolicy #random.choice(policies)  # RVOPolicy #
        agents.append(
            Agent(positions_list[2*ag_id+1][0], positions_list[2*ag_id+1][1],
                  positions_list[2*ag_id][0], positions_list[2*ag_id][1], radius, pref_speed, None,policy , UnicycleDynamicsMaxAcc,
                  [OtherAgentsStatesSensor], 2*ag_id+1,cooperation_coef))
    return agents

def random_agents_swap(test_case_index, number_of_agents=2, agents_policy=LearningPolicy, agents_dynamics=ExternalDynamics, agents_sensors=[]):
    pref_speed = 1.0
    radius = 0.5
    agents = []

    for ag_id in range(number_of_agents):
        x0_agent_1 = np.random.uniform(-10, 10.0)
        y0_agent_1 = np.random.uniform(-10, 10.0)
        goal_x_1 = np.random.uniform(-10, 10.0)
        goal_y_1 = np.random.uniform(-10, 10.0)
        while np.linalg.norm(np.array([goal_x_1, goal_y_1]) - np.array([x0_agent_1, y0_agent_1])) < 7.0:
            goal_x_1 = np.random.uniform(-10, 10.0)
            goal_y_1 = np.random.uniform(-10, 10.0)

        agents.append(Agent(x0_agent_1, y0_agent_1,goal_x_1, goal_y_1, radius, pref_speed, None, RVOPolicy, UnicycleDynamics,
                  [OtherAgentsStatesSensor], 0))
        agents.append(
            Agent(goal_x_1, goal_y_1,x0_agent_1, y0_agent_1, radius, pref_speed, None, RVOPolicy, UnicycleDynamics,
                  [OtherAgentsStatesSensor], 0))
    return agents

def get_testcase_2agents_swap(test_case_index, num_test_cases=10, agents_policy=LearningPolicy, agents_dynamics=ExternalDynamics, agents_sensors=[]):
    pref_speed = 1.0
    radius = 0.5
    # Move alone
    if test_case_index == 0:
        x0_agent_1 = -10.0
        y0_agent_1 = 0
        goal_x_1 = 10
        goal_y_1 = 0
        x0_agent_2 = 20.0
        y0_agent_2 = 20.0
        goal_x_2 = 20.5
        goal_y_2 = 20.5
        pref_speed2 = pref_speed
    # Move behind
    elif test_case_index == 1:
        x0_agent_1 = -10.0
        y0_agent_1 = 0.0
        goal_x_1 = 10.0
        goal_y_1 = 0
        x0_agent_2 = -8.0
        y0_agent_2 = 0
        goal_x_2 = 12.0
        goal_y_2 = 0
        pref_speed2 = pref_speed
    # Agent Stopped in the Middle
    elif test_case_index == 2:
        x0_agent_1 = -10.0
        y0_agent_1 = 0.0
        goal_x_1 = 10.0
        goal_y_1 = 0.0
        x0_agent_2 = 0.5
        y0_agent_2 = 0.5
        goal_x_2 = 0.0
        goal_y_2 = 0.0
        pref_speed2 = pref_speed
    # swap
    else:
        x0_agent_1 = -10.0
        y0_agent_1 = 0.0
        goal_x_1 = 10.0
        goal_y_1 = 0.0
        x0_agent_2 = goal_x_1
        y0_agent_2 = goal_y_1
        goal_x_2 = x0_agent_1
        goal_y_2 = y0_agent_1
        pref_speed2 = pref_speed

    print(test_case_index)
    agents = [Agent(x0_agent_1, y0_agent_1,goal_x_1, goal_y_1, radius, pref_speed, None, RVOPolicy, UnicycleDynamicsMaxAcc,
                  [OtherAgentsStatesSensor], 0),
              Agent(x0_agent_2, y0_agent_2, goal_x_2, goal_y_2, radius, pref_speed2, None, agents_policy, UnicycleDynamicsMaxAcc,
                    [OtherAgentsStatesSensor], 1)
        ]
    return agents

def get_testcase_unit_tests(test_case_index, num_test_cases=10, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[]):
    pref_speed = 1.0
    radius = 0.5
    test_case = np.random.randint(0,4)
    # Moving in the same direction
    if test_case == 0:
        x0_agent_1 = -10
        y0_agent_1 = 0
        goal_x_1 = 10
        goal_y_1 = 0
        goal_x_2 = 15
        goal_y_2 = 0
        x0_agent_2 = -5
        y0_agent_2 = 0

    # Swap y-axis
    elif test_case == 1:
        x0_agent_1 = -10
        y0_agent_1 = 0
        goal_x_1 = 10
        goal_y_1 = 0
        goal_x_2 = 20
        goal_y_2 = 20
        x0_agent_2 = 20
        y0_agent_2 = 20

    # Move opposite directions
    elif test_case == 2:
        x0_agent_1 = -10
        y0_agent_1 = 0
        goal_x_1 = 10
        goal_y_1 = 0
        goal_x_2 = -10
        goal_y_2 = 0
        x0_agent_2 = 10
        y0_agent_2 = 0
    # crossing
    else:
        x0_agent_1 = -10
        y0_agent_1 = 0
        goal_x_1 = 10
        goal_y_1 = 0
        goal_x_2 = 0
        goal_y_2 = -10
        x0_agent_2 = 0
        y0_agent_2 = 10
    # Swap agents
    if test_case_index % 2 == 0:
        agents = [
            Agent(x0_agent_1, y0_agent_1, goal_x_1, goal_y_1, radius, pref_speed, None, agents_policy, agents_dynamics,
                  agents_sensors, 0),
            Agent(x0_agent_2, y0_agent_2, goal_x_2, goal_y_2, radius, pref_speed, None, agents_policy, agents_dynamics,
                  agents_sensors, 1)
        ]
    else:
        agents = [
            Agent(x0_agent_2, y0_agent_2, goal_x_2, goal_y_2, radius, pref_speed, None, agents_policy, agents_dynamics,
                  agents_sensors, 0),
            Agent(x0_agent_1, y0_agent_1, goal_x_1, goal_y_1, radius, pref_speed, None, agents_policy, agents_dynamics,
                  agents_sensors, 1)
        ]
    return agents

def get_testcase_easy():

    num_agents = 2
    side_length = 2
    speed_bnds = [0.5, 1.5]
    radius_bnds = [0.2, 0.8]

    test_case = tc.generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds)

    agents = cadrl_test_case_to_agents(test_case)
    return agents

def get_testcase_fixed_initial_conditions(agents):
    new_agents = []
    for agent in agents:
        goal_x, goal_y = get_new_goal(agent.pos_global_frame)
        new_agent = Agent(agent.pos_global_frame[0], agent.pos_global_frame[1], goal_x, goal_y, agent.radius, agent.pref_speed, agent.heading_global_frame, agent.policy.__class__, agent.dynamics_model.__class__, [], agent.id)
        new_agents.append(new_agent)
    return new_agents

def get_testcase_fixed_initial_conditions_for_non_ppo(agents):
    new_agents = []
    for agent in agents:
        if agent.policy.str == "PPO":
            start_x, start_y = get_new_start_pos()
        else:
            start_x, start_y = agent.pos_global_frame
        goal_x, goal_y = get_new_goal(agent.pos_global_frame)
        new_agent = Agent(start_x, start_y, goal_x, goal_y, agent.radius, agent.pref_speed, agent.heading_global_frame, agent.policy.__class__, agent.dynamics_model.__class__, [], agent.id)
        new_agents.append(new_agent)
    return new_agents

def get_new_goal(pos):
    bounds = np.array([[-5, 5], [-5, 5]])
    dist_from_pos_threshold = 4.
    far_from_pos = False
    while not far_from_pos:
        gx, gy = np.random.uniform(bounds[:,0], bounds[:,1])
        far_from_pos = np.linalg.norm(pos - np.array([gx, gy])) >= dist_from_pos_threshold
    return gx, gy

def small_test_suite(num_agents, test_case_index, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[], vpref_constraint=False, radius_bnds=None):
    cadrl_test_case = preset_testCases(num_agents)[test_case_index]
    agents = cadrl_test_case_to_agents(cadrl_test_case, agents_policy=agents_policy, agents_dynamics=agents_dynamics, agents_sensors=agents_sensors)
    return agents

def full_test_suite(num_agents, test_case_index, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[], vpref_constraint=False, radius_bounds=None):
    cadrl_test_case = preset_testCases(num_agents, full_test_suite=True, vpref_constraint=vpref_constraint, radius_bounds=radius_bounds)[test_case_index]
    agents = cadrl_test_case_to_agents(cadrl_test_case, agents_policy=agents_policy, agents_dynamics=agents_dynamics, agents_sensors=agents_sensors)
    return agents

def full_test_suite_carrl(num_agents, test_case_index, seed=None, other_agent_policy_options=None):
    cadrl_test_case = preset_testCases(num_agents, full_test_suite=True, vpref_constraint=False, radius_bounds=None, carrl=True, seed=seed)[test_case_index]
    agents = []

    if other_agent_policy_options is None:
        other_agent_policy_options = [RVOPolicy]
    else:
        other_agent_policy_options = [policy_dict[pol] for pol in other_agent_policy_options]
    other_agent_policy = other_agent_policy_options[test_case_index%len(other_agent_policy_options)] # dont just sample (inconsistency btwn same test_case)
    agents.append(cadrl_test_case_to_agents([cadrl_test_case[0,:]], agents_policy=CARRLPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[OtherAgentsStatesSensor])[0])
    agents.append(cadrl_test_case_to_agents([cadrl_test_case[1,:]], agents_policy=other_agent_policy, agents_dynamics=UnicycleDynamics, agents_sensors=[OtherAgentsStatesSensor])[0])
    agents[1].id = 1
    return agents

def get_testcase_random_carrl():
    num_agents = 2
    side_length = 2
    speed_bnds = [0.5, 1.5]
    radius_bnds = [0.2, 0.8]
    test_case = tc.generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds)
    agents = []
    agents.append(cadrl_test_case_to_agents([test_case[0,:]], agents_policy=CARRLPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[OtherAgentsStatesSensor])[0])
    agents.append(cadrl_test_case_to_agents([test_case[1,:]], agents_policy=RVOPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[OtherAgentsStatesSensor])[0])
    agents[1].id = 1
    return agents

def formation(agents, letter, num_agents=6, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[OtherAgentsStatesSensor]):
    formations = {
        'A': 2*np.array([
              [-1.5, 0.0], # A
              [1.5, 0.0],
              [0.75, 1.5],
              [-0.75, 1.5],
              [0.0, 1.5],
              [0.0, 3.0]
            ]),
        'C': 2*np.array([
              [0.0, 0.0], # C
              [-0.5, 1.0],
              [-0.5, 2.0],
              [0.0, 3.0],
              [1.5, 0.0],
              [1.5, 3.0]
              ]),
        'L': 2*np.array([
            [0.0, 0.0], # L
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.75, 0.0],
            [1.5, 0.0]
            ]),
        'D': 2*np.array([
            [0.0, 0.0],
            [0.0, 1.5],
            [0.0, 3.0],
            [1.5, 1.5],
            [1.2, 2.5],
            [1.2, 0.5],
            ]),
        'R': 2*np.array([
            [0.0, 0.0],
            [0.0, 1.5],
            [0.0, 3.0],
            [1.3, 2.8],
            [1.2, 1.7],
            [1.7, 0.0],
            ]),
    }

    agent_inds = np.arange(num_agents)
    np.random.shuffle(agent_inds)

    new_agents = []
    for agent in agents:
        start_x, start_y = agent.pos_global_frame
        goal_x, goal_y = formations[letter][agent_inds[agent.id]]
        new_agent = Agent(start_x, start_y, goal_x, goal_y, agent.radius, agent.pref_speed, agent.heading_global_frame, agents_policy, agents_dynamics, agents_sensors, agent.id)
        new_agents.append(new_agent)
    return new_agents

def cadrl_test_case_to_agents(test_case, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[]):
    ###############################
    # This function accepts a test_case in legacy cadrl format and converts it
    # into our new list of Agent objects. The legacy cadrl format is a list of
    # [start_x, start_y, goal_x, goal_y, pref_speed, radius] for each agent.
    ###############################

    agents = []
    policies = [NonCooperativePolicy, LearningPolicy, StaticPolicy]
    if Config.EVALUATE_MODE or Config.PLAY_MODE:
        agent_policy_list = [agents_policy for _ in range(np.shape(test_case)[0])]
    else:
        # Random mix of agents following various policies
        # agent_policy_list = np.random.choice(policies,
        #                                      np.shape(test_case)[0],
        #                                      p=[0.05, 0.9, 0.05])
        agent_policy_list = np.random.choice(policies,
                                             np.shape(test_case)[0],
                                             p=[0.0, 1.0, 0.0])

        # Make sure at least one agent is following PPO
        #  (otherwise waste of time...)
        if LearningPolicy not in agent_policy_list:
            random_agent_id = np.random.randint(len(agent_policy_list))
            agent_policy_list[random_agent_id] = LearningPolicy

    agent_dynamics_list = [agents_dynamics for _ in range(np.shape(test_case)[0])]
    agent_sensors_list = [agents_sensors for _ in range(np.shape(test_case)[0])]

    for i, agent in enumerate(test_case):
        px = agent[0]
        py = agent[1]
        gx = agent[2]
        gy = agent[3]
        pref_speed = agent[4]
        radius = agent[5]
        if Config.EVALUATE_MODE:
            # initial heading is pointed toward the goal
            vec_to_goal = np.array([gx, gy]) - np.array([px, py])
            heading = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        else:
            heading = np.random.uniform(-np.pi, np.pi)

        agents.append(Agent(px, py, gx, gy, radius, pref_speed, heading, agent_policy_list[i], agent_dynamics_list[i], agent_sensors_list[i], i))
    return agents

def preset_testCases(num_agents, full_test_suite=False, vpref_constraint=False, radius_bounds=None, carrl=False, seed=None):
    if full_test_suite:
        num_test_cases = 500

        if vpref_constraint:
            pref_speed_string = 'vpref1.0_r{}-{}/'.format(radius_bounds[0], radius_bounds[1])
        else:
            pref_speed_string = ''

        filename = test_case_filename.format(
                num_agents=num_agents, num_test_cases=num_test_cases, pref_speed_string=pref_speed_string,
                dir=os.path.dirname(os.path.realpath(__file__)))
        if carrl:
            filename = filename[:-2]+'_carrl'+filename[-2:]
        if seed is not None:
            filename = filename[:-2]+'_seed'+str(seed).zfill(3)+filename[-2:]
        test_cases = pickle.load(open(filename, "rb"), encoding='latin1')

    else:
        if num_agents == 1:
            test_cases = []
            # fixed speed and radius
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [3.0/1.4, -3.0/1.4, -3.0/1.4, 3.0/1.4, 1.0, 0.3]
                ]))

        elif num_agents == 2:
            test_cases = []
            # fixed speed and radius
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0/1.4, -3.0/1.4, -3.0/1.4, 3.0/1.4, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-2.0, -1.5, 2.0, 1.5, 1.0, 0.5],
                [-2.0, 1.5, 2.0, -1.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [0.0, -3.0, 0.0, 3.0, 1.0, 0.5]
                ]))
            # variable speed and radius
            test_cases.append(np.array([
                [-2.5, 0.0, 2.5, 0.0, 1.0, 0.3],
                [2.5, 0.0, -2.5, 0.0, 0.8, 0.4]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 0.6, 0.5],
                [3.0/1.4, -3.0/1.4, -3.0/1.4, 3.0/1.4, 1.0, 0.4]
                ]))
            test_cases.append(np.array([
                [-2.0, 0.0, 2.0, 0.0, 0.9, 0.35],
                [2.0, 0.0, -2.0, 0.0, 0.85, 0.45]
                ]))
            test_cases.append(np.array([
                [-4.0, 0.0, 4.0, 0.0, 1.0, 0.4],
                [-2.0, 0.0, 2.0, 0.0, 0.5, 0.4]
                ]))

        elif num_agents == 3 or num_agents == 4:
            test_cases = []
            # hardcoded to be 3 agents for now
            d = 3.0
            l1 = d*np.cos(np.pi/6)
            l2 = d*np.sin(np.pi/6)
            test_cases.append(np.array([
                [0.0, d, 0.0, -d, 1.0, 0.5],
                [l1, -l2, -l1, l2, 1.0, 0.5],
                [-l1, -l2, l1, l2, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, 1.5, 1.0, 0.5]
                ]))
            # hardcoded to be 4 agents for now
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.3],
                [3.0, -1.5, -3.0, -1.5, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.3],
                [3.0, -3.0, -3.0, -3.0, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [0.0, -3.0, 0.0, 3.0, 1.0, 0.5],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.5],
                [0.0, 3.0, 0.0, -3.0, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-2.0, -1.5, 2.0, 1.5, 1.0, 0.5],
                [-2.0, 1.5, 2.0, -1.5, 1.0, 0.5],
                [-2.0, -4.0, 2.0, -4.0, 0.9, 0.35],
                [2.0, -4.0, -2.0, -4.0, 0.85, 0.45]
                ]))
            test_cases.append(np.array([
                [-4.0, 0.0, 4.0, 0.0, 1.0, 0.4],
                [-2.0, 0.0, 2.0, 0.0, 0.5, 0.4],
                [-4.0, -4.0, 4.0, -4.0, 1.0, 0.4],
                [-2.0, -4.0, 2.0, -4.0, 0.5, 0.4]
                ]))

        elif num_agents == 5:
            test_cases = []

            radius = 4
            tc = gen_circle_test_case(num_agents, radius)
            test_cases.append(tc)

            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, 3.0, 3.0, 3.0, 1.0, 0.5],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.5]
                ]))

        elif num_agents == 6:
            test_cases = []

            radius = 5
            tc = gen_circle_test_case(num_agents, radius)
            test_cases.append(tc)

            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, 3.0, 3.0, 3.0, 1.0, 0.5],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.5],
                [-3.0, -4.5, 3.0, -4.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, 0.7, 3.0, 0.7, 1.0, 0.3],
                [3.0, 0.7, -3.0, 0.7, 1.0, 0.3],
                [-3.0, -0.7, 3.0, -0.7, 1.0, 0.3],
                [3.0, -0.7, -3.0, -0.7, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, 1.0, 3.0, 1.0, 1.0, 0.3],
                [3.0, 1.0, -3.0, 1.0, 1.0, 0.3],
                [-3.0, -1.0, 3.0, -1.0, 1.0, 0.3],
                [3.0, -1.0, -3.0, -1.0, 1.0, 0.3]
                ]))

        elif num_agents == 10:
            test_cases = []

            radius = 5
            tc = gen_circle_test_case(num_agents, radius)
            test_cases.append(tc)

        elif num_agents == 20:
            test_cases = []

            radius = 10
            tc = gen_circle_test_case(num_agents, radius)
            test_cases.append(tc)

        else:
            print("[preset_testCases in Collision_Avoidance.py]\
                    invalid num_agents")
            assert(0)
    return test_cases

def gen_circle_test_case(num_agents, radius):
    tc = np.zeros((num_agents, 6))
    for i in range(num_agents):
        tc[i, 4] = 1.0
        tc[i, 5] = 0.5
        theta_start = (2*np.pi/num_agents)*i
        theta_end = theta_start + np.pi
        tc[i, 0] = radius*np.cos(theta_start)
        tc[i, 1] = radius*np.sin(theta_start)
        tc[i, 2] = radius*np.cos(theta_end)
        tc[i, 3] = radius*np.sin(theta_end)
    return tc

def get_testcase_hololens_and_ga3c_cadrl():
    goal_x1 = 3
    goal_y1 = 3
    goal_x2 = 2
    goal_y2 = 5
    agents = [
              Agent(-goal_x1, goal_y1, goal_x1, -goal_y1, 0.5, 1.0, 0.5, ExternalPolicy, ExternalDynamics, [], 0), # hololens
              Agent(goal_x1, goal_y1, goal_x1, -goal_y1, 0.5, 1.0, 0.5, ExternalPolicy, ExternalDynamics, [], 1), # real robot
              Agent(-goal_x1+np.random.uniform(-3,3), -goal_y1+np.random.uniform(-1,1), goal_x1, goal_y1, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 2)
              ]
              # Agent(goal_x1, goal_y1, -goal_x1, -goal_y1, 0.5, 2.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 1),
              # Agent(-goal_x2, -goal_y2, goal_x2, goal_y2, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 2),
              # Agent(goal_x2, goal_y2, -goal_x2, -goal_y2, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 3),
              # Agent(-goal_x2, goal_y2, goal_x2, -goal_y2, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 4),
              # Agent(-goal_x1, goal_y1, goal_x1, -goal_y1, 0.5, 1.0, 0.5, ExternalPolicy, ExternalDynamics, [], 5)]
    return agents

def agent_with_obstacle(number_of_agents=1, ego_agent_policy=MPCPolicy,other_agents_policy=[RVOPolicy], ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    '''
    In this scenario there is an obstacle in the middle and there is 1 agent that needs to cross the room, avoiding the obstacle
    There will always be 1 other agent in the environment
    '''
    pref_speed = 1.0
    radius = 0.5
    agents = []

    #Add obstacle in the middle
    obstacle = []
    #Square
    obstacle_1 = [(2,2), (0,2), (0,0), (2,0)]

    #Triangle
    #obstacle_1 = [(0, 2), (-3, -2), (3, -2)]
    obstacle.append(obstacle_1)

    distance = np.random.uniform(6.0, 8.0)
    angle = np.random.uniform(-np.pi, np.pi)
    x0_agent_1 = distance * np.cos(angle)
    y0_agent_1 = distance * np.sin(angle)
    goal_x_1 = -x0_agent_1
    goal_y_1 = -y0_agent_1

    agents.append(Agent(x0_agent_1, y0_agent_1, goal_x_1, goal_y_1, radius, pref_speed, None, ego_agent_policy,
                        ego_agent_dynamics,
                        [OtherAgentsStatesSensor,LaserScanSensor], 0))
    agents.append(Agent(goal_x_1, goal_y_1, x0_agent_1, y0_agent_1, radius, pref_speed, None, other_agents_policy,
                        other_agents_dynamics,
                        [OtherAgentsStatesSensor], 1))

    if "Static" in str(agents[0].policy):
        #agents[0].sensors[1].static_obstacles_manager.obstacle = obstacle
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def test_agent_with_obstacle(number_of_agents=1, ego_agent_policy=MPCPolicy,other_agents_policy=RVOPolicy, ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    '''
    In this scenario there is an obstacle in the middle and there is 1 agent that needs to cross the room, avoiding the obstacle
    The number_of_agents include the ego_agent. So for number_of_agents = 4, there is 1 ego_agent and 3 other_agents
    The position of the obstacle is random.
    '''

    pref_speed = 1.0 #np.random.uniform(1.0, 0.5)
    radius = 0.5 #np.random.uniform(0.4, 0.6)
    agents = []
    positions_list = []

    n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    #Add obstacle in the middle
    obstacle = []

    # Size of square
    size_square = 3 #np.random.uniform(1, 3)
    # Upper x,y value square
    x_v_up = np.random.uniform(-3,3)
    y_v_up = np.random.uniform(-3,3)
    # Lower x,y value of square
    x_v_low = x_v_up - size_square
    y_v_low = y_v_up - size_square
    obstacle_corners = [(x_v_up, y_v_up), (x_v_low, y_v_up), (x_v_low, y_v_low), (x_v_up, y_v_low)]
    obstacle.append(obstacle_corners)

    in_collision = False
    while not in_collision:
        distance = np.random.uniform(2.0, 5.0)
        angle = np.random.uniform(-np.pi, np.pi)
        x0_agent_1 = distance * np.cos(angle)
        y0_agent_1 = distance * np.sin(angle)
        goal_x_1 = -x0_agent_1
        goal_y_1 = -y0_agent_1
        goal = np.array([goal_x_1, goal_y_1])
        initial_pose = np.array([x0_agent_1, y0_agent_1])
        if is_pose_valid_with_obstacles(initial_pose, obstacle) and is_pose_valid_with_obstacles(goal, obstacle):
            in_collision = True
        else:
            in_collision = False
    positions_list.append(np.array([goal_x_1, goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents / 2) - 1):
        in_collision = True
        while in_collision:
            distance = np.random.uniform(2.0, 5.0)
            angle = np.random.uniform(-np.pi, np.pi)
            x0_agent_1 = distance * np.cos(angle)
            y0_agent_1 = distance * np.sin(angle)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal = np.array([goal_x_1, goal_y_1])
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            in_collision = not (
                        is_pose_valid_with_obstacles(initial_pose, obstacle) and is_pose_valid_with_obstacles(goal,
                                                                                                              obstacle) and is_pose_valid(
                    goal, positions_list) and is_pose_valid(initial_pose, positions_list))
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents / 2)):
        cooperation_coef = 0.5
        if ag_id == 0:
            agents.append(Agent(positions_list[0][0], positions_list[0][1],
                                positions_list[1][0], positions_list[1][1], radius, pref_speed,
                                None, ego_agent_policy, ego_agent_dynamics,
                                [OtherAgentsStatesSensor, OccupancyGridSensor], ag_id))
        else:
            agents.append(Agent(positions_list[2 * ag_id][0], positions_list[2 * ag_id][1],
                                positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1], radius, pref_speed,
                                None, other_agents_policy, other_agents_dynamics,
                                [OtherAgentsStatesSensor], 2 * ag_id, cooperation_coef))

        agents.append(
            Agent(positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1],
                  positions_list[2 * ag_id][0], positions_list[2 * ag_id][1], radius, pref_speed, None,
                  other_agents_policy, other_agents_dynamics,
                  [OtherAgentsStatesSensor], 2 * ag_id + 1, cooperation_coef))

    if "Static" in str(agents[0].policy):
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def train_stage_1(number_of_agents=4, ego_agent_policy=MPCPolicy,other_agents_policy=[RVOPolicy], ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    '''
    This is stage 1 of the training scenario.
    Square/wall shaped obstacles: [0,4]
    Random agents: [0-4]
    Goal distance: [8,12]m
    '''
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    pref_speed = 1.0 #np.random.uniform(1.0, 0.5)
    radius = 0.5 #np.random.uniform(0.4, 0.6)
    agents = []
    n_agents = random.randint(2, np.maximum(number_of_agents, 2))  # no. of other agents is randomly chosen

    min_obs = 1
    max_obs = 3
    ready = False

    #Minimum and maximum distance of agent
    min_dist = 2.0
    max_dist = 5.0

    #Minimum and maximum distance of obstacles
    min_range = -4.0
    max_range = 6.0

    while not ready:
        obstacle = compute_random_obstacles(min_obs, max_obs, min_range, max_range)
        positions_list = compute_first_agent(min_obs,max_obs,obstacle, min_dist,max_dist, min_range,max_range)
        positions_list = compute_other_agents(n_agents, obstacle, positions_list, min_dist,max_dist)
        if len(positions_list) == int(n_agents / 2) * 2:
            ready = True

    for ag_id in range(int(n_agents / 2)):
        cooperation_coef = 0.5
        if ag_id == 0:
            agents.append(Agent(positions_list[0][0], positions_list[0][1],
                                positions_list[1][0], positions_list[1][1], radius, pref_speed,
                                None, ego_agent_policy, ego_agent_dynamics,
                                [OtherAgentsStatesSensor, LaserScanSensor], 0))
        else:
            agents.append(Agent(positions_list[2 * ag_id][0], positions_list[2 * ag_id][1],
                                positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1], radius, pref_speed,
                                None, other_agents_policy, other_agents_dynamics,
                                [OtherAgentsStatesSensor], 2 * ag_id, cooperation_coef))

        agents.append(
            Agent(positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1],
                  positions_list[2 * ag_id][0], positions_list[2 * ag_id][1], radius, pref_speed, None,
                  other_agents_policy, other_agents_dynamics,
                  [OtherAgentsStatesSensor], 2 * ag_id + 1, cooperation_coef))

    if "Static" in str(agents[0].policy):
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def compute_random_obstacles(min_n_of_obstacles,max_n_of_obstacles, min_range, max_range):
    ## Define obstacles
    obstacle = []
    # Number of obstacles
    n_obstacles = random.randint(min_n_of_obstacles, max_n_of_obstacles)

    for i in range(n_obstacles):
        shape = np.random.choice(['square', 'rectangle'])
        if shape == 'square':
            overlap = False
            count_s = 0
            while not overlap:
                # Size of square
                if count_s > 5:
                    size_square = 1
                    if count_s > 50:
                        size_square = 0.5
                else:
                    size_square = np.random.uniform(1, 3)
                # Upper x,y value square
                x_v_up = np.random.uniform(min_range, max_range)
                y_v_up = np.random.uniform(min_range, max_range)
                # Lower x,y value of square
                x_v_low = x_v_up - size_square
                y_v_low = y_v_up - size_square
                obstacle_corners = [(x_v_up, y_v_up), (x_v_low, y_v_up), (x_v_low, y_v_low), (x_v_up, y_v_low)]
                overlap = is_shape_valid(obstacle_corners, obstacle)
                count_s += 1
            obstacle.append(obstacle_corners)
        else:
            overlap = False
            count_r = 0
            while not overlap:
                # Rectangle
                if count_r > 5:
                    size_rec_x = 1
                    size_rec_y = 2
                    if count_r > 50:
                        size_rec_x = 0.5
                        size_rec_y = 1
                else:
                    size_rec_x = np.random.uniform(1, 3)
                    if size_rec_x > 2:
                        size_rec_y = np.random.uniform(1, 2)
                    else:
                        size_rec_y = np.random.uniform(3, 4)
                # Upper x,y value square
                x_v_up = np.random.uniform(min_range, max_range)
                y_v_up = np.random.uniform(min_range, max_range)
                # Lower x,y value of square
                x_v_low = x_v_up - size_rec_x
                y_v_low = y_v_up - size_rec_y
                obstacle_corners = [(x_v_up, y_v_up), (x_v_low, y_v_up), (x_v_low, y_v_low), (x_v_up, y_v_low)]
                overlap = is_shape_valid(obstacle_corners, obstacle)
                count_r += 1
            obstacle.append(obstacle_corners)

    return obstacle

def compute_first_agent(min_obs,max_obs,obstacle,min_dist,max_dist,min_range,max_range):
    positions_list = []
    ## Define Agents
    in_collision = True
    count_stuck = 0
    while in_collision:
        if count_stuck > 50:
            obstacle = compute_random_obstacles(min_obs,max_obs, min_range,max_range)
            count_stuck = 0
        distance = np.random.uniform(min_dist, max_dist)
        angle = np.random.uniform(-np.pi, np.pi)
        x0_agent_1 = distance * np.cos(angle)
        y0_agent_1 = distance * np.sin(angle)
        goal_x_1 = -x0_agent_1
        goal_y_1 = -y0_agent_1
        goal = np.array([goal_x_1, goal_y_1])
        initial_pose = np.array([x0_agent_1, y0_agent_1])
        in_collision = not (is_pose_valid_with_obstacles(initial_pose, obstacle) and is_pose_valid_with_obstacles(goal,
                                                                                                               obstacle))
        count_stuck += 1
    positions_list.append(np.array([goal_x_1, goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))
    return positions_list

def compute_other_agents(n_agents,obstacle, positions_list,min_dist, max_dist):
    for ag_id in range(int(n_agents/2)-1):
        in_collision = True
        count_stuck = 0
        while in_collision:
            if count_stuck > 50:
                positions_list = []
                break
            distance = np.random.uniform(2.0, 5.0)
            angle = np.random.uniform(-np.pi, np.pi)
            x0_agent_1 = distance*np.cos(angle)
            y0_agent_1 = distance*np.sin(angle)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = not(is_pose_valid_with_obstacles(initial_pose, obstacle) and is_pose_valid_with_obstacles(goal, obstacle) and is_pose_valid(goal, positions_list) and is_pose_valid(initial_pose, positions_list))
            count_stuck += 1
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))
    return positions_list


def train_stage_2(number_of_agents=10, ego_agent_policy=MPCPolicy,other_agents_policy=[RVOPolicy], ego_agent_dynamics=FirstOrderDynamics, other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    '''
    This is stage 2 of the training scenario.
    Square/wall shaped obstacles: [2,10]
    Random agents: [2-6]
    Goal distance: [16,20]m
    '''

    pref_speed = 1.0 #np.random.uniform(1.0, 0.5)
    radius = 0.5 #np.random.uniform(0.4, 0.6)
    agents = []
    positions_list = []

    if seed:
        random.seed(seed)
        np.random.seed(seed)

    n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    min_obs = 2
    max_obs = 5
    ready = False

    # Minimum and maximum distance of agent
    min_dist = 2.0
    max_dist = 6.0

    # Minimum and maximum distance of obstacles
    min_range = -5.0
    max_range = 6.0

    while not ready:
        obstacle = compute_random_obstacles(min_obs, max_obs, min_range, max_range)
        positions_list = compute_first_agent(min_obs, max_obs, obstacle, min_dist, max_dist, min_range, max_range)
        positions_list = compute_other_agents(n_agents, obstacle, positions_list, min_dist, max_dist)
        if len(positions_list) == int(n_agents / 2) * 2:
            ready = True

    for ag_id in range(int(n_agents / 2)):
        cooperation_coef = 0.5
        if ag_id == 0:
            agents.append(Agent(positions_list[0][0], positions_list[0][1],
                                positions_list[1][0], positions_list[1][1], radius, pref_speed,
                                None, ego_agent_policy, ego_agent_dynamics,
                                [OtherAgentsStatesSensor, OccupancyGridSensor], 0))
        else:
            agents.append(Agent(positions_list[2 * ag_id][0], positions_list[2 * ag_id][1],
                                positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1], radius, pref_speed,
                                None, other_agents_policy, other_agents_dynamics,
                                [OtherAgentsStatesSensor], 2 * ag_id, cooperation_coef))

        agents.append(
            Agent(positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1],
                  positions_list[2 * ag_id][0], positions_list[2 * ag_id][1], radius, pref_speed, None,
                  other_agents_policy, other_agents_dynamics,
                  [OtherAgentsStatesSensor], 2 * ag_id + 1, cooperation_coef))

    if "Static" in str(agents[0].policy):
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def agent_with_door(number_of_agents=4, ego_agent_policy=MPCPolicy, other_agents_policy=[RVOPolicy],ego_agent_dynamics=FirstOrderDynamics, other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    '''
    In this scenario there is a opening in the middle of two obstacles (also in the middle)
    The agents are on the opposite sides of the obstacles and have to pass through the door to get to the other side
    other_agents = [2-number_of_agents]
    obstacles = 2
    '''

    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    #Add door
    obstacle = []
    obstacle_1 = [(-2, 1), (-20, 1), (-20, -1), (-2, -1)]
    obstacle_2 = [(20, 1), (2, 1), (2, -1), (20, -1)]
    obstacle.extend([obstacle_1,obstacle_2])

    positions_list = []

    #Define initial agent
    sign = random.choice((-1,1))
    x0_agent_1 = np.random.uniform(-7.0, 7.0)
    y0_agent_1 = sign*np.random.uniform(3.0, 5.0)
    goal_x_1 = -x0_agent_1
    goal_y_1 = -y0_agent_1

    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2)-1):
        in_collision = False
        while not in_collision:
            sign = random.choice((-1, 1))
            x0_agent_1 = np.random.uniform(-7.0, 7.0)
            y0_agent_1 = sign * np.random.uniform(3.0, 5.0)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(goal, positions_list) and is_pose_valid(initial_pose, positions_list)
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(int(n_agents/2)):
        cooperation_coef = 0.5
        if ag_id == 0:
            agents.append(Agent(positions_list[0][0], positions_list[0][1],
                                positions_list[1][0], positions_list[1][1], radius, pref_speed,
                                None, ego_agent_policy, ego_agent_dynamics,
                                [OtherAgentsStatesSensor, OccupancyGridSensor], 0))
        else:
            agents.append(Agent(positions_list[2 * ag_id][0], positions_list[2 * ag_id][1],
                                positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1], radius, pref_speed,
                                None, other_agents_policy, other_agents_dynamics,
                                [OtherAgentsStatesSensor], 2 * ag_id, cooperation_coef))

        agents.append(
            Agent(positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1],
                  positions_list[2 * ag_id][0], positions_list[2 * ag_id][1], radius, pref_speed, None,
                  other_agents_policy, other_agents_dynamics,
                  [OtherAgentsStatesSensor], 2 * ag_id + 1, cooperation_coef))

    if "Static" in str(agents[0].policy):
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def agent_with_multiple_obstacles(number_of_agents=1, ego_agent_policy=MPCPolicy,other_agents_policy=RVOPolicy, ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    '''
        In this scenario there are multiple obstacles in the environment
        other_agents = [2-number_of_agents]
        obstacles = 8
    '''
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []

    if seed:
        random.seed(seed)
        np.random.seed(seed)

    #Add multiple obstacles
    obstacle = []
    obstacle_1 = [(-6, -4), (-10, -4), (-10, -7), (-6, -7)]     #obstacle 1
    obstacle_2 = [(0, 0), (-3, 0), (-3, -3), (0, -3)]           #obstacle 2
    obstacle_3 = [(10, -1), (8, -1), (8, -5), (10, -5)]         #obstacle 3
    obstacle_4 = [(-7, 4), (-10, 4), (-10, 2), (-7, 2)]         #obstacle 4
    obstacle_5 = [(5, 9), (2, 9), (2, 7), (5, 7)]               #obstacle 5
    obstacle_6 = [(-3, 7), (-7, 7), (-7, 6.5), (-3, 6.5)]       #obstacle 6
    obstacle_7 = [(4, -6), (1, -6), (1, -7), (4, -7)]           #obstacle 7
    obstacle_8 = [(7, 4), (1, 4), (1, 3), (7, 3)]               #obstacle 8
    obstacle.extend([obstacle_1, obstacle_2, obstacle_3, obstacle_4, obstacle_5, obstacle_6, obstacle_7, obstacle_8])

    positions_list = []
    in_collision = False
    while not in_collision:
        distance = np.random.uniform(8.0, 10.0)
        angle = np.random.uniform(-np.pi, np.pi)
        x0_agent_1 = distance * np.cos(angle)
        y0_agent_1 = distance * np.sin(angle)
        goal_x_1 = -x0_agent_1
        goal_y_1 = -y0_agent_1
        goal = np.array([goal_x_1, goal_y_1])
        initial_pose = np.array([x0_agent_1, y0_agent_1])
        if is_pose_valid_with_obstacles(initial_pose, obstacle) and is_pose_valid_with_obstacles(goal, obstacle):
            in_collision = True
        else:
            in_collision = False
    positions_list.append(np.array([goal_x_1,goal_y_1]))
    positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    n_agents = random.randint(0,np.maximum(number_of_agents-1,0))
    if not seed:
        n_agents = number_of_agents - 1

    for ag_id in range(n_agents):
        in_collision = False
        while not in_collision:
            distance = np.random.uniform(8.0, 10.0)
            angle = np.random.uniform(-np.pi, np.pi)
            x0_agent_1 = distance * np.cos(angle)
            y0_agent_1 = distance * np.sin(angle)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            if is_pose_valid_with_obstacles(initial_pose, obstacle) and is_pose_valid_with_obstacles(goal,obstacle) and is_pose_valid(goal, positions_list) and is_pose_valid(initial_pose, positions_list):
                in_collision = True
            else:
                in_collision = False
        positions_list.append(np.array([goal_x_1, goal_y_1]))
        positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents + 1):
        if ag_id == 0:
            agents.append(Agent(positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1],
                                positions_list[2 * ag_id][0], positions_list[2 * ag_id][1], radius, pref_speed,
                                None, ego_agent_policy, ego_agent_dynamics,
                                [OtherAgentsStatesSensor, OccupancyGridSensor], ag_id)) # TODO: this should just be ag_id right?
        else:
            agents.append(Agent(positions_list[2 * ag_id + 1][0], positions_list[2 * ag_id + 1][1],
                                positions_list[2 * ag_id][0], positions_list[2 * ag_id][1], radius, pref_speed,
                                None, other_agents_policy, other_agents_dynamics,
                                [OtherAgentsStatesSensor], ag_id))

    if "MPCStaticObsPolicy" == str(agents[0].policy):
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def only_two_agents(number_of_agents=4, ego_agent_policy=RVOPolicy,other_agents_policy=RVOPolicy, ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Corridor scenario
    obstacle = []
    obstacle_1 = [(20,8), (-20, 8), (-20, 5), (20, 5)]
    obstacle_2 = [(20, -5), (-20, -5), (-20, -8), (20, -8)]
    obstacle.extend([obstacle_1, obstacle_2])

    ini_positions_list = []
    goal_positions_list = []

    sign = random.choice((-1,1))
    x0_agent_1 = sign*np.random.uniform(7.0, 12.0)
    y0_agent_1 = np.random.uniform(-4, 4)
    goal_x_1 = -x0_agent_1
    goal_y_1 = random.choice((-1,1))*y0_agent_1

    ini_positions_list.append(np.array([goal_x_1, goal_y_1]))
    goal_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
    #n_agents = random.randint(0,np.maximum(number_of_agents-1,0))
    #if not seed:
    n_agents = number_of_agents - 2

    for ag_id in range(int(n_agents/2)+1):
        if ag_id == 0:
            agents.append(Agent(ini_positions_list[ag_id][0], ini_positions_list[ag_id][1],
                              goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                              None, ego_agent_policy, ego_agent_dynamics,
                              [OtherAgentsStatesSensor, OccupancyGridSensor], 2*ag_id))
        else:
            agents.append(Agent(ini_positions_list[ag_id][0], ini_positions_list[ag_id][1],
                              goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                              None, other_agents_policy, other_agents_dynamics,
                              [OtherAgentsStatesSensor, OccupancyGridSensor], 2*ag_id))

        policy = random.choice([other_agents_policy, NonCooperativePolicy])
        cooperation_coef_ = np.random.uniform(0.0, 0.5)
        agents.append(Agent(goal_positions_list[ag_id][0], goal_positions_list[ag_id][1],
                            ini_positions_list[ag_id][0], ini_positions_list[ag_id][1], radius, pref_speed,
                            None, policy, other_agents_dynamics,
                            [OtherAgentsStatesSensor, OccupancyGridSensor], 2*ag_id+1,cooperation_coef_))


        agents[ag_id].end_condition = ec._corridor_check_if_at_goal

    agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def only_two_agents_with_obstacle(number_of_agents=4, ego_agent_policy=RVOPolicy,other_agents_policy=RVOPolicy, ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Corridor scenario
    obstacle = []
    obstacle_1 = [(20,8), (-20, 8), (-20, 5), (20, 5)]
    obstacle_2 = [(20, -5), (-20, -5), (-20, -8), (20, -8)]
    obstacle.extend([obstacle_1, obstacle_2])

    # Size of square
    size_square = np.random.uniform(1, 2)
    # Upper x,y value square
    x_v_up = np.random.uniform(-2,2)
    y_v_up = np.random.uniform(-2,2)
    # Lower x,y value of square
    x_v_low = x_v_up - size_square
    y_v_low = y_v_up - size_square
    obstacle_corners = [(x_v_up, y_v_up), (x_v_low, y_v_up), (x_v_low, y_v_low), (x_v_up, y_v_low)]
    obstacle.append(obstacle_corners)

    ini_positions_list = []
    goal_positions_list = []

    sign = random.choice((-1,1))
    x0_agent_1 = sign*np.random.uniform(7.0, 12.0)
    y0_agent_1 = np.random.uniform(-4, 4)
    goal_x_1 = -x0_agent_1
    goal_y_1 = random.choice((-1,1))*y0_agent_1

    ini_positions_list.append(np.array([goal_x_1, goal_y_1]))
    goal_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
    #n_agents = random.randint(0,np.maximum(number_of_agents-1,0))
    #if not seed:
    n_agents = number_of_agents - 2

    for ag_id in range(int(n_agents/2)+1):
        if ag_id == 0:
            agents.append(Agent(ini_positions_list[ag_id][0], ini_positions_list[ag_id][1],
                              goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                              None, ego_agent_policy, ego_agent_dynamics,
                              [OtherAgentsStatesSensor, OccupancyGridSensor], 2*ag_id))
        else:
            agents.append(Agent(ini_positions_list[ag_id][0], ini_positions_list[ag_id][1],
                              goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                              None, other_agents_policy, other_agents_dynamics,
                              [OtherAgentsStatesSensor, OccupancyGridSensor], 2*ag_id))

        policy = random.choice([other_agents_policy, NonCooperativePolicy])
        cooperation_coef_ = np.random.uniform(0.0, 0.5)
        agents.append(Agent(goal_positions_list[ag_id][0], goal_positions_list[ag_id][1],
                            ini_positions_list[ag_id][0], ini_positions_list[ag_id][1], radius, pref_speed,
                            None, other_agents_policy, other_agents_dynamics,
                            [OtherAgentsStatesSensor, OccupancyGridSensor], 2*ag_id+1,cooperation_coef_))


        agents[ag_id].end_condition = ec._corridor_check_if_at_goal

    agents[0].policy.static_obstacles_manager.obstacle = obstacle
    #agents[1].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle


def agent_with_corridor(number_of_agents=4, ego_agent_policy=RVOPolicy,other_agents_policy=RVOPolicy, ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    n_agents = random.randint(2, np.maximum(number_of_agents, 2))

    # Corridor scenario
    obstacle = []
    rotation_angle = np.random.uniform(-np.pi,np.pi)
    obstacle_1 = [(15, 3.5), (-15, 3.5), (-15, 2.3), (15, 2.3)]
    obstacle_2 = [(15, -2.3), (-15, -2.3), (-15, -3.5), (15, -3.5)]
    # obstacle_3 = [(11, 4.5), (10, 4.5), (10, -4.5), (11, -4.5)]
    # obstacle_4 = [(-10, 4.5), (-11, 4.5), (-11, -4.5), (-10, -4.5)]
    # obstacle_1 = [rotate_obs(obstacle_1[i][0], obstacle_1[i][1], rotation_angle) for i in range(len(obstacle_1))]
    # obstacle_2 = [rotate_obs(obstacle_2[i][0], obstacle_2[i][1], rotation_angle) for i in range(len(obstacle_2))]
    obstacle.extend([obstacle_1, obstacle_2])

    ini_positions_list = []
    goal_positions_list = []

    sign = random.choice((-1,1))
    x0_agent_1 = sign*np.random.uniform(4.0, 13.0)
    y0_agent_1 = np.random.uniform(-1.8, 1.8)
    goal_x_1 = -x0_agent_1
    goal_y_1 = y0_agent_1*random.choice((-1,1))

    # x0_agent_1, y0_agent_1 = rotate_obs(x0_agent_1, y0_agent_1, rotation_angle)
    # goal_x_1, goal_y_1 = rotate_obs(goal_x_1, goal_y_1, rotation_angle)

    goal_positions_list.append(np.array([goal_x_1, goal_y_1]))
    ini_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents):
        in_pose_valid_ = False
        while not in_pose_valid_:
            sign = random.choice((-1, 1))
            x0_agent_1 = sign * np.random.uniform(4.0, 13.0)
            y0_agent_1 = np.random.uniform(-1.8, 1.8)
            goal_x_1 = -x0_agent_1
            goal_y_1 = y0_agent_1 * random.choice((-1,1))

            # x0_agent_1, y0_agent_1 = rotate_obs(x0_agent_1, y0_agent_1, rotation_angle)
            # goal_x_1, goal_y_1 = rotate_obs(goal_x_1, goal_y_1, rotation_angle)

            goal = np.array([goal_x_1, goal_y_1])
            initial_pose = np.array([x0_agent_1, y0_agent_1])
            if sign == 1:
                in_pose_valid_ = is_pose_valid(goal, goal_positions_list) and is_pose_valid(initial_pose,
                                                                                            ini_positions_list)
            else:
                in_pose_valid_ = is_pose_valid(goal, ini_positions_list) and is_pose_valid(initial_pose,
                                                                                           goal_positions_list)
        ini_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
        goal_positions_list.append(np.array([goal_x_1, goal_y_1]))


    #random.shuffle(ini_positions_list)
    #random.shuffle(goal_positions_list)

    for ag_id in range(n_agents):
        if np.random.uniform(0,1)>0.8:
           other_agents_policy = NonCooperativePolicy
        else:
           other_agents_policy = RVOPolicy
        if ag_id == 0:
            agents.append(Agent(ini_positions_list[ag_id][0], ini_positions_list[ag_id][1],
                                goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                                None, ego_agent_policy, ego_agent_dynamics,
                                [OtherAgentsStatesSensor, OccupancyGridSensor], ag_id))
        else:
            agents.append(Agent(ini_positions_list[ag_id][0], ini_positions_list[ag_id][1],
                                goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                                None, other_agents_policy, other_agents_dynamics,
                                [OtherAgentsStatesSensor], ag_id))

        # #policy = random.choice([other_agents_policy, NonCooperativePolicy])
        # cooperation_coef_ = np.random.uniform(0.1,1.0)
        # if np.random.uniform(0,1)>0.8:
        #    other_agents_policy = NonCooperativePolicy
        # else:
        #    other_agents_policy = RVOPolicy
        # agents.append(Agent(goal_positions_list[ag_id][0], goal_positions_list[ag_id][1],
        #                     ini_positions_list[ag_id][0], ini_positions_list[ag_id][1], radius, pref_speed,
        #                     None, other_agents_policy, other_agents_dynamics,
        #                     [OtherAgentsStatesSensor], 2 * ag_id + 1, cooperation_coef_))
        #
        # agents[ag_id].end_condition = ec._corridor_check_if_at_goal

    agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def rotate_obs(x, y, theta):
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    return x_new, y_new

def agent_with_corridor_with_obstacle(number_of_agents=5, ego_agent_policy=MPCPolicy, other_agents_policy=RVOPolicy, ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Corridor scenario
    obstacle = []
    obstacle_1 = [(20,8), (-20, 8), (-20, 5), (20, 5)]
    obstacle_2 = [(20, -5), (-20, -5), (-20, -8), (20, -8)]
    obstacle.extend([obstacle_1, obstacle_2])

    # Size of square
    size_square = np.random.uniform(2, 4)
    # Upper x,y value square
    x_v_up = 2#np.random.uniform(-4,4)
    y_v_up = 2#np.random.uniform(-4,4)
    # Lower x,y value of square
    x_v_low = x_v_up - size_square
    y_v_low = y_v_up - size_square
    obstacle_corners = [(x_v_up, y_v_up), (x_v_low, y_v_up), (x_v_low, y_v_low), (x_v_up, y_v_low)]
    obstacle.append(obstacle_corners)

    ini_positions_list = []
    goal_positions_list = []

    sign = random.choice((-1,1))
    x0_agent_1 = sign*np.random.uniform(7.0, 12.0)
    y0_agent_1 = np.random.uniform(-4, 4)
    goal_x_1 = -x0_agent_1
    goal_y_1 = y0_agent_1
    if sign ==1:
        ini_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
        goal_positions_list.append(np.array([goal_x_1, goal_y_1]))
    else:
        ini_positions_list.append(np.array([goal_x_1, goal_y_1]))
        goal_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
    #n_agents = random.randint(0,np.maximum(number_of_agents-1,0))
    #if not seed:
    n_agents = number_of_agents - 2

    for ag_id in range(int(n_agents/2)):
        in_pose_valid_ = False
        while not in_pose_valid_:
            sign = random.choice((-1, 1))
            x0_agent_1 = sign*np.random.uniform(7.0, 12.0)
            y0_agent_1 = np.random.uniform(-4, 4)
            goal_x_1 = -x0_agent_1
            goal_y_1 = y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_pose_valid_ = is_pose_valid(goal, goal_positions_list) and is_pose_valid(initial_pose, ini_positions_list)
        if sign == 1:
            ini_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
            goal_positions_list.append(np.array([goal_x_1, goal_y_1]))
        else:
            ini_positions_list.append(np.array([goal_x_1, goal_y_1]))
            goal_positions_list.append(np.array([x0_agent_1, y0_agent_1]))

    random.shuffle(ini_positions_list)
    random.shuffle(goal_positions_list)

    for ag_id in range(int(n_agents/2)+1):
        if ag_id == 0:
            agents.append(Agent(ini_positions_list[ag_id][0], ini_positions_list[ag_id][1],
                              goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                              None, ego_agent_policy, ego_agent_dynamics,
                              [OtherAgentsStatesSensor, OccupancyGridSensor], 2*ag_id))
        else:
            agents.append(Agent(ini_positions_list[ag_id][0], ini_positions_list[ag_id][1],
                              goal_positions_list[ag_id][0], goal_positions_list[ag_id][1], radius, pref_speed,
                              None, other_agents_policy, other_agents_dynamics,
                              [OtherAgentsStatesSensor, OccupancyGridSensor], 2*ag_id))

        agents.append(Agent(goal_positions_list[ag_id][0], goal_positions_list[ag_id][1],
                            ini_positions_list[ag_id][0], ini_positions_list[ag_id][1], radius, pref_speed,
                            None, other_agents_policy, other_agents_dynamics,
                            [OtherAgentsStatesSensor, OccupancyGridSensor], 2*ag_id+1))


        agents[ag_id].end_condition = ec._corridor_check_if_at_goal

    agents[0].policy.static_obstacles_manager.obstacle = obstacle


    return agents, obstacle

def single_agent_in_a_corridor_with_obstacle(number_of_agents=5, ego_agent_policy=MPCPolicy, other_agents_policy=RVOPolicy, ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Corridor scenario
    obstacle = []
    obstacle_1 = [(20,8), (-20, 8), (-20, 5), (20, 5)]
    obstacle_2 = [(20, -5), (-20, -5), (-20, -8), (20, -8)]
    obstacle.extend([obstacle_1, obstacle_2])

    # Size of square
    size_square = np.random.uniform(2, 4)
    # Upper x,y value square
    x_v_up = 2#np.random.uniform(-4,4)
    y_v_up = 2#np.random.uniform(-4,4)
    # Lower x,y value of square
    x_v_low = x_v_up - size_square
    y_v_low = y_v_up - size_square
    obstacle_corners = [(x_v_up, y_v_up), (x_v_low, y_v_up), (x_v_low, y_v_low), (x_v_up, y_v_low)]
    obstacle.append(obstacle_corners)

    ini_positions_list = []
    goal_positions_list = []

    sign = random.choice((-1,1))
    x0_agent_1 = sign*np.random.uniform(7.0, 12.0)
    y0_agent_1 = np.random.uniform(-4, 4)
    goal_x_1 = -x0_agent_1
    goal_y_1 = y0_agent_1
    ini_positions_list.append(np.array([x0_agent_1, y0_agent_1]))
    goal_positions_list.append(np.array([goal_x_1, goal_y_1]))

    agents.append(Agent(ini_positions_list[0][0], ini_positions_list[0][1],
                       goal_positions_list[0][0], goal_positions_list[0][1], radius, pref_speed,
                       None, ego_agent_policy, ego_agent_dynamics,
                       [LaserScanSensor], 0))



    agents[0].end_condition = ec._corridor_check_if_at_goal

    agents[0].policy.static_obstacles_manager.obstacle = obstacle


    return agents, obstacle


def agent_with_crossing(number_of_agents=1, ego_agent_policy=MPCPolicy, other_agents_policy=RVOPolicy, ego_agent_dynamics=FirstOrderDynamics,other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    pref_speed = 1.0#np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    n_agents = number_of_agents

    # Corridor scenario
    obstacle = []
    obstacle_1 = [(10,10), (2, 10), (2, 2), (10, 2)]
    obstacle_2 = [(-2, 10), (-10, 10), (-10, 2), (-2, 2)]
    obstacle_3 = [(10, -2), (2, -2), (2, -10), (10, -10)]
    obstacle_4 = [(-2, -2), (-10, -2), (-10, -10), (-2, -10)]
    obstacle.extend([obstacle_1, obstacle_2, obstacle_3, obstacle_4])

    positions_list_1 = []
    Long = np.random.uniform(7.0, 10.0)
    Short = np.random.uniform(-1.0, 1.0)
    total1 = [Long, Short]
    total2 = [Short, Long]
    total = random.choice((total1, total2))
    x0_agent_1 = total[0]
    y0_agent_1 = total[1]
    goal_x_1 = -total[0]
    goal_y_1 = -total[1]
    positions_list_1.append(np.array([goal_x_1,goal_y_1]))
    positions_list_1.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents):
        in_collision = False
        while not in_collision:
            Long = np.random.uniform(7.0, 10.0)
            Short = np.random.uniform(-1.0, 1.0)
            total1 = [Long, Short]
            total2 = [Short, Long]
            total = random.choice((total1, total2))
            x0_agent_1 = total[0]
            y0_agent_1 = total[1]
            goal_x_1 = -total[0]
            goal_y_1 = -total[1]
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(goal, positions_list_1) or is_pose_valid(initial_pose, positions_list_1)
        positions_list_1.append(np.array([goal_x_1, goal_y_1]))
        positions_list_1.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents+1):
        if ag_id == 0:
            agents.append(Agent(positions_list_1[2*ag_id+1][0], positions_list_1[2*ag_id+1][1],
                              positions_list_1[2*ag_id][0], positions_list_1[2*ag_id][1], radius, pref_speed,
                              None, ego_agent_policy, ego_agent_dynamics,
                              [OtherAgentsStatesSensor,OccupancyGridSensor], ag_id))
        else:
            agents.append(Agent(positions_list_1[2*ag_id+1][0], positions_list_1[2*ag_id+1][1],
                              positions_list_1[2*ag_id][0], positions_list_1[2*ag_id][1], radius, pref_speed,
                              None, other_agents_policy, other_agents_dynamics,
                              [OtherAgentsStatesSensor], ag_id))

    if "Static" in str(agents[0].policy):
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle


def agent_with_hallway(number_of_agents=6, ego_agent_policy=MPCPolicy, other_agents_policy=RVOPolicy, ego_agent_dynamics=FirstOrderDynamics, other_agents_dynamics=UnicycleDynamics,agents_sensors=[], seed=None, obstacle=None):
    pref_speed = 1.0  # np.random.uniform(1.0, 0.5)
    radius = 0.5# np.random.uniform(0.5, 0.5)
    agents = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Corridor scenario
    obstacle = []
    obstacle_1 = [(10,6), (3, 6), (3, -6), (10, -6)]
    obstacle_2 = [(-3, 6), (-10, 6), (-10, -6), (-3, -6)]
    obstacle_3 = [(-10, 10), (-10.5, 10), (-10.5, -10), (-10, -10)]
    obstacle_4 = [(10, 10), (10.5, 10), (10.5, -10), (10, -10)]
    obstacle.extend([obstacle_1, obstacle_2, obstacle_3, obstacle_4])

    positions_list_1 = []
    sign = np.random.choice([-1,1])
    x0_agent_1 = np.random.uniform(-9,9)
    y0_agent_1 = sign * np.random.uniform(9,10)
    goal_x_1 = -x0_agent_1
    goal_y_1 = -y0_agent_1
    positions_list_1.append(np.array([goal_x_1,goal_y_1]))
    positions_list_1.append(np.array([x0_agent_1, y0_agent_1]))

    n_agents = random.randint(1,np.maximum(number_of_agents-1,1))
    #if not seed:
    #    n_agents = number_of_agents - 1

    for ag_id in range(n_agents):
        in_collision = False
        while not in_collision:
            sign = np.random.choice([-1, 1])
            x0_agent_1 = np.random.uniform(-9, 9)
            y0_agent_1 = sign * np.random.uniform(9, 10)
            goal_x_1 = -x0_agent_1
            goal_y_1 = -y0_agent_1
            goal=np.array([goal_x_1,goal_y_1])
            initial_pose= np.array([x0_agent_1, y0_agent_1])
            in_collision = is_pose_valid(goal, positions_list_1) or is_pose_valid(initial_pose, positions_list_1)
        positions_list_1.append(np.array([goal_x_1, goal_y_1]))
        positions_list_1.append(np.array([x0_agent_1, y0_agent_1]))

    for ag_id in range(n_agents+1):
        if ag_id == 0:
            agents.append(Agent(positions_list_1[2*ag_id+1][0], positions_list_1[2*ag_id+1][1],
                              positions_list_1[2*ag_id][0], positions_list_1[2*ag_id][1], radius, pref_speed,
                              None, ego_agent_policy, ego_agent_dynamics,
                              [OtherAgentsStatesSensor,LaserScanSensor], ag_id))
        else:
            agents.append(Agent(positions_list_1[2*ag_id+1][0], positions_list_1[2*ag_id+1][1],
                              positions_list_1[2*ag_id][0], positions_list_1[2*ag_id][1], radius, pref_speed,
                              None, other_agents_policy, other_agents_dynamics,
                              [OtherAgentsStatesSensor], ag_id))

    if "MPCRLStaticObsPolicy" == str(agents[0].policy):
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def get_testcase_hololens_and_cadrl():
    goal_x = 3
    goal_y = 3
    agents = [Agent(-goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamics, [OccupancyGridSensor, LaserScanSensor], 0),
              Agent(goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, 0.5, CADRLPolicy, UnicycleDynamics, [], 1),
              Agent(-goal_x, goal_y, goal_x, -goal_y, 0.5, 1.0, 0.5, ExternalPolicy, ExternalDynamics, [], 2)]
    return agents

if __name__ == '__main__':
    seed = 1
    carrl = False

    np.random.seed(seed)
    # speed_bnds = [0.5, 1.5]
    speed_bnds = [1.0, 1.0]
    # radius_bnds = [0.2, 0.8]
    radius_bnds = [0.1, 0.1]
    num_agents = 4
    side_length = 4

    ## CARRL
    if carrl:
        num_agents = 2
        side_length = 2
        speed_bnds = [0.5, 1.5]
        radius_bnds = [0.2, 0.8]

    num_test_cases = 500
    test_cases = []

    for i in range(num_test_cases):
        test_case = tc.generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds)
        test_cases.append(test_case)

    if speed_bnds == [1., 1.]:
        pref_speed_string = 'vpref1.0_r{}-{}/'.format(radius_bnds[0], radius_bnds[1])
    else:
        pref_speed_string = ''

    filename = test_case_filename.format(
                num_agents=num_agents, num_test_cases=num_test_cases, pref_speed_string=pref_speed_string,
                dir=os.path.dirname(os.path.realpath(__file__)))
    if carrl:
        filename = filename[:-2] + '_carrl' + filename[-2:]
    filename = filename[:-2] + '_seed' + str(seed).zfill(3) + filename[-2:]

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    pickle.dump(test_cases, open(filename, "wb"))