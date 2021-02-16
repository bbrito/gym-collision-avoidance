import numpy as np

def _check_if_at_goal(agent):
    is_near_goal = (agent.pos_global_frame[0] - agent.goal_global_frame[0]) ** 2 + (
                agent.pos_global_frame[1] - agent.goal_global_frame[1]) ** 2 <= agent.near_goal_threshold ** 2
    agent.is_at_goal = is_near_goal

def _corridor_check_if_at_goal(agent):
    direction = np.sign(agent.global_state_history[0,1])
    agent.is_at_goal = -1*direction*agent.pos_global_frame[0]>5