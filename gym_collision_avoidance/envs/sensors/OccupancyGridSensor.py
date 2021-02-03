import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.config import Config
import math
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

class OccupancyGridSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.x_width = Config.SUBMAP_WIDTH
        self.y_width = Config.SUBMAP_HEIGHT
        self.grid_cell_size = Config.SUBMAP_RESOLUTION
        self.plot = False
        self.name = 'local_grid'

    def sense(self, agents, agent_index, top_down_map):

        """
        # Grab (i,j) coordinates of the upper right and lower left corner of the desired OG map, within the entire map
        host_agent = agents[agent_index]
        [map_i_high, map_j_low], _ = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame-np.array([3., 3]))
        [map_i_low, map_j_high], _ = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame+np.array([3., 3.]))

        # Assume areas outside static_map should be filled with zeros
        og_map = np.zeros((int(6/top_down_map.grid_cell_size), int(6/top_down_map.grid_cell_size)), dtype=bool)

        if map_i_low >= top_down_map.map.shape[0] or map_i_high < 0 or map_j_low >= top_down_map.map.shape[1] or map_j_high < 0:
            # skip rest ==> og_map and top_down_map have no overlap
            print("*** no overlap btwn og_map and top_down_map ***")
            print("*** map dims:", map_i_low, map_i_high, map_j_low, map_j_high)
            return og_map

        # super crappy logic to handle when the OG map partially overlaps with the map 
        if map_i_low < 0:
            og_i_low = -map_i_low
            og_i_high = og_map.shape[0]
        elif map_i_high >= top_down_map.map.shape[0]:
            og_i_low = 0
            og_i_high = og_map.shape[0] - (map_i_high - top_down_map.map.shape[0])
        else:
            og_i_low = 0
            og_i_high = og_map.shape[0]

        if map_j_low < 0:
            og_j_low = -map_j_low
            og_j_high = og_map.shape[1]
        elif map_j_high >= top_down_map.map.shape[1]:
            og_j_low = 0
            og_j_high = og_map.shape[1] - (map_j_high - top_down_map.map.shape[1])
        else:
            og_j_low = 0
            og_j_high = og_map.shape[1]

        # Don't grab a map index outside the map's boundaries
        map_i_low = np.clip(map_i_low, 0, top_down_map.map.shape[0])
        map_i_high = np.clip(map_i_high, 0, top_down_map.map.shape[0])
        map_j_low = np.clip(map_j_low, 0, top_down_map.map.shape[1])
        map_j_high = np.clip(map_j_high, 0, top_down_map.map.shape[1])

        # Fill the correct OG map indices with the section of the map that has been selected
        og_map[og_i_low:og_i_high, og_j_low:og_j_high] = top_down_map.map[map_i_low:map_i_high, map_j_low:map_j_high]
        resized_og_map = self.resize(og_map)
        return resized_og_map

        """
        # Get position of ego agent
        ego_agent = agents[agent_index]
        ego_agent_pos = ego_agent.pos_global_frame
        ego_agent_heading = ego_agent.heading_global_frame

        # Get map indices of ego agent
        ego_agent_pos_idx, _ = top_down_map.world_coordinates_to_map_indices(ego_agent_pos)

        span_x = int(np.ceil(self.x_width))  # 60
        span_y = int(np.ceil(self.y_width))  # 60

        # Get submap indices around ego agent
        start_idx_x, start_idx_y, end_idx_x, end_idx_y = top_down_map.getSubmapByIndices(ego_agent_pos_idx[0],
                                                                                     ego_agent_pos_idx[1], span_x, span_y)

        # Get the batch_grid with filled in values
        float_map = top_down_map.map.astype(float)
        float_map = self.rotate_grid_around_center(float_map, ego_agent_pos_idx, angle=-ego_agent_heading*180/np.pi)
        batch_grid = float_map[start_idx_x:end_idx_x, start_idx_y:end_idx_y]

        # Rotate grid such that it is aligned with the heading
        batch_grid = batch_grid.astype(bool)

        if self.plot:
            self.plot_top_down_map(top_down_map.map,ego_agent_pos_idx, start_idx_x, start_idx_y, ego_agent_heading, title='Original')
            self.plot_top_down_map(float_map, ego_agent_pos_idx, start_idx_x, start_idx_y, ego_agent_heading, title='Rotated')
            self.plot_batch_grid(batch_grid)

        return batch_grid

    # Plot
    def plot_top_down_map(self, top_down_map, ego_agent_idx, start_idx_x, start_idx_y, heading, title):
        fig = plt.figure(title)
        ax = fig.subplots(1)
        ax.imshow(top_down_map, aspect='equal')
        ax.scatter(ego_agent_idx[1], ego_agent_idx[0], s=100, c='red', marker='o')
        rect = patches.Rectangle((start_idx_y, start_idx_x), self.x_width, self.y_width, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        aanliggend = 20 * math.cos(heading)
        overstaand = -20 * math.sin(heading)
        if 'Rotated' == str(title):
            ax.arrow(ego_agent_idx[1], ego_agent_idx[0], 20,0, width=3, head_width=10,
                     head_length=10, fc='yellow')  # agent poiting direction
        else:
            ax.arrow(ego_agent_idx[1], ego_agent_idx[0], aanliggend, overstaand,width=3, fc='yellow', head_width=10, head_length=10)  # agent poiting direction

        plt.show()

    def plot_batch_grid(self, batch_grid):
        fig = plt.figure("batch_grid")
        ax = fig.subplots(1)
        ax.imshow(batch_grid, aspect='equal')
        ax.scatter(self.x_width/2, self.y_width/2, s=100, c='red', marker='o')
        ax.arrow(self.x_width/2, self.y_width/2, 10, 0,width=1, head_width=3,head_length=3, fc='yellow')  # agent poiting direction

        plt.show()

    def resize(self, og_map):
        resized_og_map = og_map.copy()
        return resized_og_map

    def rotate_grid_around_center(self, grid, agent_pos, angle):
        """
        inputs:
          grid: numpy array (gridmap) that needs to be rotated
          angle: rotation angle in degrees
        """
        # Rotate grid into direction of initial heading
        grid = grid.copy()
        rows, cols = grid.shape
        M = cv2.getRotationMatrix2D(center=(agent_pos[1], agent_pos[0]), angle=angle, scale=1)
        grid = cv2.warpAffine(grid, M, (rows, cols))

        return grid

if __name__ == '__main__':
    from gym_collision_avoidance.envs.Map import Map
    from gym_collision_avoidance.envs.agent import Agent
    from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
    from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics

    top_down_map = Map(x_width=10, y_width=10, grid_cell_size=0.1)
    agents = [Agent(0, 3.05, 10, 10, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamics, [], 0)]
    top_down_map.add_agents_to_map(agents)
    og = OccupancyGridSensor()
    og_map = og.sense(agents, 0, top_down_map)

    print(og_map)