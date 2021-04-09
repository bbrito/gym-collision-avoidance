import numpy as np
import math
import pylab as pl
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from mpc_rl_collision_avoidance.policies.StaticObstacleManager import StaticObstacleManager

class AngularMapSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.no_of_slices = Config.NUM_OF_SLICES
        self.max_range = Config.MAX_RANGE
        self.x_width = 20 * self.max_range
        self.y_width = 20 * self.max_range
        # For full FOV:
        self.angle_max = np.pi
        self.angle_min = -np.pi
        self.name = 'angular_map'
        self.plot = True
        self.static_obstacles_manager = StaticObstacleManager()

        # Either calculation from laserscan data or occupancy grid data can be chosen. Both give the same results
        self.Laserscan = True
        self.Occupancygrid = False

        # For laserscan
        if self.Laserscan:
            self.num_beams = Config.LASERSCAN_LENGTH
            self.range_resolution = 0.1
            self.min_range = 0  # meters
            self.min_angle = -np.pi
            self.max_angle = np.pi

            self.angles = np.linspace(self.min_angle, self.max_angle, self.num_beams)
            self.ranges = np.arange(self.min_range, self.max_range, self.range_resolution)
            self.debug = False

    def sense(self, agents, agent_index, top_down_map):
        '''
        The angular map encodes the distance of the next obstacle within angular ranges.
        The output of the map is a one dimensional vector 'l'.

        This function looks at every obstacle inside a grid around the ego agent. For every point in the obstacle it
        calculates the distance to the the ego_agent and in which slice it is using polar coordinates.
        Then it saves the point of the obstacle that is closest to the agent inside that slice.
        '''

        # Get position of ego agent
        self.ego_agent = agents[agent_index]

        # Take the heading of the agent into account
        self.heading = self.ego_agent.heading_global_frame

        # Initialize angular map
        Angular_Map = self.max_range * np.ones([self.no_of_slices])  # vector of 72

        # Angles
        self.radial_resolution = (self.angle_max - self.angle_min) / self.no_of_slices

        ## Occupancy grid
        if self.Occupancygrid:
            angular_map = self.angular_map_from_batch_grid(Angular_Map, top_down_map)
            self.static_obstacles_manager.angular_map = angular_map

        ## Laserscan
        if self.Laserscan:
            sensor = LaserScanSensor
            ranges = sensor.sense(self, agents, agent_index, top_down_map)
            angular_map = self.angular_map_from_laser_scan(Angular_Map, ranges)
            self.static_obstacles_manager.get_list_of_nearest_obstacles(self.ego_agent, (self.max_range + 1))
            self.static_obstacles_manager.angular_map = angular_map

        if self.plot:
            self.plot_angular_grid(((1 - ranges) * Config.MAX_RANGE), agents[0])
            if self.Laserscan:
                ego_agent_pos = self.ego_agent.pos_global_frame
                # Get map indices of ego agent
                ego_agent_pos_idx, _ = top_down_map.world_coordinates_to_map_indices(ego_agent_pos)
                self.plot_top_down_map(top_down_map, ego_agent_pos_idx, agents[0])



        return 1-angular_map/Config.MAX_RANGE

    def angular_map_from_batch_grid(self, Angular_Map, top_down_map):
        '''
        This function creates an angular map from the cornerpoints of the obstacles.
        1) It first finds the nearest obstacles
        2) It then only looks at the obstacles that are closer than a certain threshold
        3) Then it computes the three closest corner points
        4) Then it computes 2 lines from those closest corner points
        5) Then it iterates along these lines to obtain the angular map
        '''
        ego_agent_pos = self.ego_agent.pos_global_frame

        # Orientation
        if self.heading >= 0:
            self.orientation = self.heading
        else:
            self.orientation = (2*np.pi)+self.heading

        # Obstacles
        self.static_obstacles_manager.get_list_of_nearest_obstacles(self.ego_agent, (self.max_range + 1))
        obs_in_range = self.static_obstacles_manager.obstacles_in_range

        if len(obs_in_range) == 0:
            # If there is no obstacle in the sensor range, immediately return the angular map
            return Angular_Map
        else:
            for obst_coor in obs_in_range:
                lines = []
                # Get cornerpoints of obstacles that are close to the agent
                corners_imp = self.static_obstacles_manager.get_important_corners(self.ego_agent, obst_coor)
                corner1 = corners_imp[0]
                corner2 = corners_imp[1]
                corner3 = corners_imp[2]
                if corner1[0] == corner2[0]:
                    # Line is vertical
                    start = corner1[1]
                    end = corner2[1]
                    constant_vert = corner1[0]
                elif corner1[0] == corner3[0]:
                    # Line is vertical
                    start = corner1[1]
                    end = corner3[1]
                    constant_vert = corner1[0]
                elif corner2[0] == corner3[0]:
                    # Line is vertical
                    start = corner2[1]
                    end = corner3[1]
                    constant_vert = corner2[0]
                else:
                    print('ERROR: The obstacles are not defined correctly')
                    return
                line_vert = np.linspace(start, end, (int(abs(start - end)) * 8) + 1)
                lines.append(line_vert)
                if corner1[1] == corner2[1]:
                    # Line is horizontal
                    start = corner1[0]
                    end = corner2[0]
                    constant_hor = corner1[1]
                elif corner1[1] == corner3[1]:
                    # Line is horizontal
                    start = corner1[0]
                    end = corner3[0]
                    constant_hor = corner1[1]
                elif corner2[1] == corner3[1]:
                    # Line is horizontal
                    start = corner2[0]
                    end = corner3[0]
                    constant_hor = corner2[1]
                else:
                    print('ERROR: The obstacles are not defined correctly')
                    return
                line_hor = np.linspace(start, end, (int(abs(start - end)) * 8) + 1)
                lines.append(line_hor)
                # Create two lines from the 3 most important coordinates
                for i in range(len(lines)):
                    # Iterate over the line
                    for idx in lines[i]:
                        if i == 0:
                            rel_coords = np.array([constant_vert, idx]) - ego_agent_pos
                        else:
                            rel_coords = np.array([idx, constant_hor]) - ego_agent_pos
                        l2norm = np.linalg.norm(rel_coords)  # Distance between ego agent and obstacle point

                        # We start counting from positive x-axis (+self.orientation)
                        phi = math.atan2(rel_coords[1], rel_coords[0]) - self.orientation
                        if phi < 0:
                            phi = (2*np.pi) + phi
                            if phi < 0:
                                phi = (2*np.pi) + phi
                        # Compute the index of the vector in the angular map aligned with orientation of ego_agent
                        rad_idx = int(phi / self.radial_resolution)

                        Angular_Map[rad_idx] = min(Angular_Map[rad_idx], l2norm)

        # Plot for debugging
        if self.plot is True:
            ego_agent_pos_idx, _ = top_down_map.world_coordinates_to_map_indices(ego_agent_pos)
            self.plot_top_down_map(top_down_map, ego_agent_pos_idx)

        return Angular_Map

    def angular_map_from_laser_scan(self, Angular_Map, ranges):
        # Convert slice angles to start angle of the laserscan
        angles = self.angles + self.heading
        self.phi = angles

        # Get correct slice index for angles
        rad_idx = self.phi / self.radial_resolution
        rad_idx = rad_idx.astype(int)
        rad_idx = rad_idx - rad_idx[0]

        # range value = distance value
        j = 0
        for i in rad_idx:
            if i >= 0 and i < self.no_of_slices:
                Angular_Map[i] = min(Angular_Map[i], ranges[j])
                j += 1

        return Angular_Map

    # Plot
    def plot_angular_grid(self, Angular_Map, ag):
        fig = pl.figure("Angular grid")
        ax_ped_grid = pl.subplot()
        ax_ped_grid.clear()
        self.plot_Angular_map_vector(ax_ped_grid, Angular_Map,ag, max_range=6.0)
        ax_ped_grid.plot(30, 30, color='r', marker='o', markersize=4)
        ax_ped_grid.scatter(0, 0, s=100, c='red', marker='o')
        aanliggend = 1 * math.cos(ag.heading_global_frame)
        overstaand = 1 * math.sin(ag.heading_global_frame)
        ax_ped_grid.arrow(0, 0, aanliggend, overstaand, head_width=0.5, head_length=0.5)  # agent poiting direction
        ax_ped_grid.set_xlim([-self.max_range - 1, self.max_range + 1])
        ax_ped_grid.set_ylim([-self.max_range - 1, self.max_range + 1])
        ax_ped_grid.set_axis_off()
        fig.canvas.draw()

        # sleep(0.5)  # Time in seconds.
        #pl.show(block=False)
        #sleep(0.5)  # Time in seconds.


    def plot_Angular_map_vector(self, ax, Angular_Map,ag, max_range=6):
        number_elements = Angular_Map.shape[0]
        if self.Occupancygrid:
            #Angular_Map = Angular_Map[::-1]
            min_angle = self.orientation # reverse the entire array
        else:
            min_angle = ag.heading_global_frame - np.pi
        cmap = pl.get_cmap('gnuplot')


        for ii in range(number_elements):
            if self.Laserscan:
                angle_start = (min_angle + ii * (2 * np.pi / Config.NUM_OF_SLICES)) * 180 / np.pi
                angle_end = (min_angle + (ii + 1) * (2 * np.pi / Config.NUM_OF_SLICES)) * 180 / np.pi
            if self.Occupancygrid:
                angle_start = (min_angle + ii * self.radial_resolution) * 180 / np.pi
                angle_end = (min_angle + (ii + 1) * self.radial_resolution) * 180 / np.pi


            distance_cone = pl.matplotlib.patches.Wedge((0.0, 0.0),
                                                        Angular_Map[ii],
                                                        angle_start, angle_end,
                                                        facecolor=cmap(Angular_Map[ii] / max_range),
                                                        alpha=0.5)

            ax.add_artist(distance_cone)

    # Plot
    def plot_top_down_map(self, top_down_map, ego_agent_idx, ag):
        fig = plt.figure("Top down map")
        draw_circle = plt.Circle((ego_agent_idx[1], ego_agent_idx[0]), 60, fill=False, edgecolor='red')
        ax = fig.subplots(1)
        ax.imshow(top_down_map.map, aspect='equal', cmap = 'Blues')
        ax.scatter(ego_agent_idx[1], ego_agent_idx[0], s=100, c='red', marker='o')
        aanliggend = 20 * math.cos(ag.heading_global_frame)
        overstaand = 20 * math.sin(ag.heading_global_frame)
        ax.arrow(ego_agent_idx[1], ego_agent_idx[0], aanliggend, overstaand, head_width=5, head_length=5)  # agent poiting direction
        ax.add_artist(draw_circle)
        ax.set_axis_off()
        plt.show()


