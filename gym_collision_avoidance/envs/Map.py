from copy import copy
import numpy as np
import imageio
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

class Map():
    def __init__(self, x_width, y_width, grid_cell_size, map_filename=None):
        # Set desired map parameters (regardless of actual image file dims)
        self.x_width = x_width
        self.y_width = y_width
        self.grid_cell_size = grid_cell_size

        # Load the image file corresponding to the static map, and resize according to desired specs
        self.dims = (int(self.x_width/self.grid_cell_size),int(self.y_width/self.grid_cell_size))
        self.origin_coords = np.array([(self.x_width / 2.) / self.grid_cell_size, (self.y_width / 2.) / self.grid_cell_size])
        if map_filename is None:
            self.static_map = np.zeros(self.dims, dtype=bool)
        else:
            '''
            ## This is the version of Michael everett: 
            self.static_map = imageio.imread(map_filename)
            print(self.static_map)
            print(self.static_map.shape)
            if self.static_map.shape != self.dims:
                # print("Resizing map from: {} to {}".format(self.static_map.shape, dims))
                self.static_map = scipy.misc.imresize(self.static_map, self.dims, interp='nearest')
            self.static_map = np.invert(self.static_map).astype(bool)
            '''
            ## This is the version of Sant:
            # map_filename contains obstacles as defined in test_cases
            self.static_map = self.get_occupancy_grid(map_filename)

            # Convert to bool
            self.static_map = self.static_map.astype(bool)

        self.map = copy(self.static_map)

        #self.origin_coords = np.array([(self.x_width/2.)/self.grid_cell_size, (self.y_width/2.)/self.grid_cell_size])

    def world_coordinates_to_map_indices(self, pos):
        # for a single [px, py] -> [gx, gy]
        gx = int(np.floor(self.origin_coords[0]-pos[1]/self.grid_cell_size))
        gy = int(np.floor(self.origin_coords[1]+pos[0]/self.grid_cell_size))
        grid_coords = np.array([gx, gy])
        #in_map = gx >= 0 and gy >= 0 and gx < self.map.shape[0] and gy < self.map.shape[1]
        in_map = gx >= 0 and gy >= 0 and gx < self.dims[0] and gy < self.dims[1] # replaced self.map.shape[0] with self.dims[0]
        return grid_coords, in_map

    def world_coordinates_to_map_indices_vec(self, pos):
        # for a 3d array of [[[px, py]]] -> gx=[...], gy=[...]
        gxs = np.floor(self.origin_coords[0]-pos[:,:,1]/self.grid_cell_size).astype(int)
        gys = np.floor(self.origin_coords[1]+pos[:,:,0]/self.grid_cell_size).astype(int)
        in_map = np.logical_and.reduce((gxs >= 0, gys >= 0, gxs < self.map.shape[0], gys < self.map.shape[1]))
        
        # gxs, gys filled to -1 if outside map to ensure you don't query pts outside map
        not_in_map_inds = np.where(in_map == False)
        gxs[not_in_map_inds] = -1
        gys[not_in_map_inds] = -1
        return gxs, gys, in_map

    def add_agents_to_map(self, agents):
        self.map = self.static_map.copy()
        for agent in agents:
            mask = self.get_agent_mask(agent.pos_global_frame, agent.radius)
            self.map[mask] = 255

    def get_agent_map_indices(self, pos, radius):
        x = np.arange(0, self.map.shape[1])
        y = np.arange(0, self.map.shape[0])
        mask = (x[np.newaxis,:]-pos[1])**2 + (y[:,np.newaxis]-pos[0])**2 < (radius/self.grid_cell_size)**2
        return mask

    def get_agent_mask(self, global_pos, radius):
        [gx, gy], in_map = self.world_coordinates_to_map_indices(global_pos)
        if in_map:
            mask = self.get_agent_map_indices([gx,gy], radius)
            return mask
        else:
            return np.zeros_like(self.map)

    def getSubmapByIndices(self, center_idx_x, center_idx_y, span_x, span_y):
        """
        Extract a submap of span (span_x, span_y) around
        center index (center_idx_x, center_idx_y)
        """

        # Start corner indices of the submap
        start_idx_x = max(0, int(center_idx_x - np.floor(span_x / 2)))
        start_idx_y = max(0, int(center_idx_y - np.floor(span_y / 2)))

        # Compute end indices (assure size of submap is correct, if out pf bounds)
        max_idx_x = self.map.shape[0] - 1
        max_idx_y = self.map.shape[1] - 1

        # End indices of the submap (this corrects for the bounds of the grid
        end_idx_x = start_idx_x + span_x
        if end_idx_x > max_idx_x:
            end_idx_x = max_idx_x
            start_idx_x = end_idx_x - span_x
        end_idx_y = start_idx_y + span_y
        if end_idx_y > max_idx_y:
            end_idx_y = max_idx_y
            start_idx_y = end_idx_y - span_y

        return start_idx_x, start_idx_y, end_idx_x, end_idx_y

    def get_occupancy_grid(self, obstacles):

        # This function puts the obstacles in the right places of the static grid
        # Dimension is 300 by 300 because this is also used in the trained auto-encoder model

        #Initialize variables
        occupancy_grid = np.zeros(shape=self.dims) #dimension (300,300), dtype float64

        # For every obstacle, change grid value to 1
        for obs in obstacles:
            # Initialize variables
            start_idx, _ = self.world_coordinates_to_map_indices(obs[1])
            end_idx, _ = self.world_coordinates_to_map_indices(obs[3])

            for ii in range(start_idx[0],(end_idx[0]+1),1):
                for jj in range(start_idx[1],(end_idx[1]+1),1):
                    occupancy_grid[ii, jj] = 1

            '''
            This can maybe be used if the obstacles are not square/rectangle or if they are crocket. 
            # Initialize variables
            start_idx_x = np.inf
            start_idx_y = np.inf
            end_idx_x = 0
            end_idx_y = 0

            for j, k in obstacles[i]:
                grid_coords, _ = self.world_coordinates_to_map_indices([j,k])
                if grid_coords[0] < start_idx_x:
                    start_idx_x = grid_coords[0]
                if grid_coords[1] < start_idx_y:
                    start_idx_y = grid_coords[1]
                if grid_coords[0] > end_idx_x:
                    end_idx_x = grid_coords[0]
                if grid_coords[1] > end_idx_y:
                    end_idx_y = grid_coords[1]
            x = list(range(start_idx_x, end_idx_x+1))
            y = list(range(start_idx_y, end_idx_y+1))
            for ii in x:
                for jj in y:
                    occupancy_grid[ii, jj] = 1'''

        return occupancy_grid

    def get_occupancy_grid2(self, obstacles):

        # fig = plt.figure()
        # plt.clf()
        # plt.xlim([-25, 25])
        # plt.ylim([-25, 25])
        # ax = fig.add_subplot(1, 1, 1)
        # for obs in obstacles:
        #     ax.add_patch(plt.Polygon(obs, fill=True))
        # plt.axis('off')
        # check = fig.savefig('../gym-collision-avoidance/gym_collision_avoidance/envs/world_maps/WORLDMAP.png', bbox_inches='tight')
        # plt.close()
        occupancy_grid = Image.open(
            '../gym-collision-avoidance/gym_collision_avoidance/envs/world_maps/001.png').convert('L')
        occupancy_grid = occupancy_grid.resize(self.dims)
        occupancy_grid.save('../gym-collision-avoidance/gym_collision_avoidance/envs/world_maps/resizedWORLDMAP.png')
        MAP = np.array(occupancy_grid.getdata()).reshape((occupancy_grid.size[1], occupancy_grid.size[0]))
        occupancy_grid = np.where(MAP < 255, True, False)


        # # Dit werkt nog niet echt goed
        # occupancy_grid = np.where(MAP < 255, 0, 255)
        # occupancy_grid = occupancy_grid.astype('uint8')
        # # Detect polygons
        # _, threshold = cv2.threshold(occupancy_grid, 240, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # font = cv2.FONT_HERSHEY_COMPLEX
        #
        # # Make list of polygons with its corners
        # corners_polygons = []
        # for cnt in contours:
        #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        #
        #     # Convert corners to world coordinates
        #     approx_list = approx.tolist()  # Set correct format
        #     corners = []
        #
        #     for pos in approx_list:
        #         world_pos = pos #map_to_world(pos, resolution, map_origin)  # Convert to world coordinates
        #         corners.append(world_pos)
        #
        #     corners_polygons.append(corners)
        #
        #     cv2.drawContours(occupancy_grid, [approx], 0, (0), 5)
        #     x = approx.ravel()[0]
        #     y = approx.ravel()[1]
        #
        #     if len(approx) == 3:
        #         cv2.putText(occupancy_grid, "Triangle", (x, y), font, 1, (0))
        #     elif len(approx) == 4:
        #         cv2.putText(occupancy_grid, "Rectangle", (x, y), font, 1, (0))
        #     elif len(approx) == 5:
        #         cv2.putText(occupancy_grid, "Pentagon", (x, y), font, 1, (0))
        #     elif 6 < len(approx) < 15:
        #         cv2.putText(occupancy_grid, "Ellipse", (x, y), font, 1, (0))
        #     else:
        #         cv2.putText(occupancy_grid, "Circle", (x, y), font, 1, (0))
        #
        # fig = plt.figure()
        # ax = fig.subplots(1)
        # ax.imshow(occupancy_grid)
        # plt.show()
        return occupancy_grid









