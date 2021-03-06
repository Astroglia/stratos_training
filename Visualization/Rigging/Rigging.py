

#class for simulating movement of a robotic finger as a two-joint vector.

import numpy as np
import math

class TwoJointModel:
    def __init__(self, initial_v1_x=1.0, initial_v1_y=0.0, initial_v2_x=1.5, initial_v2_y=0.0):
        self.v1_x, self.v1_y = [ initial_v1_x, initial_v1_y ]
        self.v2_x, self.v2_y = [ initial_v2_x, initial_v2_y ]

    @staticmethod
    def rotate_vector(x, y, rotation):
        new_x = x*math.cos(rotation) - y*math.sin(rotation)
        new_y = x*math.sin(rotation) + y*math.cos(rotation)
        return [ new_x, new_y ]

    #rotate v2 around v1.
    def rotate_around_v1(self, sim_x, sim_y, rotation):
        t_x = math.cos(rotation)*(sim_x - self.v1_x) - math.sin(rotation)*(sim_y - self.v1_y) + self.v1_x
        t_y = math.sin(rotation)*(sim_x - self.v1_x) + math.cos(rotation)*(sim_y - self.v1_y) + self.v1_y
        return t_x, t_y

    #rotates v1 and v2 by v1_rotation, then rotates v2 by v2_rotation. 
    def rigid_rotation(self, v1_rotation, v2_rotation):
        self.v1_x, self.v1_y = TwoJointModel.rotate_vector(self.v1_x, self.v1_y, v1_rotation)

        temp_v2_x, temp_v2_y = TwoJointModel.rotate_vector(self.v2_x, self.v2_y, v1_rotation)
        self.v2_x, self.v2_y = self.rotate_around_v1(temp_v2_x, temp_v2_y, v2_rotation)

    def get_v1_coords(self):
        return [ self.v1_x, self.v1_y ]
    def get_v2_coords(self):
        return [ self.v2_x, self.v2_y ]
    def get_all_coords(self):
        return {'v1_x': self.v1_x, 'v1_y': self.v1_y, 'v2_x': self.v2_x, 'v2_y': self.v2_y }
    def get_all_coords_list_form(self):
        return [ {'x': self.v1_x, 'y': self.v1_y}, {'x': self.v2_x, 'y': self.v2_y} ]