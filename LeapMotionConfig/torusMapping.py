import numpy as np
import math

#### REFERENCE MATRIX: 
#                [METACARPAL] [PROX PHALANX] [M. PHALANX] [D. PHALANX]
#               [[  0.          33.65723038  36.76741409  88.29069519] [THUMB]
#               [ 13.11555862  69.48578644 161.43367004 163.93757629]  [INDEX]
#               [  8.68810368  42.49485397 138.06262207 161.1791687 ]  [MIDDLE]
#               [ 10.69950581  30.49986267 118.01464844 158.56045532]  [RING]
#               [ 15.83293056  33.06761169  82.57316589 134.74368286]] [PINKY]
#
# row corresponds to axis 0, column to axis 1. so motion_data[2, :] means all columns from row 2.

#pretty much just angle to x/y coordinates, but this function is primarily to reduce complexity of the dataset for the neural network.
#then the computation of x/y coordinate to multijoint movement is done via robotic hand programming to find the movement required to 
#get to a new x/y coordinate.
def convert_multijoint_to_torus_space(single_motion_data):
    conversion_dict = { }

    prox_bone_length = 0.5
    mid_bone_length = 0.5

    finger_dict_names = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']
    row_shape = single_motion_data.shape[0]
    for i in range(row_shape):
        curr_finger = single_motion_data[i, :]
        prox_phalanx_angle = -curr_finger[1]
        mid_phalanx_angle = -curr_finger[2] #all angles with respect to the palm.

        # proximal bone x/y coordinate
        prox_x = prox_bone_length*math.cos(prox_phalanx_angle*(math.pi/180.0)) 
        prox_y = prox_bone_length*math.sin(prox_phalanx_angle*(math.pi/180.0))

        #the base x/y coordinates of the middle phalanx (e.g. what they would be if starting from the origin (the palm))
        mid_base_x = mid_bone_length*math.cos(mid_phalanx_angle*(math.pi/180.0)) 
        mid_base_y = mid_bone_length*math.sin(mid_phalanx_angle*(math.pi/180.0)) 

        # x/y coordinates of the end of the middle phalanx
        mid_x = prox_x + mid_base_x
        mid_y = prox_x + mid_base_y

        conversion_dict[finger_dict_names[i]] = [ mid_x , mid_y ]

    return conversion_dict
    
def batch_multijoint_to_torus_space(motion_data_list):
    final_list = [ ]
    for i in motion_data_list:
        if type(i) == list: #if no downsampling was done.
            temp_list = [ ]
            for j in i:
                temp_list.append(convert_multijoint_to_torus_space(j))
            final_list.append(temp_list)
        else:
            final_list.append( convert_multijoint_to_torus_space(i) )
    return final_list