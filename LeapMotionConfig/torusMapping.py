import numpy as np
import math
from sympy import symbols, Eq, solve, sqrt, cos, sin

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
def convert_multijoint_to_torus_space(single_motion_data, return_type='RESULTANT'):
    conversion_dict = { }

    prox_bone_length = 1.0
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

        #the mid x/y coordinates 
        mid_x = (prox_bone_length + mid_bone_length)*math.cos(mid_phalanx_angle*(math.pi/180.0))
        mid_y = (prox_bone_length + mid_bone_length)*math.sin(mid_phalanx_angle*(math.pi/180.0)) 

        #mid_x = prox_x + mid_base_x
        #mid_y = prox_x + mid_base_y
        if (-0.5 < mid_x < 0.5) and (-0.5 < mid_y < 0.5):
            print( mid_x, " || ",mid_y)

        #print(" --- ")
        #print( [ prox_x, prox_y])
        #print( [ mid_x, mid_y])
        if(return_type == 'RESULTANT'):
            conversion_dict[finger_dict_names[i]] = [ mid_x , mid_y ]
        elif(return_type == 'PROXIMAL'):
            conversion_dict[finger_dict_names[i]] = [ prox_x , prox_y ]
        elif(return_type == 'MIDDLE'):
            conversion_dict[finger_dict_names[i]] = [ mid_base_x , mid_base_y ]

    return conversion_dict
    
def batch_multijoint_to_torus_space(motion_data_list, return_type='RESULTANT'):
    final_list = [ ]
    for i in motion_data_list:
        if type(i) == list: #if no downsampling was done.
            temp_list = [ ]
            for j in i:
                temp_list.append(convert_multijoint_to_torus_space(j, return_type))
            final_list.append(temp_list)
        else:
            final_list.append( convert_multijoint_to_torus_space(i, return_type) )
    return final_list

def get_quadrant(x, y):
    if (x>=0) and (y>=0):       return 1
    elif (x>=0) and (y<=0):     return 2
    elif (x<=0) and (y<=0):     return 3
    elif (x<=0) and (y>=0):     return 4

def determine_rotation_direction(current_x, current_y, destination_x, destination_y, calculated_angle): #good god this looks awful.
    c_quad = get_quadrant(current_x, current_y)
    d_quad = get_quadrant(destination_x, destination_y)
    if( ( (c_quad == 3) and (d_quad == 4) ) or ( (c_quad == 1) and (d_quad == 2) ) ) : 
        return calculated_angle
    elif ( ( (c_quad == 4) and (d_quad == 3) ) or ( (c_quad == 2) and (d_quad == 1) ) ): 
        return -calculated_angle
    ### SAME QUADRANTS    
    elif( ( c_quad == 1) and (d_quad == 1) ) or ( ( c_quad == 2) and (d_quad == 2) ):
        if current_y > destination_y: return -calculated_angle
        else: return calculated_angle              
    elif( ( c_quad == 3) and (d_quad == 3) ) or ( ( c_quad == 4) and (d_quad == 4) ): 
        if current_y > destination_y: return calculated_angle
        else: return -calculated_angle    
    ### ADJACENT QUADRANTS
    elif( (c_quad == 2) and (d_quad == 3) ) or ((c_quad == 4) and (d_quad == 1)): 
        if current_y > destination_y: return calculated_angle
        else: return -calculated_angle
    elif( (c_quad == 3) and (d_quad == 2) ) or ((c_quad == 1) and (d_quad == 4)):
        if current_y > destination_y: return -calculated_angle
        else: return calculated_angle
    ### OPPOSITE QUADRANTS
    elif( ( (c_quad == 3) and (d_quad == 1) ) or ( (c_quad == 4) and (d_quad == 2))   ):
        return -calculated_angle
    elif( ( (c_quad == 1) and (d_quad == 3) ) or ( (c_quad == 2) and (d_quad == 4))):
        return calculated_angle

#currently assumes 2 joints.
#current_coordinates --> the x/y coordinate of the torus map. (the output from convert_multijoint_to_torus_space)
#destination_coordinates --> the requested coordinates to move the two-joint model to
#bone_length --> list of bone lengths.
def two_joint_decode_torus_space_mapping(current_coordinates, destination_coordinates, bone_lengths=None):
    if bone_lengths == None:
        bone_lengths = { 'PROX': 1.0, 'MID': 0.5 }
    #1. , find the angle on the inner circle that results in a distance of bone_length away from destination_coordinates.
    def distance_calculation(dest_x, dest_y, curr_x, curr_y, origin_x, origin_y, radius):
        t = symbols('t') #theta
        #distance calculation between points, but since we have a circle we use radius*trig_id(theta)
        #this results in two possible locations that is a distance of --radius-- away from {dest_x, dest_y}
        dist = (dest_x - (origin_x + radius*cos(t)))**2 + (dest_y - (origin_y + radius*sin(t)))**2 - radius**2
        sympy_eq = Eq(dist)
        solution = solve(sympy_eq, t, simplify=False) #should be two pairs of solutions.

        solutions = [ ]
        for i in solution:
            x = radius*math.cos(i)
            y = radius*math.sin(i)
            distance = math.sqrt( (dest_x - x)**2 + (dest_y - y)**2  )
            if (radius - 0.05) < distance < (radius + 0.05):
                solutions.append( [x, y] )
        return solutions
    c_x_v1, c_y_v1 = [ current_coordinates[0][0], current_coordinates[0][1] ] #proximal vector 1 coordinates
    c_x_v2, c_y_v2 = [ current_coordinates[1][0], current_coordinates[1][1] ] #middle   vector 2 coordinates
    d_x, d_y = [ destination_coordinates[0], destination_coordinates[1] ]

    prox_coord_solutions = distance_calculation(d_x, d_y, c_x_v1, c_y_v1, 0, 0, bone_lengths['PROX'])
    #1.a since distance_calculation can give 2 correct results (for a given point, there's always two points on a circle
    # equidistance away from it), discard the result that would form an "illegal" robotic finger mapping.
    def discard_illegal_coordinate_solutions( solution_1, solution_2 ):
        #check if solutions are in different quadrants. return the valid solution if so.
        def separate_quadrant_solution(solution_input_1, solution_input_2): 
            solution_1_quadrant = get_quadrant(solution_input_1[0], solution_input_1[1])
            solution_2_quadrant = get_quadrant(solution_input_2[0], solution_input_2[1])
            if solution_1_quadrant is not solution_2_quadrant:
                if solution_2_quadrant < solution_1_quadrant:
                    smallest_quadrant = [solution_2_quadrant, solution_input_2]
                    largest_quadrant = [solution_1_quadrant, solution_input_1]
                else:
                    smallest_quadrant = [solution_1_quadrant, solution_input_1]
                    largest_quadrant =  [solution_2_quadrant, solution_input_2]
                if (smallest_quadrant == 1) and (largest_quadrant == 4):
                    return largest_quadrant[1]
                else:
                    return smallest_quadrant[1]
            else:
                return None
        possible_solution = separate_quadrant_solution(solution_1, solution_2)
        if possible_solution is not None:
            return possible_solution
        #otherwise they're in the same quadrant, so whichever has the largest atan angle with respect to origin will be the valid solution
        else: 
            solution_1_angle = math.atan( solution_1[1]/solution_1[0])
            solution_2_angle = math.atan( solution_2[1]/solution_2[0])
            if solution_1_angle > solution_2_angle:
                return solution_1_angle
            else:
                return solution_2_angle

   # print(prox_coord_solutions)
    valid_prox_phalanx_coord = discard_illegal_coordinate_solutions( prox_coord_solutions[0], prox_coord_solutions[1] )
   # print(valid_prox_phalanx_coord)
   # print("==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    #2. , find the angle needed to move the proximal phalanx vector to be a distance of bone_length away from 
    #the destination x/y (because bone_length is the length of the middle phalanx bone)
    def vector_angle_2D(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    prox_angle_solution = vector_angle_2D([c_x_v1, c_y_v1], [ valid_prox_phalanx_coord[0], valid_prox_phalanx_coord[1] ])
    #2d vector angle is not sign-dependant (it just gives the angle between vectors), so we need to decide which way to rotate
    #the proximal phalanx:
    prox_rotation = determine_rotation_direction( valid_prox_phalanx_coord[0], valid_prox_phalanx_coord[1], d_x, d_y, prox_angle_solution)

    #3. , find the simulated coordinate for vector 2 if it was rigidly attached to vector 1. Use this coordinate to find
    #how much vector 2 should be rotated to reach the destination coordinate.
    sim_x_v2 =  c_x_v2*math.cos(prox_rotation) - c_y_v2*math.sin(prox_rotation) 
    sim_y_v2 =  c_x_v2*math.sin(prox_rotation) + c_y_v2*math.cos(prox_rotation)

    #4. , as mentioned, find the angle between the simulated coordinates and the destination coordinates. this is the 
    #amount to rotate vector 2.
    mid_angle_solution = vector_angle_2D( [sim_x_v2, sim_y_v2], [ d_x, d_y ])
    #same issue was proximal angle solution (no direction indication)
    mid_rotation = determine_rotation_direction( sim_x_v2, sim_y_v2, d_x, d_y, mid_angle_solution)
   # mid_rotation = determine_rotation_direction( c_x_v2, c_y_v2, d_x, d_y, mid_angle_solution)

    return_dict = {'prox_phalanx_coord': valid_prox_phalanx_coord, 'prox_angle_change': prox_rotation, 
                   'mid_phalanx_coord': [d_x, d_y], 'mid_angle_change': mid_rotation}
    return return_dict