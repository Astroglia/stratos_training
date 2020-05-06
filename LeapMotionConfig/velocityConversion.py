#### Description
#### Functions within this file modify leapmotion data (conversion to difference matrices, normalization)
#### Data structure is mostly conserved.
import datetime
import numpy as np

#normalizes a given angle matrix to values between -1 and 1, using the given normalization_max for the maximum value.
def normalize_angle_matrix(matrix, normalization_max=360):
    return matrix / normalization_max

#convert given matrix list to rotational velocities (change in degrees per second)
#TODO grab the bone lengths from LeapMotion so true rotational velocity can be calculated
#uses the final index from the previous matrix list/previous time deltas to convert the first matrix in matrix_list to velocities.
# Returns:
    # list of rotational velocities in degrees/second.
        # if previous_matrix=None, return value has one less element than the input matrix list.
def get_velocity_conversions(matrix_list, time_deltas, previous_matrix=None, previous_time_deltas=None):
    return_arr = []
    time_delta_per_matrix = (time_deltas[1] - time_deltas[0]).total_seconds() / len(matrix_list)  #assumed constant rate of input (e.g. 45 fps) not necessarily true.
    if previous_matrix is not None:
        matrix_list.insert(0, previous_matrix[-1])
    for i in range(1, len(matrix_list)):
        rotational_velocity_matrix = rotational_velocity_calculation(matrix_list[i], matrix_list[i-1], time_delta_per_matrix) 
        return_arr.append( rotational_velocity_matrix )
    return return_arr

# given two angle matrices and the difference between them, calculate the measured velocity.
def rotational_velocity_calculation(current_matrix, previous_matrix, time_difference):
    return (current_matrix - previous_matrix) / time_difference

#first derivative calculation.
def diff_calc(matrix, previous_matrix):
    return matrix - previous_matrix