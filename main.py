import numpy as np
import pickle
import time
import dataLoad
from LeapMotionConfig import subSampling

# loads all pickled data from data folder. return list of data
unconfigured_data = dataLoad.load_directory_data('data_folder')
feature_matrices = dataLoad.batch_unwrap_feamats( unconfigured_data )
motion_data = dataLoad.batch_unwrap_motion_data( unconfigured_data )

#just using a single file for example
feature_matrices = feature_matrices[0]
motion_data = motion_data[0]

#convert all motion data list of lists to a single list. Downsample each list to nearest 5 degrees.
one_to_one_motion_data = []
for i, matrix_list in enumerate(motion_data):
    lengths = map( len, matrix_list )
    if( 0 not in lengths ):
        single_matrix = subSampling.list_to_single_matrix(matrix_list)
        downsampled_matrix = subSampling.round_to_nearest(single_matrix, interval=5)
        one_to_one_motion_data.append(downsampled_matrix)
    else: #motion data was empty (bug in data collection). Zero-out associated feature matrix, create dummy motion data.
        one_to_one_motion_data.append( subSampling.get_LeapMotion_zero_matrix() )
        feature_matrices[i] = 0*feature_matrices[i]

# print one of each index
print( one_to_one_motion_data[56] )
print( feature_matrices[56])

#print length for verification
print(len(one_to_one_motion_data))
print(len(feature_matrices))
