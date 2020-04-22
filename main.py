import numpy as np
import pickle
import time
import dataLoad
from LeapMotionConfig import subSampling, modifications

# loads all pickled data from data folder. return list of data
unconfigured_data = dataLoad.load_directory_data('./data_folder/current')
for key, value in unconfigured_data[0].items():
    print (key)

feature_matrices = dataLoad.batch_unwrap_feamats( unconfigured_data )
motion_data = dataLoad.batch_unwrap_motion_data( unconfigured_data )
time_deltas = dataLoad.batch_unwrap_time_deltas( unconfigured_data )

#just use data from a single file for example
feature_matrices = feature_matrices[0]
motion_data = motion_data[0] 
time_deltas = time_deltas[0]

#convert all motion data list of lists to a single list. Downsample each list to nearest 5 degrees.
motion_data_processed = []
for i, matrix_list in enumerate(motion_data):
    lengths = map( len, matrix_list )
    if( 0 not in lengths ):
        matrix = subSampling.list_to_single_matrix(matrix_list)
        matrix = subSampling.round_to_nearest(matrix, interval=10)
        matrix = subSampling.decimate_individual_joint_complexity(matrix, thresholds=[70, 40, 40 ,40 ,50])
       # matrix = subSampling.decimate_joint_complexity(matrix, threshold=45) #anything above 50 degrees in the proximal phalanx is considered movement.
    # velocity matrix conversions if wanted:
    #    if i == 0:  velocity_matrix_list = modifications.get_velocity_conversions(matrix_list, time_deltas[i], previous_matrix=None, previous_time_deltas=None)
    #    else:       velocity_matrix_list = modifications.get_velocity_conversions(matrix_list, time_deltas[i], previous_matrix=motion_data[i-1], previous_time_deltas=time_deltas[i-1])
    #    single_matrix = subSampling.list_to_single_matrix(velocity_matrix_list)
    #    downsampled_matrix = subSampling.round_to_nearest(single_matrix, interval=5)
        motion_data_processed.append(matrix)

    else: #motion data was empty (bug in data collection). Zero-out associated feature matrix, create dummy motion data.
        
        motion_data_processed.append( subSampling.get_LeapMotion_zero_matrix() )
        feature_matrices[i] = 0*feature_matrices[i]

# print one of each index
print( motion_data_processed[1160] )
print( feature_matrices[1160] )

# for i in range(500, 2500):
#      print(i, " " , motion_data_processed[i])

#print length for verification
print(len(motion_data_processed))
print(len(feature_matrices))
from Visualization import singlePlot

plot_object = singlePlot.singlePlot(features=feature_matrices, motion_data=motion_data_processed, feature_count=14)
plot_object.select_channels([ 10 ]) #plot channels from first channel.
plot_object.plot_data()