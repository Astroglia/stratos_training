import numpy as np
import pickle
import time
import dataLoad
from LeapMotionConfig import subSampling, modifications

# loads all pickled data from data folder. return list of data, each list is a dictionary of feature matrices, motion data, and the feature matrices time deltas
unconfigured_data = dataLoad.load_directory_data('./data_folder/current')

print("KEYS IN DATA: ")
for key, value in unconfigured_data[0].items(): print (key)

#unwrap data (database storage requires formatting numpy arrays weirdly, so this needs to be done)
#goes through every set of data given.
feature_matrices = dataLoad.batch_unwrap_feamats( unconfigured_data )
motion_data      = dataLoad.batch_unwrap_motion_data( unconfigured_data )
time_deltas      = dataLoad.batch_unwrap_time_deltas( unconfigured_data )

# if there were multiple pickle files, then select all data from the first file loaded:
feature_matrices = feature_matrices[0]
motion_data      = motion_data[0] 
time_deltas      = time_deltas[0]

#Each feature matrix has multiple sets of recorded motion data (since a feature matrix is 200-1000 ms of data, we get 10 to 50 sets of motion data, because motion data is being recorded at ~20ms)
#This loop runs through all motion_data and converts sets of motion data to a 1:1 correspondance with their feature matrix
#motion data is also sometimes not recorded if no hand is detected by the leap motion controller, so it accounts for that as well.
motion_data_processed = [ ]
for i, matrix_list in enumerate(motion_data):
    lengths = map(len, matrix_list) #find if no data was recorded
    if( 0 not in lengths ):
        motion_data_processed.append( subSampling.list_to_single_matrix(matrix_list) ) #convert [ [motion_data_0A, motion_data_0B, .. ], [ motion_data1A, ... ], ... ] to [ motion_data0, motion_data1, ]
    else: #if no data was recorded, then zero everything out. TODO : not optimal (change to null and then remove all nulls).
        motion_data_processed.append(subSampling.get_LeapMotion_zero_matrix() )
        feature_matrices[i] = 0*feature_matrices[i]

#additional processing: round data to nearest 10 degrees, then convert the data set to either 1 (motion) or 0 (no motion)
for i in range(len(motion_data_processed)):
    motion_data_processed[i] = subSampling.round_to_nearest( motion_data_processed[i], interval=10)
    motion_data_processed[i] = subSampling.decimate_individual_joint_complexity(motion_data_processed[i], thresholds=[70, 40, 40 ,40 ,50]) #thresholds in degrees.

print("list lengths -- these should be the same :")
print(len(feature_matrices))
print(len(motion_data_processed))

from Visualization import singlePlot
plot_object = singlePlot.singlePlot(features=feature_matrices, motion_data=motion_data_processed, feature_count=14)
plot_object.select_channels([ 10 ]) #plot features calculated from the 10th channel in recording.
plot_object.plot_simple_threshold()
