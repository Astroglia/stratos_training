import numpy as np
import pickle
import datetime
from os import listdir
from os.path import isfile, join

#loads all pickle data in a given directory.
# returns: list of de-pickled data.
def load_directory_data(directory_name):
    data_files = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]
    return_arr = []
    for file in data_files:
        with open('./' + directory_name + '/' + file , 'rb') as fileManagerObject:
            return_arr.append( pickle.load(fileManagerObject ))
    return return_arr

##### 
# Batch functions for unwrapping feature matrices/motion data/time deltas
# These functions will handle: the output from load_directory_data, which contains: [ data_dictionary, data_dictionary, data_dictionary... ]
#####
def batch_unwrap_feamats(raw_data):
    return_arr = []
    for i in raw_data:
        feature_matrix_shape = list( i['NEURAL_SHAPE'] )
        return_arr.append( unwrap_feamats( i['NEURAL_DATA'], feature_matrix_shape[0], feature_matrix_shape[1]) )
    return return_arr
def batch_unwrap_motion_data(raw_data):
    return_arr = []
    for i in raw_data:
        return_arr.append( unwrap_motion_data( i['MOTION_DATA'] ) )
    return return_arr
def batch_unwrap_time_deltas(raw_data):
    return_arr = []
    for i in raw_data: #each dataset given
        temp_arr = []
        for j in i['TIME_DELTAS']: #time deltas is a list of lists. each base element is a pair of time deltas for one feature matrix.
            temp_arr.append( unwrap_time_deltas(j) )
        return_arr.append(temp_arr)
    return return_arr


##### 
# Single functions for unwrapping data.
#####

# unwraps feature data from list of featureset queries
# returns: [  feature_matrix, feature_matrix ... ]
def unwrap_feamats(data, time_dimension, stacked_feature_dimension):
    split_array = np.split( data, int(data.shape[1]/stacked_feature_dimension),axis=1)
    return split_array

# unwraps motion data from list of motion data queries.
# returns: [  [numpy_array, numpy_array...] ...  ]
def unwrap_motion_data(data):
    final_list_arr = []
    for feature_set_angles in data: #set of data for one feature matrix.
        temp_list_1 = []
        for hand_angles in feature_set_angles: #set of angle data for all joints in the hand.
            interpreted_angles = eval(hand_angles)
            temp_list_2 = []
            for joint_angles in interpreted_angles: #joint angles for one finger.
                temp_list_2.append( joint_angles['data'] )

            temp_list_1.append( np.array(temp_list_2) )
    
        final_list_arr.append(temp_list_1)

    return final_list_arr

# unwraps single pair of time deltas with max cpu clock precision.
def unwrap_time_deltas(delta_list):
    return_arr = []
    return_arr.append( datetime.datetime.strptime( delta_list[0],'%Y-%m-%d %H:%M:%S.%f' ) )
    return_arr.append( datetime.datetime.strptime( delta_list[1],'%Y-%m-%d %H:%M:%S.%f' ) )
    return return_arr