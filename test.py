# FOR DOCUMENTATION: read example.py.

####################################### TORUS MAPPING TESTS #######################################
import matplotlib.pyplot as plt, numpy as np
import pickle, time

import dataLoad
from LeapMotionConfig import subSampling, velocityConversion, torusMapping
from Visualization.Rigging import Rigging, RigPlotting

unconfigured_data = dataLoad.load_directory_data('./data_folder/current')
feature_matrices = dataLoad.batch_unwrap_feamats( unconfigured_data )
motion_data      = dataLoad.batch_unwrap_motion_data( unconfigured_data )
time_deltas      = dataLoad.batch_unwrap_time_deltas( unconfigured_data )
feature_matrices = feature_matrices[0]
motion_data      = motion_data[0] 
time_deltas      = time_deltas[0]

motion_data_processed = [ ]
for i, matrix_list in enumerate(motion_data):
    lengths = map(len, matrix_list) #find if no data was recorded
    if 0 not in lengths:
        motion_data_processed.append( subSampling.list_to_single_matrix(matrix_list) )
    else:     #TODO : not optimal (change to null and then remove all nulls).
        motion_data_processed.append(subSampling.get_LeapMotion_zero_matrix() )
        feature_matrices[i] = 0*feature_matrices[i]

torus_mapped_proximal = torusMapping.batch_multijoint_to_torus_space(motion_data_processed, return_type='PROXIMAL')
torus_mapped_mid = torusMapping.batch_multijoint_to_torus_space(motion_data_processed, return_type='MIDDLE')
torus_mapped_resultant = torusMapping.batch_multijoint_to_torus_space(motion_data_processed, return_type='RESULTANT')

torus_mapped_proximal = torus_mapped_proximal[1900:]
torus_mapped_mid = torus_mapped_mid[1900:]
torus_mapped_resultant = torus_mapped_resultant[1900:]

prox_index, mid_index, resultant_index = [ [],[],[] ]
for i in range(len(torus_mapped_proximal)):
    prox_index.append( torus_mapped_proximal[i]['INDEX'])
    mid_index.append( torus_mapped_mid[i]['INDEX'])
    resultant_index.append( torus_mapped_resultant[i]['INDEX'])

fingerModel = Rigging.TwoJointModel() #

# test_cases = [ [ 1.2, -0.2 ], [ 1.4, 0.4 ], [ -0.4, -1.2 ], [ 1, -0.8] ]
# for i, dest_coord in enumerate(test_cases): #resultant_index
#     decoded_map = torusMapping.two_joint_decode_torus_space_mapping( [fingerModel.get_v1_coords(), fingerModel.get_v2_coords()], dest_coord )
#     v1_rot = decoded_map['prox_angle_change']
#     v2_rot = decoded_map['mid_angle_change']
#     fingerModel.rigid_rotation(v1_rot, v2_rot)
#     print("DET : ", fingerModel.get_v2_coords())
#     print("ACT : ", dest_coord)
#     print("----")

################### TESTING OF TORUS DECODING
circle_bounds = [ 0.5, 1.0 , 1.5 ]
c_list = [ 'mediumspringgreen', 'crimson']
plot_methods = [ 'ACTUAL', 'PREDICTED' ]

RiggedModel = Rigging.TwoJointModel()
RiggedModelPlotter = RigPlotting.MultiRiggedVectorPlot(circle_boundaries=circle_bounds, color_list=c_list, vector_plot_methods=plot_methods)
for i, dest_coord in enumerate(resultant_index):
    # get the new vector rotations for v1 and v2 from single coordinate
    decoded_map = torusMapping.two_joint_decode_torus_space_mapping([RiggedModel.get_v1_coords(), RiggedModel.get_v2_coords()], dest_coord)
    v1_rotation, v2_rotation = [ decoded_map['prox_angle_change'], decoded_map['mid_angle_change']]

    #perform rotation on vectors.
    RiggedModel.rigid_rotation(v1_rotation, v2_rotation)

    #plot new vectors.
    v1_act, v2_act = [prox_index[i], mid_index[i]]
    act_xy = [{'x': v1_act[0], 'y': v1_act[1]}, {'x': v2_act[0], 'y': v2_act[1]}]
    RiggedModelPlotter.plot_new_vectors( [ act_xy, RiggedModel.get_all_coords_list_form() ] )

# for i in range(len(motion_data_processed)):
#   #  motion_data_processed[i] = subSampling.round_to_nearest( motion_data_processed[i], interval=1)
#     motion_data_processed[i] = motion_data_processed[i][1, :]


# from Visualization import singlePlot
# plot_object = singlePlot.singlePlot(features=feature_matrices, motion_data=motion_data_processed, feature_count=14)
# plot_object.select_channels([ 10 ]) #plot features calculated from the 10th channel in recording.
# plot_object.plot_simple_threshold()
