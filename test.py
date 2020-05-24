# FOR DOCUMENTATION: read example.py.

import numpy as np
import pickle
import time
import dataLoad
import matplotlib.pyplot as plt
from LeapMotionConfig import subSampling, velocityConversion, torusMapping
from Visualization.Rigging import FingerRig

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
    if( 0 not in lengths ):
        motion_data_processed.append( subSampling.list_to_single_matrix(matrix_list) )
    #TODO : not optimal (change to null and then remove all nulls).
    else:
        motion_data_processed.append(subSampling.get_LeapMotion_zero_matrix() )
        feature_matrices[i] = 0*feature_matrices[i]

torus_mapped_proximal = torusMapping.batch_multijoint_to_torus_space(motion_data_processed, return_type='PROXIMAL')
torus_mapped_mid = torusMapping.batch_multijoint_to_torus_space(motion_data_processed, return_type='MIDDLE')
torus_mapped_resultant = torusMapping.batch_multijoint_to_torus_space(motion_data_processed, return_type='RESULTANT')

torus_mapped_proximal = torus_mapped_proximal[1900:]
torus_mapped_mid = torus_mapped_mid[1900:]
torus_mapped_resultant = torus_mapped_resultant[1900:]

prox_index =        [ ]
mid_index =         [ ]
resultant_index =   [ ]
for i in range(len(torus_mapped_proximal)):
    prox_index.append( torus_mapped_proximal[i]['INDEX'])
    mid_index.append( torus_mapped_mid[i]['INDEX'])
    resultant_index.append( torus_mapped_resultant[i]['INDEX'])

###### example decoding from given coordinates.
fingerModel = FingerRig.RiggedFinger()

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


def remove_pts(list_of_pts):
    list_of_pts[0].remove()
    list_of_pts.pop(0)
plt.style.use('dark_background')
def xy(r,phi):
    return r*np.cos(phi), r*np.sin(phi)
phis=np.arange(0,6.28,0.01)
r_outer =1.5 #outer boundary
r_inner = 0.5 #inner boundary
r_v1_circle = 1.0 #navigable space for vector 1.

plt.figure(figsize=(10, 10))
plt.plot(*xy(r_outer, phis), color='violet')
plt.plot(*xy(r_inner, phis), color='violet' )
plt.plot(*xy(r_v1_circle, phis), color='deeppink')

plt.scatter( 0, 0, s=20, color='violet')
past_points = [ ]
past_points_mid = [ ]
past_points_predictv1 = [ ]
past_points_predictv2 = [ ]
previous_annotation = [ plt.annotate( '0' , xy=(0.5, 0.5),  xycoords='data', xytext=(0.8, 0.95), textcoords='axes fraction',
                horizontalalignment='right', verticalalignment='top', )]


fingerModel = FingerRig.RiggedFinger()
previous_mid_phalanx_coord = [ 0, 0 ]
for i, dest_coord in enumerate(resultant_index):
    #prediction decoding.
    decoded_map = torusMapping.two_joint_decode_torus_space_mapping( [fingerModel.get_v1_coords(), fingerModel.get_v2_coords()], dest_coord )
    v1_rot = decoded_map['prox_angle_change']
    v2_rot = decoded_map['mid_angle_change']
    fingerModel.rigid_rotation(v1_rot, v2_rot)
    print("DET : ", fingerModel.get_v2_coords())
    print("ACT : ", dest_coord)
    print("----")

    #### plotting #### 

    #remove old points
    if len(past_points) > 5:
        remove_pts(past_points)
        remove_pts(past_points_mid)
        remove_pts(past_points_predictv1)
        remove_pts(past_points_predictv2)

    #plot new points
    prox_pts = prox_index[i]
    mid_pts = mid_index[i]
    print("current prox points, native: ", prox_pts)
    print("current mid points, native: ", mid_pts)
    prox_plotted_pts, = plt.plot( [0, prox_pts[0] ], [0, prox_pts[1]], color='mediumspringgreen', label=i)
    mid_plotted_pts, = plt.plot( [ prox_pts[0], prox_pts[0] + mid_pts[0]], [prox_pts[1], prox_pts[1] +  mid_pts[1] ],
     color='blueviolet', label=i)

    cpv1 = fingerModel.get_v1_coords()
    cpv2 = fingerModel.get_v2_coords()

    print("graphed mid points: " , [ prox_pts[0], prox_pts[0] + mid_pts[0]], [prox_pts[1], prox_pts[1] +  mid_pts[1] ])
    print("graphed predicted mid points: ", cpv2)
   # v1_scatter = plt.scatter( cpv2[0], cpv2[1], color='crimson')

    predict_plotted_ptsv1, = plt.plot( [ 0, cpv1[0]], [ 0, cpv1[1]], color='crimson' )
    predict_plotted_ptsv2, = plt.plot( [ cpv1[0], cpv2[0] ], [cpv1[1], cpv2[1]], color='crimson' )
    #add new points to old point list.
    past_points.append(prox_plotted_pts)
    past_points_mid.append(mid_plotted_pts)
    past_points_predictv1.append(predict_plotted_ptsv1)
    past_points_predictv2.append(predict_plotted_ptsv2)
    
    #update annotation info.
    curr_annotation = plt.annotate( "COUNTER: " + str(i) , xy=(0.5, 0.5),  xycoords='data', xytext=(0.8, 0.95), textcoords='axes fraction',
                horizontalalignment='right', verticalalignment='top', color='white', )
    previous_annotation[0].remove()
    previous_annotation.pop(0)
    previous_annotation.append(curr_annotation)

    plt.pause(0.001)


# for i in range(len(motion_data_processed)):
#   #  motion_data_processed[i] = subSampling.round_to_nearest( motion_data_processed[i], interval=1)
#     motion_data_processed[i] = motion_data_processed[i][1, :]


# from Visualization import singlePlot
# plot_object = singlePlot.singlePlot(features=feature_matrices, motion_data=motion_data_processed, feature_count=14)
# plot_object.select_channels([ 10 ]) #plot features calculated from the 10th channel in recording.
# plot_object.plot_simple_threshold()
