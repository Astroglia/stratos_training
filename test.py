# FOR DOCUMENTATION: read example.py.

import numpy as np
import pickle
import time
import dataLoad
import matplotlib.pyplot as plt
from LeapMotionConfig import subSampling, velocityConversion, torusMapping

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

torus_mapped_motion_data = torusMapping.batch_multijoint_to_torus_space(motion_data_processed)
index_finger = [ ]
for i in torus_mapped_motion_data:
    index_finger.append( i['INDEX'])
    #print(i['INDEX'])

past_points = [ ]

plt.style.use('dark_background')
def xy(r,phi):
    return r*np.cos(phi), r*np.sin(phi)
phis=np.arange(0,6.28,0.01)
r_outer =1.0
r_inner = 0.5

plt.figure(figsize=(10, 10))

plt.plot(*xy(r_outer, phis), color='violet')
plt.plot(*xy(r_inner, phis), color='violet' )

plt.scatter( 0, 0, s=20, color='violet')
for i , pts in enumerate(index_finger):
    if len(past_points) > 5:
        past_points[0].remove()
        past_points.pop(0)
    curr_plot_pts, = plt.plot( [0, pts[0]], [0, pts[1] ]) #color='mediumspringgreen', label=i)
    #curr_plot_pts.remove()
   # print(len(curr_plot_pts))

   #curr_scatter_pts = plt.scatter(pts[0], pts[1], color='mediumspringgreen', label=i)
    past_points.append(curr_plot_pts)
    plt.pause(0.010)

# for i in range(len(motion_data_processed)):
#   #  motion_data_processed[i] = subSampling.round_to_nearest( motion_data_processed[i], interval=1)
#     motion_data_processed[i] = motion_data_processed[i][1, :]


# from Visualization import singlePlot
# plot_object = singlePlot.singlePlot(features=feature_matrices, motion_data=motion_data_processed, feature_count=14)
# plot_object.select_channels([ 10 ]) #plot features calculated from the 10th channel in recording.
# plot_object.plot_simple_threshold()
