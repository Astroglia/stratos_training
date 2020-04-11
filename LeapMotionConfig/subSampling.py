#### Description
#### Functions within this file downsample leap motion data in various ways.
import numpy as np

#### REFERENCE MATRIX: 
#                [METACARPAL] [PROX PHALANX] [M. PHALANX] [D. PHALANX]
#               [[  0.          33.65723038  36.76741409  88.29069519] [THUMB]
#               [ 13.11555862  69.48578644 161.43367004 163.93757629]  [INDEX]
#               [  8.68810368  42.49485397 138.06262207 161.1791687 ]  [MIDDLE]
#               [ 10.69950581  30.49986267 118.01464844 158.56045532]  [RING]
#               [ 15.83293056  33.06761169  82.57316589 134.74368286]] [PINKY]

#Round entire matrix to the nearest interval number, e.g. :
# round_to_nearest(  [ 34.2, 39.2, 35.0, 12.2, 1.8, 2.4] , 5) --> ret: [ 35, 40, 35, 10, 0, 0 ]
#Accepts non-integer intervals.
def round_to_nearest( matrix, interval):
    matrix_to_return = matrix
    remainder_mat = matrix % interval # get how far away they are from an interval
    remainder_mat[ remainder_mat < (interval/2) ]*= -1 #e.g. interval = 3, number = 33.09 --> remainder is 0.09, so subtract 0.09.
    remainder_mat[ remainder_mat > (interval/2) ] = interval - remainder_mat[ remainder_mat > (interval/2) ] # modulus returns remainder left over, so get the opposite of that.
    matrix_to_return+= remainder_mat
    return matrix_to_return

def get_LeapMotion_zero_matrix():
    return np.array([5, 4])

#removes given row/column/removal type from matrix.
#Supported removal types:
    # Column Removals: 'METACARPAL', 'PROXIMAL_PHALANX', 'MIDDLE_PHALANX', 'DISTAL_PHALANX  
    #    Row Removals: 'THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY'
def remove_measurement(matrix, row=None, column=None, removal_type=None):
    if row is not None:
        matrix = np.delete(matrix, (row), axis=0 )
    if column is not None:
        matrix = np.delete(matrix, (column), axis=1)
    if removal_type is not None:
        removal_types = {'METACARPAL': [None, 0], 'PROXIMAL_PHALANX': [0, 1], 'MIDDLE_PHALANX': [0, 2], 'DISTAL_PHALANX': [0,3],
                            'THUMB': [0, None], 'INDEX': [1, None], 'MIDDLE': [2, None], 'RING': [3, None], 'PINKY': [4, None]  }
        row, col = removal_types[removal_type]
        matrix = remove_measurement(matrix, row=row, column=col, removal_type=None)
    return matrix    

#remove all joint information, reduce problem to general classification problem.
#Currently uses the PROXIMAL_PHALANX as reference for threshold. This is because it's the biggest indicator of if the finger is moving or not.
def decimate_joint_complexity(matrix, threshold=25):
    matrix = matrix[:, 1]
    matrix[ matrix > threshold ] = 1
    matrix[ matrix < threshold ] = 0
    return matrix

# weighted average on matrix list. e.g. for the number of elements in the list, apply a preferential weight in increasing order:
#  ( first element-> multiplied by 0), (second -> multiplied by 1, & so on)
# Reasoning: the last element in the matrix corresponds with the most recent feature data input (e.g. the 10th index). The 10th Index in the feature
# matrix is the most influential part of the matrix: new features are appended to the end of the feature matrix, so it would correspond to the newest movement.
# previous feature matrices are simply past data, and are probably used by the neural network to 'verify' movements (e.g. sustained movements).
def sequential_weighted_average(matrix_list):
    weighting = range(0, len(matrix_list) )
    mean_divisor = 0
    weighted_sum_matrix = 0
    for i, matrix in enumerate(matrix_list):
        mean_divisor+= weighting[i]
        weighted_sum_matrix+= matrix*weighting[i]
    return weighted_sum_matrix/mean_divisor

#converts multiple given matrices to a single matrix.
# Supported Conversion Types: 
    # 'sequential_weighted_average'
def list_to_single_matrix(matrix_list, conversion_type=None):
    conversion_types = {'sequential_weighted_average': sequential_weighted_average}
    if conversion_type is None:
        conversion_type = 'sequential_weighted_average'
    return conversion_types[conversion_type](matrix_list)