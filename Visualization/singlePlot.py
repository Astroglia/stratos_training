import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np 
import pickle

import matplotlib
print(matplotlib.get_backend())

class singlePlot:
    def __init__(self, features, motion_data, feature_count, data_labels=None, motion_labels=None):
        self.features = features # list of [ 10, 224 ] matrices
        self.feature_count = feature_count
        self.default_channel_plot() #automatically plot all channels
        self.motion_data = motion_data # list of [ 4, 5 ] matrices.
        self.data_to_plot = None
        self.motion_data_to_plot = None
        self.x_axis = None
        if motion_labels is None:   self.motion_labels = ['thumb', 'index','middle','ring','pinky']
        else:                       self.motion_labels = motion_labels
        if data_labels is None:     self.data_labels = ['ZC', 'SSC', 'WL', 'WAMP', 'MAB', 'MSQ', 'RMS', 'V3', 'LGDEC', 'DABS','MFL','MPR','MAVS','WMAB']
        else:                       self.data_labels = data_labels

    def plot_simple_threshold(self):
        self.data_to_plot = self.get_feature_data_to_plot()
        self.motion_data_to_plot = np.moveaxis(np.dstack(self.motion_data)[0,:,:], 0 , 1 )
        self.x_axis = range(0, len(self.data_to_plot))
        self.motion_data_to_plot = self.motion_data_to_plot*self.data_to_plot.max() # set feature data and motion data on the same scales.

        for i, label in enumerate(self.motion_labels):
            plt.fill_between(self.x_axis, self.motion_data_to_plot[:, i], alpha=0.5, label=label)
        for i, label in enumerate(self.data_labels):
            plt.plot(self.data_to_plot[:,i], alpha=0.8, label=label)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()

    def plot_single_joint(self):
        self.data_to_plot = self.get_feature_data_to_plot()
        self.motion_data_to_plot = singlePlot.scale_nested_max(self.motion_data)*self.data_to_plot.max()
        self.x_axis = range(0, len(self.data_to_plot))
        for i, label in enumerate(self.motion_labels):
            plt.fill_between(self.x_axis, self.motion_data_to_plot[:, i], alpha=0.5, label=label)
        for i, label in enumerate(self.data_labels):
            plt.plot(self.data_to_plot[:,i], alpha=0.8, label=label)

        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()  

    def plot_only_joints(self):    
        figure_handle = plt.figure()
      #  plt.style.use('dark_background')
        self.motion_data_to_plot = singlePlot.list_to_matrix(self.motion_data)
        self.x_axis = range(0, len(self.motion_data_to_plot))
        labels = [ 'METACARPAL', 'PROXIMAL PHALANX', 'MIDDLE PHALANX', 'DISTAL PHALANX' ]
        styles = [ '^', 's', 'o', '*']
        colors = [ 'deepskyblue', 'magenta', 'lime', 'red']
        for i in range(self.motion_data_to_plot.shape[1]):
            plt.plot(self.motion_data_to_plot[:,i], color=colors[i], alpha=0.8, label=labels[i])
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper right")
        plt.title("Index Finger Joint Angles")
        plt.xlabel("Discrete Dataset (200 milliseconds/dataset)")
        plt.ylabel("Joint Position With Respect to Palm (Degrees)")
        pickle.dump(figure_handle ,  open('plot.pickle',  'wb') )
        plt.show()  

    def get_feature_data_to_plot(self):
        data_to_plot = [ ]
        for i in self.features:
            most_recent = i[-1, :] #most up to date time in the feature matrix, which has shape [10, 224]
            temp = [ ]
            for j in range(len(self.channel_dict)):
                if self.channel_dict[j]:
                    channel = j
                    temp.append( most_recent[ int(channel*self.feature_count):(int(channel*self.feature_count + self.feature_count))])
            data_to_plot.append( np.dstack(temp) )
        return np.moveaxis(np.dstack(data_to_plot)[0,:,:], 0 , 1)
    
    ############ MOTION DATA FORMATTING ############
    @staticmethod
    def scale_nested_max(list_of_lists):
        matrix = np.dstack(list_of_lists)[0,:,:]
        max_arr = list ( np.max(matrix, axis=1) )
        for i, element in enumerate(max_arr):
            matrix[i, :] /= element
        return np.moveaxis(matrix, 0 , 1 )
    @staticmethod
    def list_to_matrix(list_of_lists):
        return np.moveaxis( np.dstack(list_of_lists)[0,:,:], 0, 1)

    ############ FEATURE MATRIX OPTIONS ############

    #plot feature data from X # of channels.
    def default_channel_plot(self):
        total_channels = int(self.features[0].shape[1] / self.feature_count)
        self.channel_dict = { }
        for i in range(total_channels):
            self.channel_dict[i] = False    
    #select the channels to plot feature data from.
    def select_channels(self, channels_to_plot):
        for i in channels_to_plot:
            self.channel_dict[i] = True
        print(self.channel_dict)