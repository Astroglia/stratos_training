import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np 

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

    def plot_motion_and_features(self):
        self.data_to_plot = np.moveaxis(self.get_data_to_plot(), 0, 1 )
        self.motion_data_to_plot = np.moveaxis(np.dstack(self.motion_data)[0,:,:], 0 , 1 )
        self.x_axis = range(0, len(self.data_to_plot))
        self.motion_data_to_plot = self.motion_data_to_plot*self.data_to_plot.max() # set feature data and motion data on the same scales.

        for i, label in enumerate(self.motion_labels):
            plt.fill_between(self.x_axis, self.motion_data_to_plot[:, i], alpha=0.5, label=label)
        for i, label in enumerate(self.data_labels):
            plt.plot(self.data_to_plot[:,i], alpha=0.8, label=label)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        
    def get_data_to_plot(self):
        data_to_plot = [ ]
        for i in self.features:
            most_recent = i[-1, :] #most up to date time in the feature matrix, which has shape [10, 224]
            temp = [ ]
            for j in range(len(self.channel_dict)):
                if self.channel_dict[j]:
                    channel = j
                    temp.append( most_recent[ int(channel*self.feature_count):(int(channel*self.feature_count + self.feature_count))])
            data_to_plot.append( np.dstack(temp) )
        return np.dstack(data_to_plot)[0,:,:]
        
    def default_channel_plot(self):
        total_channels = int(self.features[0].shape[1] / self.feature_count)
        self.channel_dict = { }
        for i in range(total_channels):
            self.channel_dict[i] = False    
    def select_channels(self, channels_to_plot):
        for i in channels_to_plot:
            self.channel_dict[i] = True
        print(self.channel_dict)