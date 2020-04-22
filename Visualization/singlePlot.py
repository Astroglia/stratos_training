import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np 

class singlePlot:

    def __init__(self, features, motion_data, feature_count):
#         if len(features) != len(motion_data):
#             raise ValueError('labels must be the same size as features')
        self.features = features # list of [ 10, 224 ] matrices
        self.feature_count = feature_count
        self.data_to_plot = None
        self.motion_data_to_plot = None
        self.default_channel_plot()
        self.motion_data = motion_data # list of [ 4, 5 ] matrices.

    def plot_data(self):
        self.data_to_plot = np.moveaxis(self.get_data_to_plot()[0,:,:], 0, 1 )
        self.motion_data_to_plot = np.moveaxis(np.dstack(self.motion_data)[0,:,:], 0 , 1 )
        self.x_axis_time = range(0, len(self.data_to_plot))
        print(self.data_to_plot.shape)
        print(self.motion_data_to_plot.shape)
       # plt.style.use('dark_background') #pretty
        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        #plt.plot(self.data_to_plot)
        self.motion_data_to_plot = self.motion_data_to_plot*self.data_to_plot.max() #scale to max
        plt.fill_between(self.x_axis_time, self.motion_data_to_plot[:,0], 0, alpha=0.5, label='thumb')
        plt.fill_between(self.x_axis_time, self.motion_data_to_plot[:,1], 0, alpha=0.5, label='index')
        plt.fill_between(self.x_axis_time, self.motion_data_to_plot[:,2], 0, alpha=0.5, label='middle')
        plt.fill_between(self.x_axis_time, self.motion_data_to_plot[:,3], 0, alpha=0.5, label='ring')
        plt.fill_between(self.x_axis_time, self.motion_data_to_plot[:,4], 0, alpha=0.5, label='pinky')
        plt.legend(loc="upper left")
        plt.plot(self.data_to_plot)
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
        return np.dstack(data_to_plot)
        
    def default_channel_plot(self):
        total_channels = int(self.features[0].shape[1] / self.feature_count)
        self.channel_dict = { }
        for i in range(total_channels):
            self.channel_dict[i] = False    
    def select_channels(self, channels_to_plot):
        for i in channels_to_plot:
            self.channel_dict[i] = True
        print(self.channel_dict)