import matplotlib.pyplot as plt
import numpy as np

class MultiRiggedVectorPlot:
    def __init__(self, circle_boundaries, color_list, vector_plot_methods):
        self.previous_vector_list = [ ]
        self.previous_annotations = [ ]
        self.vector_plot_methods = vector_plot_methods
        self.color_list = color_list
        self.circle_boundaries = circle_boundaries
        self.init_plot()
        self.num_plotted = 0

    def init_plot(self):
        plt.figure(figsize=(10, 10))
        plt.style.use('dark_background')
        self.init_circle_plot()
        self.init_annotations()

    @staticmethod
    def circle_plot(radius, circular_radian_list):
        return radius*np.cos(circular_radian_list), radius*np.sin(circular_radian_list)

    def init_circle_plot(self):
        radians = np.arange(0, 6.28, 0.01)
        for index, RADIUS in enumerate(self.circle_boundaries):
            plt.plot(*MultiRiggedVectorPlot.circle_plot(RADIUS, radians), 'white')

    def remove_previous_vectors(self):
        if len(self.previous_vector_list) > 0:
            for POINTSET in self.previous_vector_list[0]:
                POINTSET.remove()
            self.previous_vector_list.pop(0)

    def init_annotations(self):
        self.previous_annotations = [plt.annotate('0', xy=(0.5, 0.5), xycoords='data', xytext=(0.8, 0.95), textcoords='axes fraction',
                         horizontalalignment='right', verticalalignment='top', )]
    def remove_previous_annotations(self):
        if len(self.previous_annotations) > 0:
            self.previous_annotations[0].remove()
            self.previous_annotations.pop(0)

    def plot_new_vectors(self, input_vectors):
        self.remove_previous_vectors()
        self.remove_previous_annotations()
        self.num_plotted+=1
        new_annotation = '|INFO| '
        temp_points_plotted_list = [ ]
        for A, POINTS in enumerate(input_vectors):
            plot_method = self.vector_plot_methods[A]
            color = self.color_list[A]

            base_vector, = plt.plot([ 0, POINTS[0]['x'] ], [0, POINTS[0]['y'] ], color=color)
            temp_points_plotted_list.append(base_vector)
            for next_vector in range(1, len(POINTS)):
                next_v_pts = None
                prev = POINTS[ next_vector-1 ]
                current = POINTS[next_vector]
                if plot_method == 'ACTUAL':
                    next_v_pts, = plt.plot([prev['x'], prev['x'] + current['x'] ], [prev['y'], prev['y'] + current['y'] ], color=color)
                    new_annotation = new_annotation + ' |ACTUAL: ' + color + "|"
                elif plot_method == 'PREDICTED':
                    next_v_pts, = plt.plot( [ prev['x'], current['x']], [ prev['y'], current['y'] ], color=color )
                    new_annotation = new_annotation + " |PREDICTED: " + color + "|"
                temp_points_plotted_list.append(next_v_pts)
        self.previous_vector_list.append( temp_points_plotted_list )
        new_annotation+= ' |COUNT: ' + str(self.num_plotted) + "|"
        self.previous_annotations = [plt.annotate(new_annotation, xy=(0.01, 0.6), xycoords='data', xytext=(0.01, 0.99), textcoords='axes fraction',
                      horizontalalignment='left', verticalalignment='top', )]
        plt.pause(0.001)