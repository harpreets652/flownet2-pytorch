import matplotlib.pyplot as plt

"""
https://stackoverflow.com/questions/10944621/dynamically-updating-plot-in-matplotlib
"""

plt.ion()


class DynamicUpdate:
    def __init__(self, title=None, x_label=None, y_label=None):
        # Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], 'o-')

        # Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)

        # Other stuff
        self.ax.grid()

        if title:
            self.ax.set_title(title)

        if x_label:
            self.ax.set_xlabel(x_label)

        if y_label:
            self.ax.set_ylabel(y_label)

    def on_running(self, x_data, y_data):
        # Update data (with the new _and_ the old points)
        self.lines.set_xdata(x_data)
        self.lines.set_ydata(y_data)

        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()

        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    @staticmethod
    def clear():
        plt.close('all')
        return
