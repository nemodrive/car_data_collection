import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def sliding_plot(x, y, axis_lim=None, axis_type='equal', i_start=0, off_before=0, off_after=30):

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    l, = plt.plot(x, y, lw=2, color='red')
    #
    if axis_lim is not None:
        plt.axis(axis_lim)

    if axis_type is not None:
        plt.axis(axis_type)

    i_limits = [off_before, len(x) + off_after]

    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])

    sfreq = Slider(axfreq, 'Freq', int(i_limits[0]), i_limits[1], valinit=i_start)

    def update(val):
        i_start = int(sfreq.val)
        l.set_xdata(x[i_start - off_before: i_start + off_after].values)
        l.set_ydata(y[i_start - off_before: i_start + off_after].values)
        fig.canvas.draw_idle()
        print(i_start)

    sfreq.on_changed(update)

    plt.show()
