from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from tqdm import autonotebook as tqdm


class Visualizer:
    def __init__(self, states):
        self.states = states
        plt.ioff()
        self.fig, ax = plt.subplots()
        plt.ion()
        # ax.set_xticks(np.arange(0, states.field[1] + 1, 1))
        # ax.set_yticks(np.arange(0, states.field[0] + 1, 1))
        plt.grid()

        ax.set_xlim((0, states[0][1].field[1]))
        ax.set_ylim((0, states[0][1].field[0]))

        colors = ['b', 'r', 'g']
        cmap = matplotlib.colors.ListedColormap(colors)
        self.sc_plot = ax.scatter([], [], c=[], cmap=cmap)

    def draw(self):
        offsets = []
        colors = []
        for timestamp, state in self.states:
            points = np.asarray((list(state.pedestrians) + list(state.obstacles) + list(state.target)))
            points[..., 0], points[..., 1] = points[..., 1], points[..., 0].copy()
            offsets.append(points)
            color = np.concatenate((
                np.zeros(len(state.pedestrians)),
                np.zeros(len(state.obstacles)) + 1,
                np.zeros(len(state.target)) + 2,
            ))
            colors.append(color)

        def init(sc_plot):
            sc_plot.set_offsets([[0, 0]])
            return (sc_plot,)

        pbar = tqdm.tqdm(total=len(self.states))

        def animate(i, sc_plot):
            pbar.update(1)
            sc_plot.set_offsets(offsets[i])
            sc_plot.set_array(colors[i])
            return (sc_plot,)

        anim = animation.FuncAnimation(self.fig, partial(animate, sc_plot=self.sc_plot),
                                       init_func=partial(init, sc_plot=self.sc_plot),
                                       frames=len(self.states), interval=500, blit=True)
        return anim

    def get_video(self, anim):
        return HTML(anim.to_html5_video())
