import matplotlib.pyplot as plt
import math
from scipy.spatial import Voronoi, voronoi_plot_2d

def extractPath(path = None, firing_history = None):
    x_positions = []
    y_positions = []
    if firing_history != None:
        for t in range(len(firing_history)):
            x, y = firing_history[t][1]
            x_positions.append(x)
            y_positions.append(y)
    if path != None: 
        for t in range(len(path)):
            x, y = path[t][0]
            x_positions.append(x)
            y_positions.append(y)
    return(x_positions, y_positions)

def path(path=None, firing_history=None, path_x = None, path_y = None):
    if (path_x != None and path_y != None):
        x_positions = path_x
        y_positions = path_y
    elif (path != None): 
        x_positions, y_positions = extractPath(path=path)
    elif (firing_history != None):
        x_positions, y_positions = extractPath(firing_history=firing_history)
    plt.plot(x_positions, y_positions)
    plt.show()

def firing(firing_history):
    x_positions = []
    y_positions = []
    firing = []
    for t in range(len(firing_history)):
        x, y = firing_history[t][1]
        x_positions.append(x)
        y_positions.append(y)
        firing.append(firing_history[t][2])
    plt.scatter(x_positions, y_positions, c=firing)
    plt.show()

def firing_voronoi(positions, firing, title=""):
    """
    Plot firing rate as voronoi regions.
    """

    # compute voronoi regions
    vor = Voronoi(points=positions)

    # plot voronoi regions without borders
    voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_width=0.0)

    # map firing rate to colors
    mapper = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    colors = mapper.to_rgba(firing, norm=True)

    # color voronoi region by firing rate
    for point_region, color in zip(vor.point_region, colors):
        # exclude regions with infinite vertices
        if -1 not in vor.regions[point_region]:
            polygon = [vor.vertices[i] for i in vor.regions[point_region]]  # get vertices of region
            plt.fill(*zip(*polygon), facecolor=color, edgecolor=color, linewidth=0.1)

    # set lims to position range
    plt.xlim(positions[:, 0].min(), positions[:, 0].max())
    plt.ylim(positions[:, 1].min(), positions[:, 1].max())

    plt.title(title)
    plt.gcf().set_dpi(300)
    plt.gca().set_aspect('equal')
    plt.gcf().set_constrained_layout(True)
    plt.show()

def oscillatorActivitySpatially(firing_history, oscillator_history):
    x_positions = []
    y_positions = []
    oscillator_activity = []
    for t in range(len(firing_history)):
        x, y = firing_history[t][1]
        x_positions.append(x)
        y_positions.append(y)
        oscillator_activity.append(oscillator_history[t])
    plt.scatter(x_positions, y_positions, c=oscillator_activity)
    plt.show()

def oscillatorActivityTemporally(oscillator_history):
    t = range(len(oscillator_history))
    plt.plot(t, oscillator_history)
    plt.show()

def all(grid_cell):
    firing_history = grid_cell.firing_history
    somatic_activity = grid_cell.soma.activity_history
    n_dendrites = len(grid_cell.dendrites)
    x_positions = []
    y_positions = []
    firing = []
    for t in range(len(firing_history)):
        x, y = firing_history[t][1]
        x_positions.append(x)
        y_positions.append(y)
        firing.append(firing_history[t][2])
    fig, axs = plt.subplots(math.ceil((n_dendrites + 2) / 2), 2)
    axs[0, 0].scatter(x_positions, y_positions, c=firing)
    axs[0, 0].set_title('Firing')
    axs[0, 1].scatter(x_positions, y_positions, c=somatic_activity)
    axs[0, 1].set_title('Somatic Activity')
    row = 1
    column = 0
    for dendrite in range(n_dendrites):
        dendrite_activity = grid_cell.dendrites[dendrite].activity_history
        axs[row, column].scatter(x_positions, y_positions, c=dendrite_activity)
        axs[row, column].set_title('Dendritic Activity')
        row = math.ceil((dendrite + 2) / 2)
        column += 1
        column = column % 2
    fig.tight_layout()
    plt.show()

