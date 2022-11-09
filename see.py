import matplotlib.pyplot as plt
import math

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

