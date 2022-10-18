import matplotlib.pyplot as plt

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

def seePath(path=None, firing_history=None, path_x = None, path_y = None):
    if (path_x != None and path_y != None):
        x_positions = path_x
        y_positions = path_y
    elif (path != None): 
        x_positions, y_positions = extractPath(path=path)
    elif (firing_history != None):
        x_positions, y_positions = extractPath(firing_history=firing_history)
    plt.plot(x_positions, y_positions)
    plt.show()

def seeFiring(firing_history):
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

def seeOscillatorActivitySpatially(firing_history, oscillator_history):
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

def seeOscillatorActivityTemporally(oscillator_history):
    t = range(len(oscillator_history))
    plt.plot(t, oscillator_history)
    plt.show()