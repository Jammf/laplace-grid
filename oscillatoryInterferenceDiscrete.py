import math
import random
import numpy as np
import csv

"""
Debug List
* Work out how to choose parameters to get a desired number of firing 
    fields in the arena
* Build out discrete version of code

EDIT LIST
* Change calls to somatic oscillator to require delta-t
"""

class Soma: 
    def __init__(self, 
                 phase_step, 
                 phase_offset):
        self.phase_step = phase_step
        self.phase = phase_offset
        self.phase_history = [self.phase]
        self.activity = math.cos(self.phase)
        self.activity_history = [self.activity]
    
    def step(self):
        self.phase += self.phase_step % (2 * math.pi)
        self.activity = math.cos(self.phase)
        self.phase_history.append(self.phase)
        self.activity_history.append(self.activity)

class Dendrite(Soma):
    def __init__(self,
                 preferred_heading,
                 phase_offset):
        Soma.__init__(self, 
                      phase_offset)
        self.preferred_heading = preferred_heading

    def step(self,
             speed,
             heading):
        phase_step = 

class Oscillator:
    def __init__(self, 
                 phase_offset): # Reconcile with theta
        self.activity_history = []
        self.phase_history  = []
        self.phase = phase_offset
        self.activity = math.cos(phase_offset)

    def update(self, dTh):
        #self.activity = math.cos(self.theta_angular_frequency * t)
        self.phase = self.phase_history[-1] + dTh
        self.activity = math.cos(self.phase)
        self.phase_history.append(self.phase)
        self.activity_history.append(self.activity)
        
    def activity(self)
        return self.activity

    def phase(self)
        return self.phase


class DendriticOscillator(Oscillator):
    def __init__(self, 
                 preferred_heading, 
                 phase_offset,
                 B):
        Oscillator.__init__(self, 
                            phase_offset)
        self.preferred_heading = preferred_heading
        self.B = B
        self.heading_factor_history = []

    def update(self, 
              speed, 
              heading):
        heading_factor = math.cos(heading - self.preferred_heading)
        angular_frequency = (self.theta_angular_frequency + 
                             (self.B * speed * heading_factor))
        Oscillator.update(self,angular_frequency)
               

class GridCell:
    def __init__(self, 
                 n_dendritic, # In tests set to 1
                 theta_angular_frequency, 
                 B, 
                 preferred_headings=None, # IN RADIANS
                 phase_offsets = None):
        self.firing_history = []
        self.soma = Oscillator(theta_angular_frequency)
        self.dendrites = []
        for n in range(n_dendritic): 
            # Default offset between preferred directions corresponds to even spacing
            preferred_heading = None
            if preferred_headings == None: 
                preferred_heading = n * 2 * math.pi / n_dendritic
            else: 
                preferred_heading = preferred_headings[n]
            phase_offset = 0
            if phase_offsets != None: 
                phase_offset = phase_offsets[n]
            dendrite = DendriticOscillator(theta_angular_frequency, 
                                           preferred_heading, 
                                           B, 
                                           phase_offset)
            self.dendrites.append(dendrite)
 
    def record(self, duration=None, arena_size=None, path=None, sampling_rate=None):
        delta_t = 1
        if sampling_rate != None: 
            delta_t = 1 / sampling_rate
        if path == None: 
            path = generatePath(duration, arena_size)
        if duration == None: 
            duration = len(path)
        for step in range(duration): 
            position, heading, speed = path[step]
            t = step * delta_t
            previous_t = t - delta_t
            somatic_activity = self.soma.check(t)
            dendritic_activity = []
            for n in range(len(self.dendrites)): 
                activity = self.dendrites[n].check(t, 
                                                   previous_t, 
                                                   speed, 
                                                   heading)
                dendritic_activity.append(activity)
            membrane_potential = somatic_activity + np.prod(dendritic_activity)
            firing_rate = membrane_potential
            # Heaviside function
            if firing_rate < 0:
                firing_rate = 0
            self.firing_history.append([t, position, firing_rate])

def generateNextStep(prev_position, prev_speed, prev_heading, max_speed, min_speed):
        position = (prev_position[0] + prev_speed * math.cos(prev_heading), 
                    prev_position[1] + prev_speed * math.sin(prev_heading))
        heading = (prev_heading + random.gauss(0, (math.pi / 3))) % (2 * math.pi)
        speed = prev_speed + random.gauss(0, (max_speed - min_speed) / 8) 
        return (position, heading, speed)

def generatePath(duration, arena_size):
    min_speed = arena_size * 0.05
    max_speed = arena_size * 0.2
    path = [[(0, 0), 0, 0]]
    for iteration in range(duration):
        t = iteration + 1
        prev_position, prev_heading, prev_speed = path[iteration] 
        position, heading, speed = generateNextStep(prev_position, 
                                                    prev_speed, 
                                                    prev_heading, 
                                                    max_speed, 
                                                    min_speed)
        while (abs(position[0]) > arena_size or abs(position[1]) > arena_size):
            position, heading, speed = generateNextStep(position, 
                                                        speed, 
                                                        heading, 
                                                        max_speed, 
                                                        min_speed)
        path.append([position, heading, speed])
    path = path[1:]
    return path

def generateConstantPath(duration):
    path = [[(0, 0), 0, 0]]
    for iteration in range(duration):
        position = (100 * iteration, 0)
        heading = 0
        speed = 100
        path.append([position, heading, speed])
    path = path[1:]
    return path

def ratPath(file_name, sampling_rate): 
    sample_interval = 1 / sampling_rate
    path = [] # (Coordinates), heading, speed
    with open(file_name, newline = '') as file: 
        trajectory = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in trajectory: 
            position = (row)
            if len(path) == 0: 
                path.append([position, 0, 0])
            else: 
                prev_position = path[len(path) - 1][0]
                displacement_vector = tuple(map(lambda after, before: after - before, 
                                        position, 
                                        prev_position))
                if displacement_vector[0] == 0:
                    if displacement_vector[1] > 0: 
                        heading = math.pi / 2
                    else: 
                        heading = path[len(path) - 1][1]
                else: 
                    heading = math.atan(displacement_vector[1] / displacement_vector[0])
                    if displacement_vector[0] < 0: 
                        heading += math.pi
                    elif displacement_vector[1] < 0: 
                        heading = 2 * math.pi + heading
                speed = math.hypot(displacement_vector[0], 
                                   displacement_vector[1]) / sample_interval
                path.append([position, heading, speed])
    path = path[1:] # Remove the first observation because we don't know the heading
    return path

def test_ratPath(file_name, path):
    with open(file_name, 'w') as file: 
        writer = csv.writer(file)
        for datum in path: 
            writer.writerow(datum) 