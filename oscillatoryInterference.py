import math
import random
import numpy as np
import csv

"""
Debug List
* Try plotting the angular frequency of the D.O.s
V Double check that cosine in Python takes radians ––– confirmed
* Look at the "messy" data points
* There is an index error in the somatic vs. dendritic activity calls --- fix
* Review all units --- make sure values chosen are biologically plausible
* Check dendrite activity: 
    * Make sure dendrites are speeding up and slowing down their oscillations 
        appropriately
    * Make sure that dendrites are changing their oscillatory behavior 
        as I expect with respect to direction 
* Work out how to choose parameters to get a desired number of firing 
    fields in the arena
"""

class Oscillator:
    def __init__(self, 
                 theta_angular_frequency):
        self.theta_angular_frequency = theta_angular_frequency
        self.activity_history = []
        self.activity = 0

    def check(self, t):
        self.activity = math.cos(self.theta_angular_frequency * t)
        self.activity_history.append(self.activity)
        return self.activity


class DendriticOscillator(Oscillator):
    def __init__(self, 
                 theta_angular_frequency, 
                 preferred_heading, 
                 B, 
                 phase_offset):
        Oscillator.__init__(self, 
                            theta_angular_frequency)
        self.preferred_heading = preferred_heading
        self.B = B
        self.phase_offset = phase_offset
        self.activity_derivative_history = []

    def check(self, 
              t, 
              previous_t, 
              speed, 
              heading):
        delta_t = t - previous_t
        heading_factor = math.cos(heading - self.preferred_heading)
        angular_frequency = (self.theta_angular_frequency + 
                             (self.B * speed * heading_factor))
        previous_activity = 1 # On the zeroth run, activity starts at 1 (cosine)  
        previous_derivative = 0
        if len(self.activity_history) != 0:
            previous_activity = self.activity_history[-1]
            previous_derivative = self.activity_derivative_history[-1]
        proto_offset = math.acos(previous_activity)
        if previous_derivative > 0: 
            proto_offset = -1 * proto_offset
        offset = proto_offset / angular_frequency
        self.activity = math.cos(angular_frequency * (delta_t + offset))
        derivative = angular_frequency * -1 * math.sin(angular_frequency * (delta_t + offset))
        self.activity_history.append(self.activity)   
        self.activity_derivative_history.append(derivative)
        return self.activity

class GridCell:
    def __init__(self, 
                 n_dendritic, 
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