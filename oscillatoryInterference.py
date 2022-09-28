import math
import random
import numpy as np

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

    def check(self, 
              t, 
              speed, 
              heading):
        heading_factor = math.cos(heading - self.preferred_heading)
        angular_frequency = (self.theta_angular_frequency + 
                             self.B * speed * heading_factor)
        self.activity = math.cos(angular_frequency * t + self.phase_offset)
        self.activity_history.append(self.activity)
        return self.activity


class GridCell:
    def __init__(self, 
                 n_dendritic, 
                 theta_angular_frequency, 
                 B, 
                 preferred_headings=None, 
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

    def record(self, duration=None, arena_size=None, path=None):
        if path == None: 
            path = generatePath(duration, arena_size)
        for t in range(duration + 1): 
            position, heading, speed = path[t]
            somatic_activity = self.soma.check(t)
            dendritic_activity = []
            for n in range(len(self.dendrites)): 
                activity = self.dendrites[n].check(t, speed, heading)
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
    return path