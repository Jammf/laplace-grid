import math
import numpy as np

class Soma:
    def __init__(self, 
                 samples_per_second,
                 theta_frequency, 
                 phase_offset):
        # Radians traveled through cycle per second
        self.angular_frequency = theta_frequency * 2 * math.pi
        # Proportion of full cycle elapsed per sample
        self.phase_step = self.angular_frequency / samples_per_second 
        self.phase = phase_offset
        # Add the first phase to the history
        self.phase_history = [self.phase] 
        self.activity = math.cos(self.phase)
        # Add the first activity value to the history
        self.activity_history = [self.activity]

    # Advance the oscillator by one sampling window
    def step(self):
        self.phase += self.phase_step % (2 * math.pi)
        self.activity = math.cos(self.phase)
        self.phase_history.append(self.phase)
        self.activity_history.append(self.activity)

class Dendrite(Soma):
    def __init__(self,
                 samples_per_second, 
                 theta_frequency, 
                 preferred_heading,
                 phase_offset,
                 scaling_parameter):
        Soma.__init__(self, 
                      samples_per_second,
                      theta_frequency, 
                      phase_offset)
        self.preferred_heading = preferred_heading
        self.scaling_parameter = scaling_parameter
        self.samples_per_second = samples_per_second

    # Avance the oscillator by one sampling window
    def step(self,
             speed,
             heading):
        self.angular_frequency += speed * self.scaling_parameter * math.cos(heading - self.preferred_heading)
        self.phase_step = self.angular_frequency / self.samples_per_second 
        self.phase += self.phase_step % (2 * math.pi)
        self.activity = math.cos(self.phase)
        self.phase_history.append(self.phase)
        self.activity_history.append(self.activity)

# For only one dendrite
class GridCell:
    def __init__(self, 
                 samples_per_second,
                 theta_frequency, 
                 somatic_phase_offset,
                 scaling_parameter, 
                 n_dendritic,
                 preferred_headings=None, # IN RADIANS
                 dendritic_phase_offsets = None):
        self.firing_history = []
        self.soma = Soma(samples_per_second,
                         theta_frequency, 
                         somatic_phase_offset)
        self.dendrites = []
        for n in range(n_dendritic): 
            # Default offset between preferred directions corresponds to even spacing
            preferred_heading = preferred_headings[n]
            phase_offset = dendritic_phase_offsets[n]
            dendrite = Dendrite(samples_per_second, 
                                           theta_frequency, 
                                           preferred_heading,
                                           phase_offset,
                                           scaling_parameter)
            self.dendrites.append(dendrite)
 
    def record(self, path):
        for sample in range(len(path)): 
            position, heading, speed = path[sample]
            self.soma.step()
            somatic_activity = self.soma.activity
            dendritic_activity = []
            for n in range(len(self.dendrites)): 
                self.dendrites[n].step
                activity = self.dendrites[n].activity
                dendritic_activity.append(activity)
            membrane_potential = somatic_activity + np.prod(dendritic_activity)
            firing_rate = membrane_potential
            # Heaviside function
            if firing_rate < 0:
                firing_rate = 0
            self.firing_history.append([sample, position, firing_rate])