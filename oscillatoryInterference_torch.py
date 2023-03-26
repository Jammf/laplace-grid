"""
oscillatoryInterference
By Emma Alexandrov

Provides a GridCell class which implements the grid cell model described in 
Burgess, Barry, and O'Keefe's 2007 article in Hippocampus. 
"""
import math

import torch
from torch import nn, Tensor, vmap
from torch.nn.utils.rnn import pad_sequence


class Soma(nn.Module):
    """    
    Implements sub-threshold membrane potential oscillations. 

    ARGUMENTS: 
    - samples_per_second: float; the number of samples to be "recorded" from the 
        simulated somatic oscillator or the grid cell of which it is a part
    - theta_frequency: float (Hertz); the number of cycles per second 
        characteristic of the sub-threshold "theta" oscillation. According to 
        the paper, MEC Layer II cells show such oscillations at 8-9 Hz.

    ATTRIBUTES:
    - .angular_frequency: float (radians per second); the number of radians
        traveled through the oscillator's cycle per second
    - .phase_step: float; the proportion of the oscillator's full cycle 
        accomplished per sampling interval

    METHODS: 
    - .forward(phase): advance the state of the oscillator by one time step of
        length determined by the samples_per_second parameter
        ARGUMENTS:
        - phase: float (radians, [0, 2 * pi)); the state of the oscillator as
            position in a cycle of 2-pi radians
        RETURNS:
        - phase: float (radians, [0, 2 * pi)); the state of the oscillator as
            position in a cycle of 2-pi radians
        - activity: float ([0, 1]); the value of the membrane potential as due to
            the oscillation
    - .reset(phase_offset): return the state of the oscillator to its initial value
        ARGUMENTS:
        - phase_offset: float (radians); the point in the oscillator's cycle from
            which we wish the oscillation to begin at initialization
        RETURNS:
        - phase: float (radians, [0, 2 * pi)); the state of the oscillator as
            position in a cycle of 2-pi radians
        - activity: float ([0, 1]); the value of the membrane potential as due to
            the oscillation
    """
    def __init__(self,
                 samples_per_second,
                 theta_frequency):
        super().__init__()
        self.angular_frequency = theta_frequency * 2 * math.pi
        self.phase_step = self.angular_frequency / samples_per_second

    def forward(self, phase) -> tuple[Tensor, Tensor]:
        phase = (phase + self.phase_step) % (2 * math.pi)
        activity = torch.cos(phase)
        return phase, activity

    def reset(self, phase_offset: Tensor) -> tuple[Tensor, Tensor]:
        phase = phase_offset
        activity = torch.cos(phase)
        return phase, activity


class Dendrite(Soma):
    """
    Implements the oscillatory dendritic inputs whose frequencies vary as 
        a function of running speed and direction

    ARGUMENTS:
    - samples_per_second: float; the number of samples to be "recorded" from the
        simulated somatic oscillator or the grid cell of which it is a part
    - theta_frequency: float (Hertz); the number of cycles per second 
        characteristic of the sub-threshold "theta" oscillation. According to 
        the paper, MEC Layer II cells show such oscillations at 8-9 Hz.
    - scaling_parameter: float (radians per centimeter) the number of radians 
        through the cycle of the interference pattern (made by the summation 
        of the dendritic oscillator's activity with that of the somatic 
        oscillator) traveled per each centimeter of space that the simulated 
        animal traverses

    ATTRIBUTES: 
    - .phase_offset: float (radians); the point in the oscillator's cycle from 
        which we wish the oscillation to begin at initialization
    - .angular_frequency: float (radians); the number of radians traveled 
        through the oscillator's cycle per second
    - .phase_step: float; the proportion of the oscillator's full cycle 
        accomplished per sampling interval
    - .theta_angular_frequency: float (radians per second) the number of radians 
        traveled through the corresponding somatic oscillator's cycle per second
    - .scaling_parameter: float (radians per centimeter) the number of radians 
        through the cycle of the interference pattern (made by the summation 
        of the dendritic oscillator's activity with that of the somatic 
        oscillator) traveled per each centimeter of space that the simulated 
        animal traverses
    - .samples_per_second: float; the number of samples to be "recorded" from 
        the simulated somatic oscillator or the grid cell of which it is a part

    METHODS: 
    - .forward(speed, heading, phase, preferred_heading): advance the state of the
        oscillator by one time step of length determined by the samples_per_second
        parameter.
        ARGUMENTS:
        - speed: float (centimeters per second); the simulated animal's running 
            speed at time of sampling
        - heading: float (radians); the simulated animal's heading at time 
            of sampling
        - phase: float (radians, [0, 2 * pi)); the state of the oscillator as
            position in a cycle of 2-pi radians
        - preferred_heading: float (radians); the running direction we wish to
            provoke the biggest increase in the dendritic input's oscillatory
            frequency
        RETURNS:
        - phase: float (radians, [0, 2 * pi)); the state of the oscillator as
            position in a cycle of 2-pi radians
        - activity: float ([0, 1]); the value of the membrane potential as due to
            the oscillation
    - .reset(): return the state of the oscillator to its initial value
        ARGUMENTS:
        - phase_offset: float (radians); the point in the oscillator's cycle from
            which we wish the oscillation to begin at initialization
        RETURNS:
        - phase: float (radians, [0, 2 * pi)); the state of the oscillator as
            position in a cycle of 2-pi radians
        - activity: float ([0, 1]); the value of the membrane potential as due to
            the oscillation
    """
    def __init__(self,
                 samples_per_second: float,
                 theta_frequency: float,
                 scaling_parameter: float):
        super().__init__(samples_per_second, theta_frequency)
        self.theta_angular_frequency = theta_frequency * 2 * math.pi
        self.scaling_parameter = scaling_parameter
        self.samples_per_second = samples_per_second

    def forward(self,
             speed,
             heading,
             phase,
             preferred_heading):
        angular_frequency = self.theta_angular_frequency + \
                            (speed * self.scaling_parameter *
                             torch.cos(heading - preferred_heading))
        phase_step = angular_frequency / self.samples_per_second
        phase = (phase + phase_step) % (2 * math.pi)
        activity = torch.cos(phase)
        return phase, activity

    def reset(self, phase_offset: Tensor) -> tuple[Tensor, Tensor]:
        phase = phase_offset
        activity = torch.cos(phase)
        return phase, activity


class GridCell:
    """
    Implements the oscillatory interference model of grid cell firing.

    ARGUMENTS: 
    - samples_per_second: float; the number of samples to be "recorded" from the 
        simulated somatic oscillator or the grid cell of which it is a part
    - theta_frequency: float (Hertz); the number of cycles per second 
        characteristic of the sub-threshold "theta" oscillation. According to 
        the paper, MEC Layer II cells show such oscillations at 8-9 Hz. 
    - somatic_phase_offset: float (radians); the point in the somatic 
        oscillator's cycle from which we wish its oscillation to begin at 
        initialization
    - cm_per_cycle: float (centimeters); the number of centimeters desired 
        between firing fields
    - n_dendritic: int between 1 and 6, inclusive; the number of dendritic
        oscillators desired
    - offset_proportions: pair of floats between 0 and 1; how far through one
        cycle of the grid pattern we want the cell's firing field shifted, with
        respect to baseline
    - orientation: float (radians) between 0 and Pi / 3; how much we want the 
        grid pattern rotated, with respect to baseline

    ATTRIBUTES:
    - .scaling_parameter: float (radians per centimeter) the number of radians 
        through the cycle of the interference pattern (made by the summation 
        of the dendritic oscillator's activity with that of the somatic 
        oscillator) traveled per each centimeter of space that the simulated 
        animal traverses
    - .firing_history: list; a record of where the simulated animal has been 
        and what the simulated cell's firing rate was there. The entries are 
        lists of the form [sample number, [position x-coordinate, 
        position y-coordinate], firing rate]
    - .soma: instance of class Soma; the simulated grid cell's somatic 
        oscillator
    - .dendrites: list containing instances of class Dendrite; the simulated 
        grid cell's dendritic oscillators 

    METHODS:
    - .record(path): simulate the activity of the grid cell as the simulated 
        animal runs a specific, unbroken, course
        ARGUMENTS:
        - path: list of lists of the form [[position x-coordinate, position 
            y-coordinate], heading, speed] representing the simulated animal's 
            trajectory through the environment; the path must be continuous
    - .clear(): return all oscillators to their initial states and empty 
        all histories
    - .test(path): simulate the activity of the grid cell as the simulated 
        animal runs a specific, potentially broken, course
        ARGUMENTS:
        - path: list of list of lists of the form [[position x-coordinate, 
            position y-coordinate], heading, speed] representing the simulated 
            animal's trajectory through the environment; the path need not 
            be continuous, as each list of lists will be treated as its own 
            path separate from the others 

    """
    def __init__(self,
                 samples_per_second,
                 theta_frequency,
                 somatic_phase_offset,
                 cm_per_cycle,
                 n_dendritic, # Natural number in [1, 6]
                 offset_proportions: tuple[float, float], # Real number in [0, 1)
                 orientation # Real number in [0, 1) correponding to [0, math.pi / 3)
                 ):
        super().__init__()
        self.samples_per_second = samples_per_second
        self.theta_frequency = theta_frequency

        self.somatic_phase_offset = torch.tensor(somatic_phase_offset)
        self.scaling_parameter = 2 * math.pi * (1 / cm_per_cycle)

        self.soma = Soma(samples_per_second,
                         theta_frequency)

        a0, a60 = offset_proportions
        a120 = a60 - a0
        dendritic_phase_offsets = torch.tensor([a0, a60, a120, 1-a0, 1-a60, 1-a120])
        dendritic_phase_offsets = dendritic_phase_offsets[0:n_dendritic]
        self.dendritic_phase_offsets = dendritic_phase_offsets * 2 * math.pi

        # Default offset between preferred directions corresponds to even spacing
        preferred_heading = torch.arange(0, n_dendritic)
        self.dendritic_preferred_headings = preferred_heading + orientation * math.pi / 3

        self.dendrite = Dendrite(samples_per_second, theta_frequency, self.scaling_parameter)

    def step(self,
             somatic_phase: Tensor,
             dendritic_phases: Tensor,
             speed: Tensor,
             heading: Tensor
             ):
        vmap_dendrite_step = vmap(
            self.dendrite,
            in_dims=(
                None,  # speed (broadcast)
                None,  # heading (broadcast)
                0,     # state_phase (map over dendrite dim)
                0      # preferred_heading (map over dendrite dim)
            )
        )

        # update states
        somatic_phase, somatic_activity = self.soma(somatic_phase)
        dendritic_phases, dendritic_activities = vmap_dendrite_step(
            speed,
            heading,
            dendritic_phases,
            self.dendritic_preferred_headings
        )  # vmap over dendrites

        membrane_potential = torch.prod(somatic_activity + dendritic_activities)
        firing_rate = torch.relu(membrane_potential)
        return firing_rate, somatic_phase, dendritic_phases, somatic_activity, dendritic_activities

    def record(self, headings, speeds, somatic_phase, dendritic_phases) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        firing_history = []
        soma_phase_history = []
        soma_activity_history = []
        dendrite_phase_histories = []
        dendrite_activity_histories = []

        for (heading, speed) in zip(headings, speeds):
            firing_rate, somatic_phase, dendritic_phases, somatic_activity, dendritic_activities = self.step(
                somatic_phase,
                dendritic_phases,
                speed,
                heading
            )

            firing_history.append(firing_rate)
            soma_phase_history.append(somatic_phase)
            soma_activity_history.append(somatic_activity)
            dendrite_phase_histories.append(dendritic_phases)
            dendrite_activity_histories.append(dendritic_activities)

        firing_history = torch.stack(firing_history)
        soma_phase_history = torch.stack(soma_phase_history)
        soma_activity_history = torch.stack(soma_activity_history)
        dendrite_phase_histories = torch.stack(dendrite_phase_histories)
        dendrite_activity_histories = torch.stack(dendrite_activity_histories)

        return firing_history, soma_phase_history, soma_activity_history, dendrite_phase_histories, dendrite_activity_histories, somatic_phase, dendritic_phases

    def batch_record(self, segments):
        # convert to tensors, with segment as batch dimension
        positions_list = []
        headings_list = []
        speeds_list = []

        for segment in segments:
            # list of tuples -> tuple of lists
            positions, headings, speeds = zip(*segment)

            positions_list.append(torch.tensor(positions))
            headings_list.append(torch.tensor(headings))
            speeds_list.append(torch.tensor(speeds))

        # pad uneven sequences with nan
        batch_positions = pad_sequence(positions_list, batch_first=True, padding_value=float('nan'))
        batch_headings = pad_sequence(headings_list, batch_first=True, padding_value=float('nan'))
        batch_speeds = pad_sequence(speeds_list, batch_first=True, padding_value=float('nan'))

        n_batch = len(segments)

        # reset states
        somatic_phase, _ = self.soma.reset(self.somatic_phase_offset)
        batch_somatic_phase = somatic_phase.repeat(n_batch)  # repeat for each path segment

        dendritic_phases, _ = vmap(self.dendrite.reset)(self.dendritic_phase_offsets)
        batch_dendritic_phases = dendritic_phases.repeat(n_batch, 1)  # repeat for each path segment

        # record firing history
        batch_firing_history, \
            batch_soma_phase_history, \
            batch_soma_activity_history,\
            batch_dendrite_phase_histories,\
            batch_dendrite_activity_histories,\
            _, _ = vmap(self.record)(
                batch_headings,
                batch_speeds,
                batch_somatic_phase,
                batch_dendritic_phases
            )

        # flatten segments and sequences
        firing_history = batch_firing_history.flatten(0, 1)
        soma_phase_history = batch_soma_phase_history.flatten(0, 1)
        soma_activity_history = batch_soma_activity_history.flatten(0, 1)
        dendrite_phase_histories = batch_dendrite_phase_histories.flatten(0, 1)
        dendrite_activity_histories = batch_dendrite_activity_histories.flatten(0, 1)
        positions = batch_positions.flatten(0, 1)

        # remove nan padding
        nan_ixs = torch.isnan(firing_history)
        firing_history = firing_history[~nan_ixs]
        soma_phase_history = soma_phase_history[~nan_ixs]
        soma_activity_history = soma_activity_history[~nan_ixs]
        dendrite_phase_histories = dendrite_phase_histories[~nan_ixs]
        dendrite_activity_histories = dendrite_activity_histories[~nan_ixs]
        positions = positions[~nan_ixs]

        # verify positions are the same after nan padding was added and removed
        assert (torch.cat(positions_list) == positions).all()

        self.firing_history = firing_history
        self.soma_phase_history = soma_phase_history
        self.soma_activity_history = soma_activity_history
        self.dendrite_phase_histories = dendrite_phase_histories
        self.dendrite_activity_histories = dendrite_activity_histories
        self.positions = torch.cat(positions_list)

        return firing_history
