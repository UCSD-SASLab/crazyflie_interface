from abc import ABC, abstractmethod
from .utils import diff_operators, quaternion

import math
import torch
import numpy as np
from multiprocessing import Pool
import torch.nn as nn
import scipy.io as spio
# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parametrized models

#test what


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Dynamics device {}".format(device))


class Dynamics(ABC):
    def __init__(self,
                 name: str, loss_type: str, set_mode: str,
                 state_dim: int, input_dim: int,
                 control_dim: int, disturbance_dim: int,
                 state_mean: list, state_var: list,
                 value_mean: float, value_var: float, value_normto: float,
                 deepReach_model: bool):
        self.name= name 
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean)
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepReach_model = deepReach_model

        assert self.loss_type in [
            'brt_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in [
            'reach', 'avoid', 'reach_avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + \
                str(len(state_descriptor)) + ' != ' + str(self.state_dim)

    # ALL METHODS ARE BATCH COMPATIBLE

    # set deepreach model. choices: "vanilla" (vanilla DeepReach V=NN(x,t)), diff (diff model V=NN(x,t) + l(x)), exact ( V=NN(x,t) + l(x))
    def set_model(self, deepreach_model):
        self.deepReach_model = deepreach_model

    # MODEL-UNIT CONVERSIONS 
    # convert model input (normalized) to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)
                          ) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord*1.0
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)
                          ) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepReach_model == 'diff':
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact':
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact_diff':
            # Another way to impose exact BC: V(x,t)= l(x) + NN(x,t) - NN(x,0)
            output0 = output[0].squeeze(dim=-1)
            output1 = output[1].squeeze(dim=-1)
            return (output0 - output1) * self.value_var / self.value_normto + self.boundary_fn(self.input_to_coord(input[0].detach())[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        if self.deepReach_model == 'exact_diff':

            dodi1 = diff_operators.jacobian(
                output[0], input[0])[0].squeeze(dim=-2)
            dodi2 = diff_operators.jacobian(
                output[1], input[1])[0].squeeze(dim=-2)

            dvdt = (self.value_var / self.value_normto) * dodi1[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi1.device)) * (dodi1[..., 1:]-dodi2[..., 1:])

            state = self.input_to_coord(input[0])[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
            return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

        dodi = diff_operators.jacobian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2

        elif self.deepReach_model == 'exact':

            dvdt = (self.value_var / self.value_normto) * \
                (input[..., 0]*dodi[..., 0] + output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto /
                    self.state_var.to(device=dodi.device)) * dodi[..., 1:]

        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # convert model io to real dv
    def io_to_2nd_derivative(self, input, output):
        hes = diff_operators.batchHessian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            vis_term1 = (self.value_var / self.value_normto /
                         self.state_var.to(device=hes.device))**2 * hes[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            vis_term2 = diff_operators.batchHessian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            hes = vis_term1 + vis_term2

        else:
            hes = (self.value_var / self.value_normto /
                   self.state_var.to(device=hes.device))**2 * hes[..., 1:]

        return hes

    def bound_control(self, control):
        return control

    def clamp_control(self, state, control):
        return control

    def clip_state(self, state):
        return state
    
    def clamp_state_input(self, state_input):
        return state_input
    
    def clamp_verification_state(self, state):
        return state
    # ALL FOLLOWING METHODS USE REAL UNITS
    @abstractmethod
    def periodic_transform_fn(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError

class VertDrone2D(Dynamics):
    control_dim = 1
    disturbance_dim = 0

    def __init__(self):
        self.gravity = 9.8  # g
        self.input_multiplier = 12.0  # K
        self.input_magnitude_max = 1.0  # u_max
        self.state_range_ = torch.tensor([[-4, 4], [-0.5, 3.5]]).to(device)  # v, z, k
        self.control_range_ = torch.tensor(
            [[-self.input_magnitude_max, self.input_magnitude_max]]
        ).to(device)
        self.eps_var_control = torch.tensor([2]).to(device)
        self.control_init = torch.ones(1).to(device) * self.gravity / self.input_multiplier

        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0

        super().__init__(
            name="VertDrone2D",
            loss_type="brt_hjivi",
            set_mode="avoid",
            state_dim=2,
            input_dim=3,  # input_dim of the NN = state_dim + 1 (time dim)
            control_dim=self.control_dim,
            disturbance_dim=self.disturbance_dim,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,  # we estimate the ground-truth value function to be within [-0.5, 1.5] w.r.t. the state_range_ we used
            value_var=1,  # Then value_mean = 0.5*(-0.5 + 1.5) and value_max = 0.5*(1.5 - -0.5)
            value_normto=0.02,  # Don't need any changes
            deepReach_model="exact",  # chioce ['vanilla', 'exact'],
        )

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def control_range(self, state):
        return [[-self.input_magnitude_max, self.input_magnitude_max]]

    def clamp_control(self, state, control):
        return self.bound_control(control)

    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[..., 0], self.control_range_[..., 1])

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist() 
        # Here we verify the training results using the training range itself, we can verify on a smaller range for "stiff" systems

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state 

    def periodic_transform_fn(self, input):
        return input.to(device)
    
    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.input_multiplier * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        return dsdt

    def boundary_fn(self, state):
        return -torch.abs(state[..., 1] - 1.5) + 1.5 # distance to ground (0m) and ceiling (3m)

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return  torch.abs(self.input_multiplier *dvds[..., 0]) * self.input_magnitude_max \
            - dvds[..., 0] * self.gravity \
            + dvds[..., 1] * state[..., 0]

    def optimal_control(self, state, dvds):
        return torch.sign(dvds[..., 0])[..., None]

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            'state_slices': [0, 1.5],
            'state_labels': ['v', 'z'],
            'x_axis_idx': 0, # which dim you want it to be the 
            'y_axis_idx': 1,
            'z_axis_idx': -1, # because there is only 2D
        }


class VertDrone2DWithDist(VertDrone2D):
    disturbance_dim = 1

    def __init__(self, max_disturbance: float):
        self.disturbance_magnitude_max = max_disturbance
        self.disturbance_range_ = torch.tensor(
            [-self.disturbance_magnitude_max, self.disturbance_magnitude_max]
        ).to(device)
        self.disturbance_init = torch.zeros(1).to(device)
        self.eps_var_disturbance = torch.tensor([2]).to(device)

        super().__init__()

    def disturbance_range(self, state):
        return [[-self.disturbance_magnitude_max, self.disturbance_magnitude_max]]

    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def bound_disturbance(self, disturbance):
        return torch.clamp(
            disturbance,
            self.disturbance_range_[..., 0],
            self.disturbance_range_[..., 1],
        )

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.input_multiplier * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0] + disturbance[..., 0]
        return dsdt

    def hamiltonian(self, state, dvds):
        return (
            torch.abs(self.input_multiplier * dvds[..., 0]) * self.input_magnitude_max
            - dvds[..., 0] * self.gravity
            + dvds[..., 1] * state[..., 0]
            - torch.abs(dvds[..., 1]) * self.disturbance_magnitude_max
        )

    def optimal_disturbance(self, state, dvds):
        return -(torch.sign(dvds[..., 1]) * self.disturbance_magnitude_max)[..., None]


class ParameterizedVertDrone2D(Dynamics):
    def __init__(self, gravity: float, input_multiplier: float, input_magnitude_max: float):
        self.gravity = gravity  # g
        self.input_multiplier = input_multiplier  # k_max
        self.input_magnitude_max = input_magnitude_max  # u_max
        self.state_range_ = torch.tensor([[-4, 4], [-0.5, 3.5], [0, self.input_multiplier]]).to(
            device
        )  # v, z, k
        self.control_range_ = torch.tensor(
            [[-self.input_magnitude_max, self.input_magnitude_max]]
        ).to(device)
        self.eps_var_control = torch.tensor([2]).to(device)
        self.control_init = torch.ones(1).to(device) * gravity / input_multiplier

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0

        super().__init__(
            name='ParameterizedVertDrone2D', loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',  # chioce ['vanilla', 'exact'],
        )

    def control_range(self, state):
        return [[-self.input_magnitude_max, self.input_magnitude_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    
    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def periodic_transform_fn(self, input):
        return input.to(device)
    
    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    # \dot k = 0
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        dsdt[..., 2] = 0
        return dsdt

    def boundary_fn(self, state):
        return -torch.abs(state[..., 1] - 1.5) + 1.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return  torch.abs(state[..., 2] *dvds[..., 0]) * self.input_magnitude_max \
            - dvds[..., 0] * self.gravity \
            + dvds[..., 1] * state[..., 0]

    def optimal_control(self, state, dvds):
        return torch.sign(dvds[..., 0])[..., None]

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            'state_slices': [0, 1.5, self.input_multiplier],
            'state_labels': ['v', 'z', 'k'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Dubins3D(Dynamics):
    def __init__(self, set_mode: str):
        self.goalR = 0.5
        self.velocity = 1.
        self.omega_max = 1.2
        self.state_range_ = torch.tensor([[-1, 1], [-1, 1], [-math.pi, math.pi]]).to(device)
        self.control_range_ = torch.tensor([[-self.omega_max, self.omega_max]]).to(device)
        self.eps_var_control = torch.tensor([1]).to(device)
        self.control_init = torch.zeros(1).to(device)
        self.set_mode=set_mode

        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0
        super().__init__(
            name="Dubins3D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=3, input_dim=5, control_dim=1, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact'
        )

    def control_range(self, state):
        return [[-self.omega_max, self.omega_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :3] = input[..., :3]
        transformed_input[..., 3] = torch.sin(input[..., 3]*self.state_var[-1])
        transformed_input[..., 4] = torch.cos(input[..., 3]*self.state_var[-1])
        return transformed_input.to(device)
    
    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity * torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt

    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - 0.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode =="avoid":
            return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2]) 
        elif self.set_mode =="reach":
            return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2]) 
        else:
            raise NotImplementedError
        
    def optimal_control(self, state, dvds):
        if self.set_mode =="avoid":
            return (self.omega_max * torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode =="reach":
            return -(self.omega_max * torch.sign(dvds[..., 2]))[..., None]
        else:
            raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Air3D(Dynamics): 
    #relative positions
    #evader and pursuer form of dubins3d
    def __init__(self, set_mode: str):
        self.goalR = 0.25
        self.evader_velocity = 0.6
        self.pursuer_velocity = 0.6

        self.omega_max = 2.0
        self.goalR = 0.25
        self.state_max = 1.5
        self.state_range_ = torch.tensor([[-self.state_max, self.state_max], [-self.state_max, self.state_max], [-math.pi, math.pi]]).to(device)
        self.control_range_ = torch.tensor([[-self.omega_max, self.omega_max]]).to(device)
        self.disturbance_range_ = torch.tensor([[-self.omega_max, self.omega_max]]).to(device)
        
        self.control_init = torch.zeros(1).to(device)
        self.disturbance_init = torch.zeros(1).to(device)
        self.set_mode=set_mode
        self.eps_var_control = torch.tensor([1]).to(device)
        self.eps_var_disturbance = torch.tensor([1]).to(device)


        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0

        super().__init__(
            name="Air3D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=3, input_dim=5, control_dim=1, disturbance_dim=1,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact'
        )

    def control_range(self, state):
        return self.control_range_.cpu().tolist()
    
    def disturbance_range(self, state):
        return self.disturbance_range_.cpu().tolist()
    
    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def bound_disturbance(self, disturbance):
        return torch.clamp(
            disturbance,
            self.disturbance_range_[..., 0],
            self.disturbance_range_[..., 1],
        )

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[..., 0], self.control_range_[..., 1])

    def clamp_control(self, state, control):
        return self.bound_control(control)
    
    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)

        transformed_input[..., :3] = input[..., :3]
        transformed_input[..., 3] = torch.sin(input[..., 3]*self.state_var[-1])
        transformed_input[..., 4] = torch.cos(input[..., 3]*self.state_var[-1])
        return transformed_input.to(device)
    
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)

        # state = [x1, x2, x3]
        # control = a (evader)
        # disturbance = b (pursuer)

        x1, x2, x3 = state[..., 0], state[..., 1], state[..., 2]
        a = control[..., 0]
        b = disturbance[..., 0]
        va = self.evader_velocity
        vb = self.pursuer_velocity

        dx1 = -va + vb * torch.cos(x3) + a * x2
        dx2 = vb * torch.sin(x3) - a * x1
        dx3 = b - a

        dsdt[..., 0] = dx1
        dsdt[..., 1] = dx2
        dsdt[..., 2] = dx3

        return dsdt

    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):

        #H(x,p)=f(x,a,b)⋅p= (−va​+vb​cosx3​+ax2​)p1​+(vb​sinx3​−ax1​)p2​+(b−a)p3

        x1 = state[..., 0]
        x2 = state[..., 1]
        x3 = state[..., 2]

        dVdx1 = dvds[..., 0]
        dVdx2 = dvds[..., 1]
        dVdx3 = dvds[..., 2]

        va = self.evader_velocity
        vb = self.pursuer_velocity
        w_max = self.omega_max

        base = (-va * dVdx1 +
                vb * torch.cos(x3) * dVdx1 +
                vb * torch.sin(x3) * dVdx2)

        # a (evader control) terms
        a_terms = x2 * dVdx1 - x1 * dVdx2 - dVdx3
        # b (pursuer control) term
        b_term = dVdx3

        if self.set_mode == "avoid":
            # max_a min_b H  ⇒ evader (a) maximizes, pursuer (b) minimizes
            return base + w_max * torch.abs(a_terms) - w_max * torch.abs(b_term)
        elif self.set_mode == "reach":
            # min_a max_b H ⇒ evader (a) minimizes, pursuer (b) maximizes
            return base - w_max * torch.abs(a_terms) + w_max * torch.abs(b_term)
        else:
            raise NotImplementedError
        
    def optimal_control(self, state, dvds):
        x1 = state[..., 0]
        x2 = state[..., 1]
        dVdx1 = dvds[..., 0]
        dVdx2 = dvds[..., 1]
        dVdx3 = dvds[..., 2]

        term = x2 * dVdx1 - x1 * dVdx2 - dVdx3

        if self.set_mode == "avoid":
            return (self.omega_max * torch.sign(term))[..., None]
        elif self.set_mode == "reach":
            return -(self.omega_max * torch.sign(term))[..., None]
        else:
            raise NotImplementedError


    def optimal_disturbance(self, state, dvds):
        dVdx3 = dvds[..., 2]

        if self.set_mode == "avoid":
            return -(self.omega_max * torch.sign(dVdx3))[..., None]
        elif self.set_mode == "reach":
            return (self.omega_max * torch.sign(dVdx3))[..., None]
        else:
            raise NotImplementedError

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x₁', 'x₂', r'$x_3$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }  


class Dubins6D(Dynamics):

    disturbance_dim = 1

    #evader and pursuer dubins dynamics with traditional positioning
    def __init__(self, collisionR: float, set_mode: str):
        self.evader_velocity = 0.6
        self.pursuer_velocity = 0.6
        self.omega_e_max = 2.0
        self.omega_p_max = 2.0
        self.state_max = 3.0

        self.goalR = collisionR  # collision radius

        self.state_range_ = torch.tensor([[-self.state_max, self.state_max], [-self.state_max, self.state_max], [-math.pi, math.pi],
                                          [-self.state_max, self.state_max], [-self.state_max, self.state_max], [-math.pi, math.pi]]).to(device)

        self.control_range_ = torch.tensor([[-self.omega_e_max, self.omega_e_max]]).to(device)
        self.disturbance_range_ = torch.tensor([[-self.omega_p_max, self.omega_p_max]]).to(device)

        self.control_init = torch.zeros(1).to(device)
        self.disturbance_init = torch.zeros(1).to(device)
        self.set_mode = set_mode

        #Sampling variance for dynamics
        desired_variance = self.omega_e_max
        self.eps_var_control = torch.tensor([desired_variance]).to(device)
        self.eps_var_disturbance = torch.tensor([desired_variance]).to(device)

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0

        super().__init__(
            name="Dubins6D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=6, input_dim=9, control_dim=1, disturbance_dim=self.disturbance_dim,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact'
        )
    
    def control_range(self, state):
        return self.control_range_.cpu().tolist()

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[..., 0], self.control_range_[..., 1])

    def clamp_control(self, state, control):
        return self.bound_control(control)
    
    def disturbance_range(self, state):
        return self.disturbance_range_.cpu().tolist()
    
    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def bound_disturbance(self, disturbance):
        return torch.clamp(
            disturbance,
            self.disturbance_range_[..., 0],
            self.disturbance_range_[..., 1],
        )
    
    

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # Wrap theta_e (index 2)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        # Wrap theta_p (index 5)
        wrapped_state[..., 5] = (wrapped_state[..., 5] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state
    
    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = input.shape[-1] + 2 

        transformed_input = torch.zeros(output_shape, device=input.device)

        # Copy x_e, y_e
        transformed_input[..., 0] = input[..., 0]  # time

        transformed_input[..., 1] = input[..., 1]  # x_e
        transformed_input[..., 2] = input[..., 2]  # y_e

        # Append sin and cos of θ_e
        theta_e = input[..., 3]
        transformed_input[..., 3] = torch.sin(theta_e * self.state_var[-1])
        transformed_input[..., 4] = torch.cos(theta_e * self.state_var[-1])

        # Copy x_p, y_p
        transformed_input[..., 5] = input[..., 4]  # x_p
        transformed_input[..., 6] = input[..., 5]  # y_p

        # Append sin and cos of θ_p
        theta_p = input[..., 6]
        transformed_input[..., 7] = torch.sin(theta_p * self.state_var[-1])
        transformed_input[..., 8] = torch.cos(theta_p * self.state_var[-1])

        return transformed_input
    
    
    
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)

        # State split
        xe, ye, theta_e = state[..., 0], state[..., 1], state[..., 2]
        xp, yp, theta_p = state[..., 3], state[..., 4], state[..., 5]

        # Control and disturbance
        ue = control[..., 0]       # evader's turn rate

        # Disturbance controls pursuer
        up = disturbance[..., 0]


        ve = self.evader_velocity
        vp = self.pursuer_velocity

        dsdt[..., 0] = ve * torch.cos(theta_e)
        dsdt[..., 1] = ve * torch.sin(theta_e)
        dsdt[..., 2] = ue

        dsdt[..., 3] = vp * torch.cos(theta_p)
        dsdt[..., 4] = vp * torch.sin(theta_p)
        dsdt[..., 5] = up

        return dsdt
    
    # def boundary_fn(self, state):
    #     xe, ye = state[..., 0], state[..., 1]
    #     xp, yp = state[..., 3], state[..., 4]
    #     dist = torch.sqrt((xe - xp) ** 2 + (ye - yp) ** 2)
    #     return dist - self.goalR

    def reach_fn(self, state):
    # Returns positive inside the reach set (e.g., inside state bounds), zero on boundary, negative outside
        evader_min = self.state_range_[[0, 1], 0].to(state.device)
        evader_max = self.state_range_[[0, 1], 1].to(state.device)
        evader_xy = state[..., [0, 1]]
        inside_min = evader_xy - evader_min
        inside_max = evader_max - evader_xy
        inside_dist = torch.min(inside_min, inside_max)
        return torch.min(inside_dist, dim=-1).values

    def avoid_fn(self, state):
        # Returns positive outside the avoid set (e.g., not in collision), zero on boundary, negative inside
        xe, ye = state[..., 0], state[..., 1]
        xp, yp = state[..., 3], state[..., 4]
        dist = torch.sqrt((xe - xp) ** 2 + (ye - yp) ** 2)
        return dist - self.goalR
    
    def boundary_fn(self, state):
        reach_constraint = self.reach_fn(state)
        avoid_constraint = self.avoid_fn(state)

        if self.set_mode == "avoid":
            return torch.minimum(avoid_constraint, reach_constraint)
            #return avoid_constraint
        elif self.set_mode == "reach":
            return reach_constraint
        elif self.set_mode == "reach_avoid":
            return torch.maximum(reach_constraint, -avoid_constraint)
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")

        
    
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    # def cost_fn(self, state_traj):
    #     return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def cost_fn(self, state_traj):
        if self.set_mode == 'avoid':
            return torch.min(self.boundary_fn(state_traj), dim=-1).values
        elif self.set_mode == 'reach':
            return torch.min(self.reach_fn(state_traj), dim=-1).values
        elif self.set_mode == 'reach_avoid':
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            # For each trajectory, at each time, clamp reach by the worst avoid violation so far
            worst_avoid = torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)
            clamped_reach = torch.clamp(reach_values, min=worst_avoid)
            return torch.min(clamped_reach, dim=-1).values
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")
    
    # def cost_fn(self, state_traj, gamma=0.9, recovery_weight=1.0):
    #     """
    #     Enhanced discounted cost function with collision recovery terms.
        
    #     Args:
    #         state_traj: Trajectory tensor [batch, timesteps, state_dim]
    #         gamma: Discount factor (0 < gamma < 1)
    #         recovery_weight: Weight for collision recovery penalty
        
    #     Returns:
    #         Enhanced cost that encourages recovery from collision states
    #     """
    #     # Get boundary values for each timestep: [batch, timesteps]
    #     boundary_values = self.boundary_fn(state_traj)  # [batch, timesteps]
        
    #     # Create discount factors for each timestep
    #     timesteps = boundary_values.shape[-1]
    #     discount_factors = gamma ** torch.arange(timesteps, device=boundary_values.device)
        
    #     # Apply discount to boundary values
    #     discounted_values = boundary_values * discount_factors.unsqueeze(0)
        
    #     # Primary cost: discounted boundary values
    #     if self.set_mode == "avoid":
    #         primary_cost = torch.min(discounted_values, dim=-1).values
    #     elif self.set_mode == "reach":
    #         primary_cost = torch.max(discounted_values, dim=-1).values
    #     else:
    #         raise NotImplementedError
        
    #     # Collision recovery penalty: encourage moving away from collision states
    #     # This provides gradient information even when in collision
    #     collision_penalty = torch.zeros_like(primary_cost)
        
    #     # For each trajectory, find the worst collision point and add recovery incentive
    #     for i in range(boundary_values.shape[0]):  # For each batch
    #         traj_boundary = boundary_values[i]  # [timesteps]
            
    #         # Find the worst (most negative) boundary value
    #         worst_collision = torch.min(traj_boundary)
            
    #         if worst_collision < 0:  # If there was a collision
    #             # Add penalty proportional to how bad the collision was
    #             # This encourages the optimizer to find trajectories that recover from collision
    #             collision_penalty[i] = worst_collision * recovery_weight
        
    #     # Combine primary cost and recovery penalty
    #     total_cost = primary_cost + collision_penalty
        
    #     return total_cost


    
    def hamiltonian(self, state, dvds):
        theta_e = state[..., 2]
        theta_p = state[..., 5]

        ve = self.evader_velocity
        vp = self.pursuer_velocity

        dV = dvds
        base = (
            ve * torch.cos(theta_e) * dV[..., 0] +
            ve * torch.sin(theta_e) * dV[..., 1] +
            vp * torch.cos(theta_p) * dV[..., 3] +
            vp * torch.sin(theta_p) * dV[..., 4]
        )

        a_term = dV[..., 2]  # derivative w.r.t theta_e
        b_term = dV[..., 5]  # derivative w.r.t theta_p

        if self.set_mode == "avoid":
            return base + self.omega_e_max * torch.abs(a_term) - self.omega_p_max * torch.abs(b_term)
        elif self.set_mode == "reach" or self.set_mode == "reach_avoid":
            return base - self.omega_e_max * torch.abs(a_term) + self.omega_p_max * torch.abs(b_term)
        else:
            raise NotImplementedError

    def optimal_control(self, state, dvds):
        dVdtheta_e = dvds[..., 2]
        if self.set_mode == "avoid":
            return self.omega_e_max * torch.sign(dVdtheta_e)[..., None]
        elif self.set_mode == "reach" or self.set_mode == "reach_avoid":
            return -self.omega_e_max * torch.sign(dVdtheta_e)[..., None]
        else:
            raise NotImplementedError
    
    def optimal_disturbance(self, state, dvds):
        dVdtheta_p = dvds[..., 5]

        if self.set_mode == "avoid":
            return -self.omega_p_max * torch.sign(dVdtheta_p)[..., None]
        elif self.set_mode == "reach" or self.set_mode == "reach_avoid":
            return self.omega_p_max * torch.sign(dVdtheta_p)[..., None]
        else:
            raise NotImplementedError
        
    def plot_config(self):
        def relative_state_fn(state):
            # state shape: [batch, 6]
            rel_x = state[..., 3] - state[..., 0]
            rel_y = state[..., 4] - state[..., 1]
            # Wrap relative angle between -pi and pi
            rel_theta = (state[..., 5] - state[..., 2] + math.pi) % (2 * math.pi) - math.pi
            return torch.stack([rel_x, rel_y, rel_theta], dim=-1)
        
        return {
            'state_slices': [0, 0, 0, 0, 0, 0],   # dummy slices, not actually used
            'state_labels': ['x_p - x_e', 'y_p - y_e', r'$\theta_p - \theta_e$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
            'fixed_evader_state': [0.0, 0.0, 0.0],
            'state_transform': relative_state_fn  # function to get relative states
        }
    
class Dubins6DNoDist(Dubins6D):

    def __init__(self, set_mode: str):
        super().__init__(set_mode)
        
        # Override disturbance-specific attributes
        self.omega_p_max = 0.0  # no disturbance, pursuer control zeroed
        
        self.disturbance_range_ = torch.tensor([[0.0, 0.0]]).to(device)
        self.disturbance_init = torch.zeros(0).to(device)  # zero-dim disturbance
        
        self.control_init = torch.zeros(1).to(device)  # evader control same as parent
        self.eps_var_control = torch.tensor([1]).to(device)
        self.eps_var_disturbance = torch.tensor([1]).to(device)
        
        self.disturbance_dim = 0

    # Override methods related to disturbance to disable disturbance
    
    def disturbance_range(self, state):
        return self.disturbance_range_.cpu().tolist()  # always zero range
    
    def clamp_disturbance(self, state, disturbance):
        # No disturbance to clamp
        return torch.zeros_like(disturbance)

    def bound_disturbance(self, disturbance):
        # No disturbance allowed
        return torch.zeros_like(disturbance)

    def optimal_disturbance(self, state, dvds):
        # No disturbance
        # Return tensor with correct shape on same device as dvds
        batch_size = dvds.shape[0] if len(dvds.shape) > 1 else 1
        return torch.zeros(batch_size, 1, device=dvds.device)

    def dsdt(self, state, control, disturbance=None):
        # Override dsdt: ignore disturbance input since no disturbance
        dsdt = torch.zeros_like(state)

        xe, ye, theta_e = state[..., 0], state[..., 1], state[..., 2]
        xp, yp, theta_p = state[..., 3], state[..., 4], state[..., 5]

        ue = control[..., 0]  # evader turn rate
        up = torch.zeros_like(ue)  # pursuer turn rate zero since no disturbance

        ve = self.evader_velocity
        vp = self.pursuer_velocity

        dsdt[..., 0] = ve * torch.cos(theta_e)
        dsdt[..., 1] = ve * torch.sin(theta_e)
        dsdt[..., 2] = ue

        dsdt[..., 3] = vp * torch.cos(theta_p)
        dsdt[..., 4] = vp * torch.sin(theta_p)
        dsdt[..., 5] = up

        return dsdt


class Dubins9D(Dynamics):
    """
    9D Dubins system: 1 evader, 2 pursuers.
    State: [x_e, y_e, theta_e, x_p1, y_p1, theta_p1, x_p2, y_p2, theta_p2]
    Control: [omega_e] (evader)
    Disturbance: [omega_p1, omega_p2] (pursuers)
    """
    disturbance_dim = 2

    def __init__(self, set_mode: str):
        self.evader_velocity = 0.6
        self.pursuer_velocity = 0.6
        self.omega_e_max = 2.0
        self.omega_p_max = 2.0
        self.state_max = 1.5
        self.goalR = 0.25  # collision radius

        self.state_range_ = torch.tensor(
            [[-self.state_max, self.state_max], [-self.state_max, self.state_max], [-math.pi, math.pi],
             [-self.state_max, self.state_max], [-self.state_max, self.state_max], [-math.pi, math.pi],
             [-self.state_max, self.state_max], [-self.state_max, self.state_max], [-math.pi, math.pi]]
        ).to(device)

        self.control_range_ = torch.tensor([[-self.omega_e_max, self.omega_e_max]]).to(device)
        self.disturbance_range_ = torch.tensor([[-self.omega_p_max, self.omega_p_max], [-self.omega_p_max, self.omega_p_max]]).to(device)

        self.control_init = torch.zeros(1).to(device)
        self.disturbance_init = torch.zeros(self.disturbance_dim).to(device)
        self.set_mode = set_mode

        desired_variance = 2.0
        self.eps_var_control = torch.tensor([desired_variance]).to(device)
        self.eps_var_disturbance = torch.tensor([desired_variance]).to(device)

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0

        super().__init__(
            name="Dubins9D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=9, input_dim=13, control_dim=1, disturbance_dim=self.disturbance_dim,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact'
        )

    def dsdt(self, state, control, disturbance):
        """
        state: (..., 9)
        control: (..., 1) [omega_e]
        disturbance: (..., 2) [omega_p1, omega_p2]
        """
        dsdt = torch.zeros_like(state)
        # Evader
        x_e, y_e, theta_e = state[..., 0], state[..., 1], state[..., 2]
        # Pursuer 1
        x_p1, y_p1, theta_p1 = state[..., 3], state[..., 4], state[..., 5]
        # Pursuer 2
        x_p2, y_p2, theta_p2 = state[..., 6], state[..., 7], state[..., 8]

        omega_e = control[..., 0]
        omega_p1 = disturbance[..., 0]
        omega_p2 = disturbance[..., 1]

        ve = self.evader_velocity
        vp = self.pursuer_velocity

        dsdt[..., 0] = ve * torch.cos(theta_e)
        dsdt[..., 1] = ve * torch.sin(theta_e)
        dsdt[..., 2] = omega_e

        dsdt[..., 3] = vp * torch.cos(theta_p1)
        dsdt[..., 4] = vp * torch.sin(theta_p1)
        dsdt[..., 5] = omega_p1

        dsdt[..., 6] = vp * torch.cos(theta_p2)
        dsdt[..., 7] = vp * torch.sin(theta_p2)
        dsdt[..., 8] = omega_p2

        return dsdt

    def boundary_fn(self, state):
        # Minimum distance to either pursuer
        xe, ye = state[..., 0], state[..., 1]
        xp1, yp1 = state[..., 3], state[..., 4]
        xp2, yp2 = state[..., 6], state[..., 7]
        dist1 = torch.sqrt((xe - xp1) ** 2 + (ye - yp1) ** 2)
        dist2 = torch.sqrt((xe - xp2) ** 2 + (ye - yp2) ** 2)
        return torch.minimum(dist1, dist2) - self.goalR

    def hamiltonian(self, state, dvds):
        ve = self.evader_velocity
        vp = self.pursuer_velocity
        theta_e = state[..., 2]
        theta_p1 = state[..., 5]
        theta_p2 = state[..., 8]

        dV = dvds
        base = (
            ve * torch.cos(theta_e) * dV[..., 0] +
            ve * torch.sin(theta_e) * dV[..., 1] +
            vp * torch.cos(theta_p1) * dV[..., 3] +
            vp * torch.sin(theta_p1) * dV[..., 4] +
            vp * torch.cos(theta_p2) * dV[..., 6] +
            vp * torch.sin(theta_p2) * dV[..., 7]
        )

        a_term = dV[..., 2]  # dV/dtheta_e
        b1_term = dV[..., 5]  # dV/dtheta_p1
        b2_term = dV[..., 8]  # dV/dtheta_p2

        if self.set_mode == "avoid":
            return base + self.omega_e_max * torch.abs(a_term) - self.omega_p_max * (torch.abs(b1_term) + torch.abs(b2_term))
        elif self.set_mode == "reach":
            return base - self.omega_e_max * torch.abs(a_term) + self.omega_p_max * (torch.abs(b1_term) + torch.abs(b2_term))
        else:
            raise NotImplementedError

    def optimal_control(self, state, dvds):
        dVdtheta_e = dvds[..., 2]
        if self.set_mode == "avoid":
            return self.omega_e_max * torch.sign(dVdtheta_e)[..., None]
        elif self.set_mode == "reach":
            return -self.omega_e_max * torch.sign(dVdtheta_e)[..., None]
        else:
            raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        dVdtheta_p1 = dvds[..., 5]
        dVdtheta_p2 = dvds[..., 8]
        if self.set_mode == "avoid":
            d1 = -self.omega_p_max * torch.sign(dVdtheta_p1)
            d2 = -self.omega_p_max * torch.sign(dVdtheta_p2)
        elif self.set_mode == "reach":
            d1 = self.omega_p_max * torch.sign(dVdtheta_p1)
            d2 = self.omega_p_max * torch.sign(dVdtheta_p2)
        else:
            raise NotImplementedError
        return torch.stack([d1, d2], dim=-1)

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # Wrap all headings to [-pi, pi]
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        wrapped_state[..., 5] = (wrapped_state[..., 5] + math.pi) % (2 * math.pi) - math.pi
        wrapped_state[..., 8] = (wrapped_state[..., 8] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def control_range(self, state):
        return self.control_range_.cpu().tolist()

    def disturbance_range(self, state):
        return self.disturbance_range_.cpu().tolist()

    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[..., 0], self.control_range_[..., 1])

    def bound_disturbance(self, disturbance):
        return torch.clamp(disturbance, self.disturbance_range_[:, 0], self.disturbance_range_[:, 1])

    def clamp_control(self, state, control):
        return self.bound_control(control)

    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def periodic_transform_fn(self, input):
        # We'll assume time is the first entry, then 9 state variables
        output_shape = list(input.shape)
        output_shape[-1] = input.shape[-1] + 3  # add 3 for sin/cos for each angle

        transformed_input = torch.zeros(output_shape, device=input.device)

        # Copy time and positions
        transformed_input[..., 0] = input[..., 0]  # time
        transformed_input[..., 1] = input[..., 1]  # x_e
        transformed_input[..., 2] = input[..., 2]  # y_e

        # θ_e
        theta_e = input[..., 3]
        transformed_input[..., 3] = torch.sin(theta_e * self.state_var[2])
        transformed_input[..., 4] = torch.cos(theta_e * self.state_var[2])

        # x_p1, y_p1
        transformed_input[..., 5] = input[..., 4]
        transformed_input[..., 6] = input[..., 5]

        # θ_p1
        theta_p1 = input[..., 6]
        transformed_input[..., 7] = torch.sin(theta_p1 * self.state_var[5])
        transformed_input[..., 8] = torch.cos(theta_p1 * self.state_var[5])

        # x_p2, y_p2
        transformed_input[..., 9] = input[..., 7]
        transformed_input[..., 10] = input[..., 8]

        # θ_p2
        theta_p2 = input[..., 9]
        transformed_input[..., 11] = torch.sin(theta_p2 * self.state_var[8])
        transformed_input[..., 12] = torch.cos(theta_p2 * self.state_var[8])

        return transformed_input

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):

        # Existing: min distance to either pursuer
        base_cost = torch.min(self.boundary_fn(state_traj), dim=-1).values
        # Coordination: penalize pursuers being too close
        optimal_dist = 0.5
        w = 0.2  # Tune this weight
        xp1, yp1 = state_traj[..., 3], state_traj[..., 4]
        xp2, yp2 = state_traj[..., 6], state_traj[..., 7]
        pursuer_dist = torch.sqrt((xp1 - xp2) ** 2 + (yp1 - yp2) ** 2)
        coordination_penalty = -w * (pursuer_dist - optimal_dist) ** 2
        return base_cost + coordination_penalty


    def plot_config(self):
        def relative_state_fn(state):
            # state shape: [batch, 9]
            rel_x1 = state[..., 3] - state[..., 0]
            rel_y1 = state[..., 4] - state[..., 1]
            rel_theta1 = (state[..., 5] - state[..., 2] + math.pi) % (2 * math.pi) - math.pi
            rel_x2 = state[..., 6] - state[..., 0]
            rel_y2 = state[..., 7] - state[..., 1]
            rel_theta2 = (state[..., 8] - state[..., 2] + math.pi) % (2 * math.pi) - math.pi
            return torch.stack([rel_x1, rel_y1, rel_theta1, rel_x2, rel_y2, rel_theta2], dim=-1)
        return {
            'state_slices': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x_e', 'y_e', r'$\theta_e$', 'x_p1', 'y_p1', r'$\theta_{p1}$', 'x_p2', 'y_p2', r'$\theta_{p2}$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
            'state_transform': relative_state_fn
        }

class Drone6DWithDist(Dynamics):
    def __init__(self, thrust_max: float, disturbance_max: float, set_mode: str):
        self.state_dim = 6
        self.control_dim = 3
        self.disturbance_dim = 3
        #self.disturbance_dim = 0

        self.input_multiplier = thrust_max  # K (gain factor)
        self.sideways_multiplier = 2.0
        self.control_max = 1.0  # u_max (normalized control bound)
        self.disturbance_max = disturbance_max
        self.Gz = -9.81
        self.max_v = 2.0
        self.collisionR = 0.50

        # State: [p_x, v_x, p_y, v_y, p_z, v_z]
        state_range_ = torch.tensor([
            [-2, 2], [-self.max_v, self.max_v],  # p_x, v_x
            [-2, 2], [-self.max_v, self.max_v],  # p_y, v_y
            [-0.5, 3.5], [-self.max_v, self.max_v],  # p_z, v_z
        ])
        control_range_ = torch.tensor([
            [-self.control_max, self.control_max],  # a_x
            [-self.control_max, self.control_max],  # a_y
            [0.0, self.control_max],  # a_z
        ])
        disturbance_range_ = torch.tensor([
            [-self.disturbance_max, self.disturbance_max],  # d_x
            [-self.disturbance_max, self.disturbance_max],  # d_y
            [-self.disturbance_max, self.disturbance_max],  # d_z
        ])

        box_bounds_ = torch.tensor([
            [-1.5, 1.5], [-self.max_v, self.max_v],
            [-1.5, 1.5], [-self.max_v, self.max_v],
            [0.0, 3.0],  [-self.max_v, self.max_v],
        ])

        state_mean_ = (state_range_[:, 0] + state_range_[:, 1]) / 2.0
        state_var_ = (state_range_[:, 1] - state_range_[:, 0]) / 2.0

        super().__init__(
            name="Drone6DWithDist", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=6, input_dim=7, control_dim=self.control_dim, disturbance_dim=self.disturbance_dim,
            state_mean=state_mean_.tolist(),
            state_var=state_var_.tolist(),
            value_mean=0.0,
            value_var=0.5,
            value_normto=0.02,
            deepReach_model='exact'
        )
        self.state_range_ = state_range_.to(device)
        self.control_range_ = control_range_.to(device)
        self.box_bounds_ = box_bounds_.to(device)
        self.disturbance_range_ = disturbance_range_.to(device)
        self.control_init = torch.tensor([0, 0, -self.Gz / self.input_multiplier]).to(device)
        self.disturbance_init = torch.zeros(3).to(device)
        self.eps_var_control = torch.tensor([self.sideways_multiplier, self.sideways_multiplier, self.input_multiplier]).to(device)
        self.eps_var_disturbance = torch.ones(3).to(device) * self.disturbance_max

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        
        # Use different multipliers for different directions
        dsdt[..., 1] = self.sideways_multiplier * control[..., 0] + disturbance[..., 0]  # x
        dsdt[..., 3] = self.sideways_multiplier * control[..., 1] + disturbance[..., 1]  # y  
        dsdt[..., 5] = self.input_multiplier * control[..., 2] + disturbance[..., 2] + self.Gz  # z

        # dsdt[..., 1] = self.sideways_multiplier * control[..., 0]   # x
        # dsdt[..., 3] = self.sideways_multiplier * control[..., 1]  # y  
        # dsdt[..., 5] = self.input_multiplier * control[..., 2] + self.Gz  # z
        
        dsdt[..., 0] = state[..., 1]  # p_x_dot = v_x
        dsdt[..., 2] = state[..., 3]  # p_y_dot = v_y
        dsdt[..., 4] = state[..., 5]  # p_z_dot = v_z
        return dsdt
    
    def hamiltonian(self, state, dvds):
        v = state[..., [1, 3, 5]]
        dVdp = dvds[..., [0, 2, 4]]
        dVdv = dvds[..., [1, 3, 5]]
        ham = (v * dVdp).sum(-1)
        
        # Add gravity term
        ham += dVdv[..., 2] * self.Gz
        
        # Control terms with correct multipliers
        if self.set_mode == 'avoid':
            # Control terms with proper scaling
            ham += (self.sideways_multiplier * torch.abs(dVdv[..., 0]) * self.control_max + 
                    self.sideways_multiplier * torch.abs(dVdv[..., 1]) * self.control_max + 
                    self.input_multiplier * torch.abs(dVdv[..., 2]) * self.control_max)

            ham -= self.disturbance_max * torch.abs(dVdv).sum(-1)
        elif self.set_mode == 'reach':
            # Control terms with proper scaling
            ham -= (self.sideways_multiplier * torch.abs(dVdv[..., 0]) * self.control_max + 
                    self.sideways_multiplier * torch.abs(dVdv[..., 1]) * self.control_max + 
                    self.input_multiplier * torch.abs(dVdv[..., 2]) * self.control_max)

            ham += self.disturbance_max * torch.abs(dVdv).sum(-1)
        
        return ham

    def dist_to_cylinder(self, state, a, b):
        """
        Computes the signed distance from the drone's position to a vertical cylinder
        centered at (a, b) in the x/y plane, with radius self.collisionR.
        Returns positive outside, zero on the surface, negative inside.
        """
        px = state[..., 0]  # p_x
        py = state[..., 2]  # p_y
        # Distance in x/y plane from drone to cylinder center
        dist_xy = torch.sqrt((px - a) ** 2 + (py - b) ** 2)
        # Signed distance to cylinder surface
        return dist_xy - self.collisionR
    

    # def boundary_fn(self, state):
    #     if self.set_mode=='avoid':
    #         return self.dist_to_cylinder(state,0.0,0.0)
    #     elif self.set_mode=='reach':
    #         return -self.dist_to_cylinder(state,5.0,5.0)
    #     else:
    #         raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")
    

    def boundary_fn(self, state):
        # state: (..., 6) - [p_x, v_x, p_y, v_y, p_z, v_z]
        # Box: px, py in [-1.5, 1.5], pz in [0, 3]
        px, py, pz = state[..., 0], state[..., 2], state[..., 4]

        box_bounds = self.box_bounds_.to(state.device)
        

        x_min, x_max = box_bounds[0, 0], box_bounds[0, 1]
        y_min, y_max = box_bounds[2, 0], box_bounds[2, 1]
        z_min, z_max = box_bounds[4, 0], box_bounds[4, 1]

        # Compute per-dimension signed distances to box faces
        dx_min = px - x_min
        dx_max = x_max - px
        dy_min = py - y_min
        dy_max = y_max - py
        dz_min = pz - z_min
        dz_max = z_max - pz

        # Inside: minimum distance to any face (negative inside, zero on surface)
        inside_dist = torch.min(torch.stack([dx_min, dx_max, dy_min, dy_max, dz_min, dz_max], dim=-1), dim=-1).values

        # For outside: compute the per-dimension "over" (how far outside the box in each dim)
        over_x = torch.clamp(px - x_max, min=0) + torch.clamp(x_min - px, min=0)
        over_y = torch.clamp(py - y_max, min=0) + torch.clamp(y_min - py, min=0)
        over_z = torch.clamp(pz - z_max, min=0) + torch.clamp(z_min - pz, min=0)
        # Norm of the "over" vector gives Euclidean distance outside
        outside_dist = torch.norm(torch.stack([over_x, over_y, over_z], dim=-1), dim=-1)

        # If all inside (all distances to faces > 0), use inside_dist; else use outside_dist
        is_inside = (dx_min > 0) & (dx_max > 0) & (dy_min > 0) & (dy_max > 0) & (dz_min > 0) & (dz_max > 0)
        return torch.where(is_inside, inside_dist, -outside_dist)

        # center = torch.tensor([0.0, 0.0, 1.5], device=state.device)
        # pos = torch.stack([px, py, pz], dim=-1)
        # dist = torch.norm(pos - center, dim=-1)
        # radius = 1.5

        # pos_sdf = radius - dist

        # return pos_sdf
       

    def equivalent_wrapped_state(self, state):
        # No periodicity, just return a clone
        return torch.clone(state)
    
    def periodic_transform_fn(self, input):
        return input.to(device)
    
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    
    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()
    
    def control_range(self, state):
        return self.control_range_.tolist()

    def disturbance_range(self, state):
        return self.disturbance_range_.tolist()

    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[:, 0], self.control_range_[:, 1])

    def bound_disturbance(self, disturbance):
        return torch.clamp(disturbance, self.disturbance_range_[:, 0], self.disturbance_range_[:, 1])
    
    def clamp_control(self, state, control):
        return self.bound_control(control)
    
    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def cost_fn(self, state_traj):
    # Use boundary function for consistency
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    
    def optimal_control(self, state, dvds):
        dVdv = dvds[..., [1, 3, 5]]
        if self.set_mode == 'avoid':
            # Scale control by multipliers
            control = torch.zeros_like(dVdv)
            control[..., 0] = self.control_max * torch.sign(dVdv[..., 0])
            control[..., 1] = self.control_max * torch.sign(dVdv[..., 1])
            control[..., 2] = self.control_max * torch.sign(dVdv[..., 2])
        elif self.set_mode == 'reach':
            # Scale control by multipliers
            control = torch.zeros_like(dVdv)
            control[..., 0] = -self.control_max * torch.sign(dVdv[..., 0])
            control[..., 1] = -self.control_max * torch.sign(dVdv[..., 1])
            control[..., 2] = -self.control_max * torch.sign(dVdv[..., 2])
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")
        return control

    def optimal_disturbance(self, state, dvds):
        dVdv = dvds[..., [1, 3, 5]]
        if self.set_mode == 'avoid':
            # Scale disturbance by multipliers to match dynamics
            disturbance = torch.zeros_like(dVdv)
            disturbance[..., 0] = -self.disturbance_max * torch.sign(dVdv[..., 0])
            disturbance[..., 1] = -self.disturbance_max * torch.sign(dVdv[..., 1])
            disturbance[..., 2] = -self.disturbance_max * torch.sign(dVdv[..., 2])
        elif self.set_mode == 'reach':
            # Scale disturbance by multipliers to match dynamics
            disturbance = torch.zeros_like(dVdv)
            disturbance[..., 0] = self.disturbance_max * torch.sign(dVdv[..., 0])
            disturbance[..., 1] = self.disturbance_max * torch.sign(dVdv[..., 1])
            disturbance[..., 2] = self.disturbance_max * torch.sign(dVdv[..., 2])
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")

        return disturbance

    #For cylinder plotting
    # def plot_config(self):
    #     return {
    #         'state_slices': [0.0, -1.43, 0.0, -1.2, 1.5, -0.1],  # [p_x, v_x, p_y, v_y, p_z, v_z]
    #         'state_labels': ['p_x', 'v_x', 'p_y', 'v_y', 'p_z', 'v_z'],
    #         'x_axis_idx': 0,  # p_x
    #         'y_axis_idx': 2,  # p_y
    #         'z_axis_idx': 4,  # p_z
    #     }
    
    # For vertical box plotting
    # def plot_config(self):
    #     return {
    #         'state_slices': [0.0, -0.8, 0.0, -1.2, 1.5, 0.0],  # [p_x, v_x, p_y, v_y, p_z, v_z]
    #         'state_labels': ['p_x', 'v_x', 'p_y', 'v_y', 'p_z', 'v_z'],
    #         'x_axis_idx': 5,  # p_x
    #         'y_axis_idx': 4,  # p_y
    #         'z_axis_idx': 0,  # p_z
    #     }
    
    # #For horizontal box plotting
    def plot_config(self):
        return {
            'state_slices': [0.0, -0.8, 0.0, -1.2, 1.5, 0.0],  # [p_x, v_x, p_y, v_y, p_z, v_z]
            'state_labels': ['p_x', 'v_x', 'p_y', 'v_y', 'p_z', 'v_z'],
            'x_axis_idx': 0,  # p_x
            'y_axis_idx': 2,  # p_y
            'z_axis_idx': 4,  # p_z
        }

class Drone6DNoDist(Drone6DWithDist):
    def __init__(self, collisionR: float, thrust_max: float, set_mode: str):
        super().__init__(collisionR, thrust_max, set_mode)
        self.disturbance_dim = 0

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        
        # Use different multipliers for different directions
        dsdt[..., 1] = self.sideways_multiplier * control[..., 0] 
        dsdt[..., 3] = self.sideways_multiplier * control[..., 1]   
        dsdt[..., 5] = self.input_multiplier * control[..., 2] + self.Gz  # z

        # dsdt[..., 1] = self.sideways_multiplier * control[..., 0]   # x
        # dsdt[..., 3] = self.sideways_multiplier * control[..., 1]  # y  
        # dsdt[..., 5] = self.input_multiplier * control[..., 2] + self.Gz  # z
        
        dsdt[..., 0] = state[..., 1]  # p_x_dot = v_x
        dsdt[..., 2] = state[..., 3]  # p_y_dot = v_y
        dsdt[..., 4] = state[..., 5]  # p_z_dot = v_z
        return dsdt

    def optimal_disturbance(self, state, dvds):
        return 0


class DronePursuitEvasion12D(Dynamics):
    """
    12D Drone pursuit-evasion system: 1 pursuer, 1 evader.
    State: [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z]
    Control: [a1_x, a1_y, a1_z] (evader control)
    Disturbance: [a2_x, a2_y, a2_z] (pursuer control)
    """
    disturbance_dim = 3

    def __init__(self, collisionR: float, thrust_max: float, set_mode: str):
        self.state_dim = 12
        self.control_dim = 3
        self.disturbance_dim = 3
        self.input_multiplier = thrust_max
        self.sideways_multiplier = 2.0
        self.control_max = 1.0
        self.disturbance_max = 1.0
        self.Gz = -9.81
        self.max_v = 2.0
        self.capture_radius = collisionR  # Distance for capture

        # State: [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z]
        state_range_ = torch.tensor([
            # Drone 1 (evader)
            [-2.5, 2.5], [-self.max_v, self.max_v],  # p1_x, v1_x
            [-2.5, 2.5], [-self.max_v, self.max_v],  # p1_y, v1_y
            [-0.5, 3.5], [-self.max_v, self.max_v],  # p1_z, v1_z
            # Drone 2 (pursuer)
            [-2.5, 2.5], [-self.max_v, self.max_v],  # p2_x, v2_x
            [-2.5, 2.5], [-self.max_v, self.max_v],  # p2_y, v2_y
            [-0.5, 3.5], [-self.max_v, self.max_v],  # p2_z, v2_z
        ])
        
        control_range_ = torch.tensor([
            [-self.control_max, self.control_max],  # a1_x
            [-self.control_max, self.control_max],  # a1_y
            [0.0, self.control_max],        # a1_z
        ])
        
        disturbance_range_ = torch.tensor([
            [-self.disturbance_max, self.disturbance_max],  # a2_x
            [-self.disturbance_max, self.disturbance_max],  # a2_y
            [0.0, self.disturbance_max],        # a2_z
        ])

        box_bounds_ = torch.tensor([
            [-2.0, 2.0], [-self.max_v, self.max_v],
            [-2.0, 2.0], [-self.max_v, self.max_v],
            [0.0, 3.0],  [-self.max_v, self.max_v],
        ])

        state_mean_ = (state_range_[:, 0] + state_range_[:, 1]) / 2.0
        state_var_ = (state_range_[:, 1] - state_range_[:, 0]) / 2.0

        super().__init__(
            name="DronePursuitEvasion12D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=12, input_dim=13, control_dim=3, disturbance_dim=3,
            state_mean=state_mean_.tolist(),
            state_var=state_var_.tolist(),
            value_mean=0.0,
            value_var=0.5,
            value_normto=0.02,
            deepReach_model='exact'
        )
        
        self.state_range_ = state_range_.to(device)
        self.control_range_ = control_range_.to(device)
        self.disturbance_range_ = disturbance_range_.to(device)
        self.box_bounds_ = box_bounds_.to(device)
        self.control_init = torch.tensor([0, 0, -self.Gz / self.input_multiplier]).to(device)
        self.disturbance_init = torch.tensor([0, 0, -self.Gz / self.input_multiplier]).to(device)
        self.eps_var_control = torch.tensor([self.sideways_multiplier, self.sideways_multiplier, self.input_multiplier]).to(device)
        self.eps_var_disturbance = torch.tensor([self.sideways_multiplier, self.sideways_multiplier, self.input_multiplier]).to(device) 

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        
        # Drone 1 (evader) dynamics
        dsdt[..., 1] = self.sideways_multiplier * control[..., 0]  # v1_x_dot
        dsdt[..., 3] = self.sideways_multiplier * control[..., 1]  # v1_y_dot
        dsdt[..., 5] = self.input_multiplier * control[..., 2] + self.Gz  # v1_z_dot
        
        # Drone 2 (pursuer) dynamics
        dsdt[..., 7] = self.sideways_multiplier * disturbance[..., 0]  # v2_x_dot
        dsdt[..., 9] = self.sideways_multiplier * disturbance[..., 1]  # v2_y_dot
        dsdt[..., 11] = self.input_multiplier * disturbance[..., 2] + self.Gz  # v2_z_dot
        
        # Position dynamics
        dsdt[..., 0] = state[..., 1]   # p1_x_dot = v1_x
        dsdt[..., 2] = state[..., 3]   # p1_y_dot = v1_y
        dsdt[..., 4] = state[..., 5]   # p1_z_dot = v1_z
        dsdt[..., 6] = state[..., 7]   # p2_x_dot = v2_x
        dsdt[..., 8] = state[..., 9]   # p2_y_dot = v2_y
        dsdt[..., 10] = state[..., 11] # p2_z_dot = v2_z
        
        return dsdt

    def boundary_fn(self, state):
        # # Evader position
        # p1 = state[..., [0, 2, 4]]  # [p1_x, p1_y, p1_z]
        # # Pursuer position
        # p2 = state[..., [6, 8, 10]]  # [p2_x, p2_y, p2_z]

        # # Horizontal (x, y) distance
        # dist_xy = torch.norm(p1[..., :2] - p2[..., :2], dim=-1)
        # # Vertical (z) distance
        # dist_z = torch.abs(p1[..., 2] - p2[..., 2])

        # radius = self.capture_radius
        # height = 1.5

        # # Constraint: outside cylinder is safe
        # horizontal_constraint = dist_xy - radius         # >0 is safe
        # # if(horizontal_constraint > 0):
        # #     capture_constraint = horizontal_constraint
        # # else:
        # #     vertical_constraint = dist_z - (height / 2)
        # #     capture_constraint = vertical_constraint
           
        # capture_constraint = horizontal_constraint

        p1 = torch.stack([state[..., 0], state[..., 2], state[..., 4]], dim=-1)
        # Drone 2 position
        p2 = torch.stack([state[..., 6], state[..., 8], state[..., 10]], dim=-1)
        #inter_drone_dist = torch.norm(p1 - p2, dim=-1) - self.capture_radius

        height = 1.5

        horizontal_dist = torch.sqrt((p1[..., 0] - p2[..., 0])**2 + (p1[..., 1] - p2[..., 1])**2) - self.capture_radius
        vertical_dist = torch.abs(p1[..., 2] - p2[..., 2]) - (height / 2)

        # Case 1: Outside in both directions
        outside_both = (horizontal_dist > 0) & (vertical_dist > 0)
        dist_outside = torch.sqrt(horizontal_dist**2 + vertical_dist**2)

        # Case 2: Outside horizontally, inside vertically
        outside_horiz = (horizontal_dist > 0) & (vertical_dist <= 0)

        # Case 3: Inside horizontally, outside vertically
        outside_vert = (horizontal_dist <= 0) & (vertical_dist > 0)

        # Case 4: Inside both (inside the cylinder)
        inside_both = (horizontal_dist <= 0) & (vertical_dist <= 0)
        dist_inside = torch.maximum(horizontal_dist, vertical_dist)  # most negative

        # Combine all cases
        inter_drone_dist = torch.where(
            outside_both, dist_outside,
            torch.where(
                outside_horiz, horizontal_dist,
                torch.where(
                    outside_vert, vertical_dist,
                    dist_inside
                )
            )
        )

        capture_constraint = inter_drone_dist

        evader_min = self.state_range_[[0, 2, 4], 0].to(state.device)
        evader_max = self.state_range_[[0, 2, 4], 1].to(state.device)


        # For each dimension, how far from the nearest boundary (positive inside, negative outside)
        px, py, pz = state[..., 0], state[..., 2], state[..., 4]

        box_bounds = self.box_bounds_.to(state.device)
        

        x_min, x_max = box_bounds[0, 0], box_bounds[0, 1]
        y_min, y_max = box_bounds[2, 0], box_bounds[2, 1]
        z_min, z_max = box_bounds[4, 0], box_bounds[4, 1]

        # Compute per-dimension signed distances to box faces
        dx_min = px - x_min
        dx_max = x_max - px
        dy_min = py - y_min
        dy_max = y_max - py
        dz_min = pz - z_min
        dz_max = z_max - pz

        # Inside: minimum distance to any face (negative inside, zero on surface)
        inside_dist = torch.min(torch.stack([dx_min, dx_max, dy_min, dy_max, dz_min, dz_max], dim=-1), dim=-1).values

        # For outside: compute the per-dimension "over" (how far outside the box in each dim)
        over_x = torch.clamp(px - x_max, min=0) + torch.clamp(x_min - px, min=0)
        over_y = torch.clamp(py - y_max, min=0) + torch.clamp(y_min - py, min=0)
        over_z = torch.clamp(pz - z_max, min=0) + torch.clamp(z_min - pz, min=0)
        # Norm of the "over" vector gives Euclidean distance outside
        outside_dist = torch.norm(torch.stack([over_x, over_y, over_z], dim=-1), dim=-1)

        # If all inside (all distances to faces > 0), use inside_dist; else use outside_dist
        is_inside = (dx_min > 0) & (dx_max > 0) & (dy_min > 0) & (dy_max > 0) & (dz_min > 0) & (dz_max > 0)
        inside_constraint = torch.where(is_inside, inside_dist, -outside_dist)
        
        if self.set_mode == 'avoid':
            # Safe if outside capture radius AND inside bounds
            return torch.minimum(capture_constraint, inside_constraint)
        else:
            
            return torch.minimum(-capture_constraint, inside_constraint)

    def hamiltonian(self, state, dvds):
        # Extract velocities and gradients
        v1 = state[..., [1, 3, 5]]   # Drone 1 velocities
        v2 = state[..., [7, 9, 11]]  # Drone 2 velocities
        dVdp1 = dvds[..., [0, 2, 4]] # Gradients w.r.t. drone 1 positions
        dVdp2 = dvds[..., [6, 8, 10]] # Gradients w.r.t. drone 2 positions
        dVdv1 = dvds[..., [1, 3, 5]]  # Gradients w.r.t. drone 1 velocities
        dVdv2 = dvds[..., [7, 9, 11]] # Gradients w.r.t. drone 2 velocities
        
        # Kinetic terms
        ham = (v1 * dVdp1).sum(-1) + (v2 * dVdp2).sum(-1)
        
        # Gravity terms
        ham += dVdv1[..., 2] * self.Gz + dVdv2[..., 2] * self.Gz
        
        # Control and disturbance terms
        if self.set_mode == 'avoid':
            # Evader (control) wants to avoid, Pursuer (disturbance) wants to capture
            # Control terms (evader maximizing distance)
            ham += (self.sideways_multiplier * (torch.abs(dVdv1[..., 0]) + torch.abs(dVdv1[..., 1])) + 
                    self.input_multiplier * torch.relu(dVdv1[..., 2])) * self.control_max

            # Disturbance terms (pursuer minimizing distance)
            ham -= (self.sideways_multiplier * (torch.abs(dVdv2[..., 0]) + torch.abs(dVdv2[..., 1])) + 
                    self.input_multiplier * torch.relu(dVdv2[..., 2])) * self.disturbance_max
        else:
            # Pursuer (control) wants to capture, Evader (disturbance) wants to avoid
            # Control terms (pursuer minimizing distance)
            ham -= (self.sideways_multiplier * (torch.abs(dVdv2[..., 0]) + torch.abs(dVdv2[..., 1])) + 
                    self.input_multiplier * torch.relu(dVdv2[..., 2])) * self.control_max
            # Disturbance terms (evader maximizing distance)
            ham += (self.sideways_multiplier * (torch.abs(dVdv1[..., 0]) + torch.abs(dVdv1[..., 1])) + 
                    self.input_multiplier * torch.relu(dVdv1[..., 2])) * self.disturbance_max
        
        return ham

    def optimal_control(self, state, dvds):
        dVdv1 = dvds[..., [1, 3, 5]]

        if self.set_mode == 'avoid':
            # Evader (control) wants to avoid capture - maximize distance
            
            u1_x = torch.sign(dVdv1[..., 0]) * self.control_max
            u1_y = torch.sign(dVdv1[..., 1]) * self.control_max
            u1_z = torch.clamp(torch.sign(dVdv1[..., 2]), min=0.0) * self.control_max
            
            return torch.stack([u1_x, u1_y, u1_z], dim=-1)
        elif self.set_mode == 'reach':
            # Pursuer (control) wants to capture - minimize distance
            
            u2_x = -torch.sign(dVdv1[..., 0]) * self.control_max
            u2_y = -torch.sign(dVdv1[..., 1]) * self.control_max
            u2_z = torch.clamp(-torch.sign(dVdv1[..., 2]), min=0.0) * self.control_max
            
            return torch.stack([u2_x, u2_y, u2_z], dim=-1)

    def optimal_disturbance(self, state, dvds):
        dVdv2 = dvds[..., [7, 9, 11]]

        if self.set_mode == 'avoid':
            # Pursuer (disturbance) wants to capture - minimize distance
            
            u2_x = -torch.sign(dVdv2[..., 0]) * self.disturbance_max
            u2_y = -torch.sign(dVdv2[..., 1]) * self.disturbance_max
            u2_z = torch.clamp(-torch.sign(dVdv2[..., 2]), min=0.0) * self.disturbance_max
            
            return torch.stack([u2_x, u2_y, u2_z], dim=-1)
        elif self.set_mode == 'reach':
            # Evader (disturbance) wants to avoid capture - maximize distance
            
            u1_x = torch.sign(dVdv2[..., 0]) * self.disturbance_max
            u1_y = torch.sign(dVdv2[..., 1]) * self.disturbance_max
            u1_z = torch.clamp(torch.sign(dVdv2[..., 2]), min=0.0) * self.disturbance_max
            
            return torch.stack([u1_x, u1_y, u1_z], dim=-1)

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        # Use boundary function for consistency
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def plot_config(self):
        return {
            'state_labels': ['p1_x', 'v1_x', 'p1_y', 'v1_y', 'p1_z', 'v1_z', 
                           'p2_x', 'v2_x', 'p2_y', 'v2_y', 'p2_z', 'v2_z'],
            'state_slices': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5, 0],
            'x_axis_idx': 0,  # p1_x
            'y_axis_idx': 2,  # p1_y
            'z_axis_idx': 4,  # p1_z
        }

    # def plot_config(self):
    #     def relative_state_fn(state):
    #         # state shape: [batch, 12]
    #         rel_x = state[..., 6] - state[..., 0]  # p2_x - p1_x
    #         rel_y = state[..., 8] - state[..., 2]  # p2_y - p1_y
    #         rel_z = state[..., 10] - state[..., 4] # p2_z - p1_z
    #         return torch.stack([rel_x, rel_y, rel_z], dim=-1)
        
    #     return {
    #         'state_slices': [0]*12,  # dummy, not used
    #         'state_labels': ['p2_x - p1_x', 'p2_y - p1_y', 'p2_z - p1_z'],
    #         'x_axis_idx': 0,  # rel_x
    #         'y_axis_idx': 1,  # rel_y
    #         'z_axis_idx': 2,  # rel_z
    #         'fixed_evader_state': [0.0, 0.0, 1.5],  # Optionally fix evader at origin
    #         'state_transform': relative_state_fn
    #     }

    # Add all the other required methods (control_range, disturbance_range, etc.)
    def control_range(self, state):
        return self.control_range_.cpu().tolist()
    
    def disturbance_range(self, state):
        return self.disturbance_range_.cpu().tolist()
    
    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[..., 0], self.control_range_[..., 1])
    
    def bound_disturbance(self, disturbance):
        return torch.clamp(disturbance, self.disturbance_range_[..., 0], self.disturbance_range_[..., 1])
    
    def clamp_control(self, state, control):
        return self.bound_control(control)
    
    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)
    
    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])
    
    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()
    
    def equivalent_wrapped_state(self, state):
        return torch.clone(state)
    
    def periodic_transform_fn(self, input):
        return input.to(device)

class CooperativeDrones12D(Dynamics):
    def __init__(self, collisionR: float, disturbance_max: float, thrust_max: float, set_mode: str):
        self.state_dim = 12
        self.control_dim = 6  # 3 for each drone
        self.disturbance_dim = 6  # 3 for each drone (downwash)
        self.collisionR = collisionR
        self.input_multiplier = thrust_max
        self.sideways_multiplier = 2.0
        self.control_max = 1.0
        self.disturbance_max = disturbance_max

        self.Gz = -9.81
        self.max_v = 2.0

        # State: [p1x, v1x, p1y, v1y, p1z, v1z, p2x, v2x, p2y, v2y, p2z, v2z]
        state_range_ = torch.tensor([
            # Drone 1 (evader)
            [-2, 2], [-self.max_v, self.max_v],  # p1_x, v1_x
            [-2, 2], [-self.max_v, self.max_v],  # p1_y, v1_y
            [-0.5, 3.5], [-self.max_v, self.max_v],  # p1_z, v1_z
            # Drone 2 (pursuer)
            [-2, 2], [-self.max_v, self.max_v],  # p2_x, v2_x
            [-2, 2], [-self.max_v, self.max_v],  # p2_y, v2_y
            [-0.5, 3.5], [-self.max_v, self.max_v],  # p2_z, v2_z
        ])
        control_range_ = torch.tensor([
            [-self.control_max, self.control_max], [-self.control_max, self.control_max], [0.0, self.control_max],  # drone 1
            [-self.control_max, self.control_max], [-self.control_max, self.control_max], [0.0, self.control_max],  # drone 2
        ])
        # control_range_ = torch.tensor([
        #     [-self.control_max, self.control_max], [-self.control_max, self.control_max], [0.0, self.control_max],  # drone 1
        #     [0,0], [0, 0], [0, 0],  # drone 2
        # ])
        disturbance_range_ = torch.tensor([
            [-self.disturbance_max, self.disturbance_max], [-self.disturbance_max, self.disturbance_max], [-self.disturbance_max, self.disturbance_max],  # drone 1
            [-self.disturbance_max, self.disturbance_max], [-self.disturbance_max, self.disturbance_max], [-self.disturbance_max, self.disturbance_max],  # drone 2
        ])
        # disturbance_range_ = torch.tensor([
        #     [-self.disturbance_max, self.disturbance_max], [-self.disturbance_max, self.disturbance_max], [-self.disturbance_max, self.disturbance_max],  # drone 1
        #     [0, 0], [0, 0], [0, 0],  # drone 2
        # ])

        state_mean_ = (state_range_[:, 0] + state_range_[:, 1]) / 2.0
        state_var_ = (state_range_[:, 1] - state_range_[:, 0]) / 2.0

        super().__init__(
            name="CooperativeDrones12D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=12, input_dim=13, control_dim=6, disturbance_dim=self.disturbance_dim,
            state_mean=state_mean_.tolist(),
            state_var=state_var_.tolist(),
            value_mean=0.0,
            value_var=1.5,
            value_normto=0.02,
            deepReach_model='exact'
        )
        self.state_range_ = state_range_.to(device)
        self.control_range_ = control_range_.to(device)
        self.disturbance_range_ = disturbance_range_.to(device)
        self.control_init = torch.zeros(6).to(device)
        self.disturbance_init = torch.zeros(6).to(device)        
        self.cylinder_radius = 0.5

        self.eps_var_control = torch.tensor([self.sideways_multiplier, self.sideways_multiplier, self.input_multiplier,
                                             self.sideways_multiplier, self.sideways_multiplier, self.input_multiplier]).to(device)

        self.eps_var_disturbance = torch.ones(6).to(device) * self.disturbance_max  # sampling variance

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        # Drone 1
        dsdt[..., 0] = state[..., 1]  # p1x_dot = v1x
        dsdt[..., 1] = self.sideways_multiplier * control[..., 0] + disturbance[..., 0]  # v1x_dot = a1x + d1x
        dsdt[..., 2] = state[..., 3]  # p1y_dot = v1y
        dsdt[..., 3] = self.sideways_multiplier * control[..., 1] + disturbance[..., 1]  # v1y_dot = a1y + d1y
        dsdt[..., 4] = state[..., 5]  # p1z_dot = v1z
        dsdt[..., 5] = self.input_multiplier * control[..., 2] + disturbance[..., 2] + self.Gz  # v1z_dot = a1z + d1z + g
        # Drone 2
        dsdt[..., 6] = state[..., 7]  # p2x_dot = v2x
        dsdt[..., 7] = self.sideways_multiplier * control[..., 3] + disturbance[..., 3]  # v2x_dot = a2x + d2x
        dsdt[..., 8] = state[..., 9]  # p2y_dot = v2y
        dsdt[..., 9] = self.sideways_multiplier * control[..., 4] + disturbance[..., 4]  # v2y_dot = a2y + d2y
        dsdt[..., 10] = state[..., 11]  # p2z_dot = v2z
        dsdt[..., 11] = self.input_multiplier * control[..., 5] + disturbance[..., 5] + self.Gz  # v2z_dot = a2z + d2z + g
        return dsdt

    def hamiltonian(self, state, dvds):
        # Extract velocities and gradients for both drones
        v1 = state[..., [1, 3, 5]]   # Drone 1 velocities [v1x, v1y, v1z]
        v2 = state[..., [7, 9, 11]]  # Drone 2 velocities [v2x, v2y, v2z]
        dVdp1 = dvds[..., [0, 2, 4]] # Gradients w.r.t. drone 1 positions [dV/dp1x, dV/dp1y, dV/dp1z]
        dVdp2 = dvds[..., [6, 8, 10]] # Gradients w.r.t. drone 2 positions [dV/dp2x, dV/dp2y, dV/dp2z]
        dVdv1 = dvds[..., [1, 3, 5]]  # Gradients w.r.t. drone 1 velocities [dV/dv1x, dV/dv1y, dV/dv1z]
        dVdv2 = dvds[..., [7, 9, 11]] # Gradients w.r.t. drone 2 velocities [dV/dv2x, dV/dv2y, dV/dv2z]
        
        # Kinetic terms: v · ∇V (position dynamics)
        ham = (v1 * dVdp1).sum(-1) + (v2 * dVdp2).sum(-1)
        
        # Gravity terms: g · ∇V (velocity dynamics)
        ham += dVdv1[..., 2] * self.Gz + dVdv2[..., 2] * self.Gz
        
        # Control and disturbance multipliers for each dimension
        # [sideways_x, sideways_y, input_z] for each drone
        control_multipliers = torch.tensor([
            self.sideways_multiplier, self.sideways_multiplier, self.input_multiplier,  # drone 1
            self.sideways_multiplier, self.sideways_multiplier, self.input_multiplier   # drone 2
        ]).to(state.device)
        
        disturbance_multipliers = torch.ones(6).to(state.device)  # All dimensions have same disturbance multiplier
        
        if self.set_mode == 'avoid':
            # Both drones cooperate to avoid collision
            # Control maximizes Hamiltonian, disturbance minimizes it
            # Drone 1 control terms
            ham += (control_multipliers[:3] * torch.abs(dVdv1) * self.control_max).sum(-1)
            # Drone 1 disturbance terms  
            ham -= (disturbance_multipliers[:3] * torch.abs(dVdv1) * self.disturbance_max).sum(-1)
            # # Drone 2 control terms
            ham += (control_multipliers[3:] * torch.abs(dVdv2) * self.control_max).sum(-1)
            # Drone 2 disturbance terms
            ham -= (disturbance_multipliers[3:] * torch.abs(dVdv2) * self.disturbance_max).sum(-1)
            
        elif self.set_mode == 'reach' or self.set_mode == 'reach_avoid':
            # Both drones cooperate to reach target separation
            # Control maximizes Hamiltonian, disturbance minimizes it
            # Drone 1 control terms
            ham += (control_multipliers[:3] * torch.abs(dVdv1) * self.control_max).sum(-1)
            # Drone 1 disturbance terms
            ham -= (disturbance_multipliers[:3] * torch.abs(dVdv1) * self.disturbance_max).sum(-1)
            # Drone 2 control terms
            ham += (control_multipliers[3:] * torch.abs(dVdv2) * self.control_max).sum(-1)
            # Drone 2 disturbance terms
            ham -= (disturbance_multipliers[3:] * torch.abs(dVdv2) * self.disturbance_max).sum(-1)
        
        return ham

    def control_range(self, state):
        return self.control_range_.tolist()

    def disturbance_range(self, state):
        return self.disturbance_range_.tolist()

    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[:, 0], self.control_range_[:, 1])

    def bound_disturbance(self, disturbance):
        return torch.clamp(disturbance, self.disturbance_range_[:, 0], self.disturbance_range_[:, 1])

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def clamp_control(self, state, control):
        return self.bound_control(control)
    
    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)
    
    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def periodic_transform_fn(self, input):
        return input.to(device)
    
    def equivalent_wrapped_state(self, state):
        # No periodicity, just return a clone
        return torch.clone(state)

    def sample_target_state(self, num_samples):
        # Uniformly sample within state range
        state_range = torch.tensor(self.state_test_range())
        samples = state_range[:, 0] + torch.rand(num_samples, self.state_dim) * (state_range[:, 1] - state_range[:, 0])
        return samples
    
    def dist_to_cylinder(self, state, a, b, height=None):
        """
        Computes the signed distance from both drones to a fixed vertical cylinder.
        Cylinder is centered at (a, b) in the x/y plane, with radius self.collisionR and height.
        Returns the minimum signed distance (negative if either drone is inside).
        """
        # Drone 1 position
        p1_x = state[..., 0]
        p1_y = state[..., 2]
        p1_z = state[..., 4]
        # Drone 2 position
        p2_x = state[..., 6]
        p2_y = state[..., 8]
        p2_z = state[..., 10]

        # Horizontal distances
        dist1_xy = torch.sqrt((p1_x - a) ** 2 + (p1_y - b) ** 2)
        dist2_xy = torch.sqrt((p2_x - a) ** 2 + (p2_y - b) ** 2)

        # Vertical distances (if you want to enforce a height)
        if height is not None:
            dist1_z = torch.abs(p1_z - 1.5)  # assuming cylinder is centered at z=1.5
            dist2_z = torch.abs(p2_z - 1.5)
            vertical_constraint1 = (height / 2) - dist1_z
            vertical_constraint2 = (height / 2) - dist2_z
            # Both must be outside in both xy and z
            dist1 = torch.minimum(dist1_xy - self.cylinder_radius, vertical_constraint1)
            dist2 = torch.minimum(dist2_xy - self.cylinder_radius, vertical_constraint2)
        else:
            dist1 = dist1_xy - self.cylinder_radius
            dist2 = dist2_xy - self.cylinder_radius

        # Return the minimum (if either drone is inside, it's a violation)

        return torch.minimum(dist1, dist2)
    

    def boundary_fn(self, state):
       # Cylinder avoidance (central obstacle)
        cylinder_dist = self.dist_to_cylinder(state, a=0.0, b=0.0) 
        # Inter-drone collision avoidance
        # Drone 1 position
        p1 = torch.stack([state[..., 0], state[..., 2], state[..., 4]], dim=-1)
        # Drone 2 position
        p2 = torch.stack([state[..., 6], state[..., 8], state[..., 10]], dim=-1)
        inter_drone_dist = torch.norm(p1 - p2, dim=-1) - self.collisionR

        height = 1.5

        horizontal_dist = torch.sqrt((p1[..., 0] - p2[..., 0])**2 + (p1[..., 1] - p2[..., 1])**2) - self.collisionR
        vertical_dist = torch.abs(p1[..., 2] - p2[..., 2]) - (height / 2)

        # Case 1: Outside in both directions
        outside_both = (horizontal_dist > 0) & (vertical_dist > 0)
        dist_outside = torch.sqrt(horizontal_dist**2 + vertical_dist**2)

        # Case 2: Outside horizontally, inside vertically
        outside_horiz = (horizontal_dist > 0) & (vertical_dist <= 0)

        # Case 3: Inside horizontally, outside vertically
        outside_vert = (horizontal_dist <= 0) & (vertical_dist > 0)

        # Case 4: Inside both (inside the cylinder)
        inside_both = (horizontal_dist <= 0) & (vertical_dist <= 0)
        dist_inside = torch.maximum(horizontal_dist, vertical_dist)  # most negative

        # Combine all cases
        inter_drone_dist = torch.where(
            outside_both, dist_outside,
            torch.where(
                outside_horiz, horizontal_dist,
                torch.where(
                    outside_vert, vertical_dist,
                    dist_inside
                )
            )
        )

        if self.set_mode == 'avoid':
            # Positive outside collision, negative inside
            return torch.minimum(cylinder_dist, inter_drone_dist)
        elif self.set_mode == 'reach':
            # Positive if at or beyond target separation, negative if closer
            target_sep = 2.0  # or any value you want
            return inter_drone_dist - target_sep
        elif self.set_mode == 'reach_avoid':
            # Example: reach a target separation while avoiding collision
            target_sep = 2.0
            avoid = dist - self.collisionR
            reach = dist - target_sep
            # max(reach, -avoid): must reach target sep and not be in collision
            return torch.maximum(reach, -avoid)
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")

    def cost_fn(self, state_traj):
        # state_traj: (..., T, state_dim)
        # Call boundary_fn for each time step
        boundary_vals = self.boundary_fn(state_traj)
        return torch.min(boundary_vals, dim=-1).values

    def optimal_control(self, state, dvds):
            # Extract velocity gradients for both drones
        dVdv1 = dvds[..., [1, 3, 5]]  # velocity gradients for drone 1
        dVdv2 = dvds[..., [7, 9, 11]]  # velocity gradients for drone 2
        
        # Optimal control: apply maximum control in direction of gradient
        u1 = self.control_max * torch.sign(dVdv1)  # drone 1 control
        u2 = self.control_max * torch.sign(dVdv2)  # drone 2 control

        
        # Combine controls for both drones
        u = torch.cat([u1, u2], dim=-1)
        return u

    def optimal_disturbance(self, state, dvds):
         # Extract velocity gradients for both drones
        dVdv1 = dvds[..., [1, 3, 5]]  # velocity gradients for drone 1
        dVdv2 = dvds[..., [7, 9, 11]]  # velocity gradients for drone 2
        
        # Optimal disturbance: worst-case scenario (opposite to gradient)
        d1 = -self.disturbance_max * torch.sign(dVdv1)  # drone 1 disturbance
        d2 = -self.disturbance_max * torch.sign(dVdv2)  # drone 2 disturbance
        
        # Combine disturbances for both drones
        d = torch.cat([d1, d2], dim=-1)
        return d
    
    #plot with cylinder
    def plot_config(self):
        return {
            'state_slices': [0.0, -1.43, 0.0, -1.2, 1.5, -0.1, 1.0, 0, 1.0, 0, 1.5, 0],
            'state_labels': ['p1x', 'v1x', 'p1y', 'v1y', 'p1z', 'v1z', 'p2x', 'v2x', 'p2y', 'v2y', 'p2z', 'v2z'],
            'x_axis_idx': 0,  # p_x
            'y_axis_idx': 2,  # p_y
            'z_axis_idx': 4,  # p_z
        }

class Drone10DWithDist(Dynamics):
    def __init__(self, thrust_max: float, max_angle: float, disturbance_max: float, max_torque: float, set_mode: str):
    
        self.state_dim = 10  # Updated to 10D
        self.control_dim = 3
        self.disturbance_dim = 3
        #self.disturbance_dim = 0

        self.control_max = 1.0  # u_max (normalized control bound)
        self.max_torque = max_torque
        self.disturbance_max = disturbance_max
        self.Gz = -9.81
        self.max_v = 2.0
        self.max_omega = 1.0  # Maximum angular velocity
        self.max_theta = max_angle  # Maximum angle (radians)
        self.collisionR = 0.50

        self.d0 = 10
        self.d1 = 8
        self.n0 = 10
        self.k_T = thrust_max
        self.mass = 1.0

        # State: [x, v_x, θ_x, ω_x, y, v_y, θ_y, ω_y, z, v_z]
        
        state_range_ = torch.tensor([
            [-2, 2], [-self.max_v, self.max_v], [-self.max_theta, self.max_theta], [-self.max_omega, self.max_omega],  # x, v_x, θ_x, ω_x
            [-2, 2], [-self.max_v, self.max_v], [-self.max_theta, self.max_theta], [-self.max_omega, self.max_omega],  # y, v_y, θ_y, ω_y
            [-0.5, 3.5], [-self.max_v, self.max_v],  # z, v_z
        ])
        control_range_ = torch.tensor([
            [-self.control_max, self.control_max],  # S_x
            [-self.control_max, self.control_max],  # S_y
            [0.0, self.control_max],  # T_z
        ])
        disturbance_range_ = torch.tensor([
            [-self.disturbance_max, self.disturbance_max],  # d_x
            [-self.disturbance_max, self.disturbance_max],  # d_y
            [-self.disturbance_max, self.disturbance_max],  # d_z
        ])

        box_bounds_ = torch.tensor([
            [-1.5, 1.5], [-self.max_v, self.max_v],
            [-1.5, 1.5], [-self.max_v, self.max_v],
            [0.0, 3.0],  [-self.max_v, self.max_v],
        ])

        state_mean_ = (state_range_[:, 0] + state_range_[:, 1]) / 2.0
        state_var_ = (state_range_[:, 1] - state_range_[:, 0]) / 2.0

        super().__init__(
            name="Drone10DWithDist", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=10, input_dim=13, control_dim=self.control_dim, disturbance_dim=self.disturbance_dim,
            state_mean=state_mean_.tolist(),
            state_var=state_var_.tolist(),
            value_mean=0.2,
            value_var=0.5,
            value_normto=0.02,
            deepReach_model='exact'
        )
        self.state_range_ = state_range_.to(device)
        self.control_range_ = control_range_.to(device)
        self.box_bounds_ = box_bounds_.to(device)
        self.disturbance_range_ = disturbance_range_.to(device)
        self.control_init = torch.tensor([0, 0, -self.Gz / self.k_T]).to(device)
        self.disturbance_init = torch.zeros(3).to(device)
        self.eps_var_control = torch.tensor([self.max_torque, self.max_torque, self.k_T]).to(device)
        self.eps_var_disturbance = torch.ones(3).to(device) * self.disturbance_max

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        
        # State: [x, v_x, θ_x, ω_x, y, v_y, θ_y, ω_y, z, v_z]
        # Control: [S_x, S_y, T_z]
        
        # Position derivatives
        dsdt[..., 0] = state[..., 1]  # x_dot = v_x
        dsdt[..., 4] = state[..., 5]  # y_dot = v_y
        dsdt[..., 8] = state[..., 9]  # z_dot = v_z
        
        # Velocity derivatives (with control and disturbance)
        dsdt[..., 1] = -self.Gz * torch.tan(state[..., 2]) + disturbance[..., 0]  # v̇_x = g * tan(θ_x) + d_x
        dsdt[..., 5] = -self.Gz * torch.tan(state[..., 6]) + disturbance[..., 1]  # v̇_y = g * tan(θ_y) + d_y
        dsdt[..., 9] = self.k_T / self.mass * control[..., 2] + disturbance[..., 2] + self.Gz  # v_z_dot = T_z + d_z - g
        
        # Angle derivatives
        dsdt[..., 2] = state[..., 3] - self.d1 * state[..., 2]  # θ_x_dot = ω_x - d1 * θ_x
        dsdt[..., 6] = state[..., 7] - self.d1 * state[..., 6]  # θ_y_dot = ω_y - d1 * θ_y
        
        dsdt[..., 3] = -self.d0 * state[..., 2] + self.n0 * self.max_torque * control[..., 0]  # ω̇_x
        dsdt[..., 7] = -self.d0 * state[..., 6] + self.n0 * self.max_torque * control[..., 1]  # ω̇_y

        
        return dsdt
    
    def hamiltonian(self, state, dvds):
        # Extract velocities and gradients
        v = state[..., [1, 5, 9]]  # [v_x, v_y, v_z]
        omega = state[..., [3, 7]]  # [ω_x, ω_y]
        theta = state[..., [2, 6]]  # [θ_x, θ_y]

        dVdp = dvds[..., [0, 4, 8]]  # [dV/dx, dV/dy, dV/dz]
        dVdv = dvds[..., [1, 5, 9]]  # [dV/dv_x, dV/dv_y, dV/dv_z]
        dVdtheta = dvds[..., [2, 6]]  # [dV/dθ_x, dV/dθ_y]
        dVdomega = dvds[..., [3, 7]]  # [dV/dω_x, dV/dω_y]
        
        # Position derivatives: v * dVdp
        ham = (v * dVdp).sum(-1)
        
        # Velocity derivatives: gravity terms and angle-dependent terms
        ham += dVdv[..., 0] * (-self.Gz * torch.tan(theta[..., 0]))  # v_x term
        ham += dVdv[..., 1] * (-self.Gz * torch.tan(theta[..., 1]))  # v_y term
        ham += dVdv[..., 2] * self.Gz  # v_z gravity term
        
        # Angle derivatives: omega * dVdtheta - damping terms
        ham += (omega * dVdtheta).sum(-1)
        ham += dVdtheta[..., 0] * (-self.d1 * theta[..., 0])  # θ_x damping
        ham += dVdtheta[..., 1] * (-self.d1 * theta[..., 1])  # θ_y damping
        
        # Angular velocity derivatives: damping terms
        ham += dVdomega[..., 0] * (-self.d0 * theta[..., 0])  # ω_x damping
        ham += dVdomega[..., 1] * (-self.d0 * theta[..., 1])  # ω_y damping
        
        # Control terms with correct multipliers
        if self.set_mode == 'avoid':

            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega[..., 0])  # S_x
            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega[..., 1])  # S_y
           
            ham += self.k_T / self.mass * self.control_max * torch.relu(dVdv[..., 2])  # T_z (only positive)

            ham -= self.disturbance_max * torch.abs(dVdv).sum(-1)

        elif self.set_mode == 'reach':
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega[..., 0])  # S_x
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega[..., 1])  # S_y
           
            ham -= self.k_T / self.mass * self.control_max * torch.relu(dVdv[..., 2])  # T_z (vertical)

            ham += self.disturbance_max * torch.abs(dVdv).sum(-1)
        
        return ham

    def dist_to_cylinder(self, state, a, b):
        """
        Computes the signed distance from the drone's position to a vertical cylinder
        centered at (a, b) in the x/y plane, with radius self.collisionR.
        Returns positive outside, zero on the surface, negative inside.
        """
        px = state[..., 0]  # x
        py = state[..., 4]  # y
        # Distance in x/y plane from drone to cylinder center
        dist_xy = torch.sqrt((px - a) ** 2 + (py - b) ** 2)
        # Signed distance to cylinder surface
        return dist_xy - self.collisionR
    

    # def boundary_fn(self, state):
    #     if self.set_mode=='avoid':
    #         return self.dist_to_cylinder(state,0.0,0.0)
    #     elif self.set_mode=='reach':
    #         return -self.dist_to_cylinder(state,5.0,5.0)
    #     else:
    #         raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")
    

    def boundary_fn(self, state):
        # state: (..., 10) - [x, v_x, θ_x, ω_x, y, v_y, θ_y, ω_y, z, v_z]
        # Box: x, y in [-1.5, 1.5], z in [0, 3]
        px, py, pz = state[..., 0], state[..., 4], state[..., 8]

        box_bounds = self.box_bounds_.to(state.device)
        

        x_min, x_max = box_bounds[0, 0], box_bounds[0, 1]
        y_min, y_max = box_bounds[2, 0], box_bounds[2, 1]
        z_min, z_max = box_bounds[4, 0], box_bounds[4, 1]

        # Compute per-dimension signed distances to box faces
        dx_min = px - x_min
        dx_max = x_max - px
        dy_min = py - y_min
        dy_max = y_max - py
        dz_min = pz - z_min
        dz_max = z_max - pz

        # Inside: minimum distance to any face (negative inside, zero on surface)
        inside_dist = torch.min(torch.stack([dx_min, dx_max, dy_min, dy_max, dz_min, dz_max], dim=-1), dim=-1).values

        # For outside: compute the per-dimension "over" (how far outside the box in each dim)
        over_x = torch.clamp(px - x_max, min=0) + torch.clamp(x_min - px, min=0)
        over_y = torch.clamp(py - y_max, min=0) + torch.clamp(y_min - py, min=0)
        over_z = torch.clamp(pz - z_max, min=0) + torch.clamp(z_min - pz, min=0)
        # Norm of the "over" vector gives Euclidean distance outside
        outside_dist = torch.norm(torch.stack([over_x, over_y, over_z], dim=-1), dim=-1)

        # If all inside (all distances to faces > 0), use inside_dist; else use outside_dist
        is_inside = (dx_min > 0) & (dx_max > 0) & (dy_min > 0) & (dy_max > 0) & (dz_min > 0) & (dz_max > 0)
        return torch.where(is_inside, inside_dist, -outside_dist)

        # center = torch.tensor([0.0, 0.0, 1.5], device=state.device)
        # pos = torch.stack([px, py, pz], dim=-1)
        # dist = torch.norm(pos - center, dim=-1)
        # radius = 1.5

        # pos_sdf = radius - dist

        # return pos_sdf
       

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # Wrap θ_x (index 2)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        # Wrap θ_y (index 6)
        wrapped_state[..., 6] = (wrapped_state[..., 6] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    
    def periodic_transform_fn(self, input):
        # Transform periodic angles θ_x and θ_y to sin/cos components
        # Input: [..., 11] - [t, x, v_x, θ_x, ω_x, y, v_y, θ_y, ω_y, z, v_z]
        # Output: [..., 13] - [t, x, v_x, sin(θ_x), cos(θ_x), ω_x, y, v_y, sin(θ_y), cos(θ_y), ω_y, z, v_z]
        
    
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] + 2  # Add 2 dimensions for sin/cos transforms
        transformed_input = torch.zeros(output_shape, device=input.device)
        
        # Copy non-periodic variables
        transformed_input[..., 0] = input[..., 0]  # t (time)
        transformed_input[..., 1] = input[..., 1]  # x
        transformed_input[..., 2] = input[..., 2]  # v_x
        transformed_input[..., 5] = input[..., 4]  # ω_x
        transformed_input[..., 6] = input[..., 5]  # y
        transformed_input[..., 7] = input[..., 6]  # v_y
        transformed_input[..., 10] = input[..., 8]  # ω_y
        transformed_input[..., 11] = input[..., 9]  # z
        transformed_input[..., 12] = input[..., 10]  # v_z
        
        theta_x = input[..., 3]
        transformed_input[..., 3] = torch.sin(theta_x * self.state_var[2])  # sin(θ_x)
        transformed_input[..., 4] = torch.cos(theta_x * self.state_var[2])  # cos(θ_x)
        
        # Transform θ_y to sin/cos 
        theta_y = input[..., 7]
        transformed_input[..., 8] = torch.sin(theta_y * self.state_var[6])  # sin(θ_y)
        transformed_input[..., 9] = torch.cos(theta_y * self.state_var[6])  # cos(θ_y)
        
        return transformed_input
    
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    
    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()
    
    def control_range(self, state):
        return self.control_range_.tolist()

    def disturbance_range(self, state):
        return self.disturbance_range_.tolist()

    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[:, 0], self.control_range_[:, 1])

    def bound_disturbance(self, disturbance):
        return torch.clamp(disturbance, self.disturbance_range_[:, 0], self.disturbance_range_[:, 1])
    
    def clamp_control(self, state, control):
        return self.bound_control(control)
    
    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def cost_fn(self, state_traj):
    # Use boundary function for consistency
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    
    def optimal_control(self, state, dvds):
        # Extract gradients for different control inputs
        dVdomega = dvds[..., [3, 7]]  # [dV/dω_x, dV/dω_y] for torque controls S_x, S_y
        dVdv = dvds[..., [1, 5, 9]]  # [dV/dv_x, dV/dv_y, dV/dv_z] for thrust control T_z
        
        control = torch.zeros_like(dVdv)


        if self.set_mode == 'avoid':
            # Control: [S_x, S_y, T_z]
            # Torque controls (S_x, S_y) based on angular velocity gradients
            control[..., 0] = self.control_max * torch.sign(dVdomega[..., 0])  # S_x
            control[..., 1] = self.control_max * torch.sign(dVdomega[..., 1])  # S_y
            # Thrust control (T_z) based on velocity gradient
            control[..., 2] = self.control_max * torch.relu(torch.sign(dVdv[..., 2]))  # T_z
        elif self.set_mode == 'reach':
            # Control: [S_x, S_y, T_z]
            # Torque controls (S_x, S_y) based on angular velocity gradients
            control[..., 0] = -self.control_max * torch.sign(dVdomega[..., 0])  # S_x
            control[..., 1] = -self.control_max * torch.sign(dVdomega[..., 1])  # S_y
            # Thrust control (T_z) based on velocity gradient
            control[..., 2] = -self.control_max * torch.relu(torch.sign(dVdv[..., 2]))  # T_z
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")
        return control

    def optimal_disturbance(self, state, dvds):
        dVdv = dvds[..., [1, 5, 9]]  # [dV/dv_x, dV/dv_y, dV/dv_z]
        disturbance = torch.zeros_like(dVdv)

        if self.set_mode == 'avoid':
            # Disturbance opposes the controller (worst case)
            disturbance[..., 0] = -self.disturbance_max * torch.sign(dVdv[..., 0])  # d_x
            disturbance[..., 1] = -self.disturbance_max * torch.sign(dVdv[..., 1])  # d_y
            disturbance[..., 2] = -self.disturbance_max * torch.sign(dVdv[..., 2])  # d_z
        elif self.set_mode == 'reach':
            # Disturbance opposes the controller (worst case)
            disturbance[..., 0] = self.disturbance_max * torch.sign(dVdv[..., 0])  # d_x
            disturbance[..., 1] = self.disturbance_max * torch.sign(dVdv[..., 1])  # d_y
            disturbance[..., 2] = self.disturbance_max * torch.sign(dVdv[..., 2])  # d_z
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")

        return disturbance

    #For cylinder plotting
    # def plot_config(self):
    #     return {
    #         'state_slices': [0.0, -1.43, 0.0, 0.0, 0.0, -1.2, 0.0, 0.0, 1.5, -0.1],  # [x, v_x, θ_x, ω_x, y, v_y, θ_y, ω_y, z, v_z]
    #         'state_labels': ['x', 'v_x', 'θ_x', 'ω_x', 'y', 'v_y', 'θ_y', 'ω_y', 'z', 'v_z'],
    #         'x_axis_idx': 0,  # x
    #         'y_axis_idx': 4,  # y
    #         'z_axis_idx': 8,  # z
    #     }
    
    #For vertical box plotting
    def plot_config(self):
        return {
            'state_slices': [0.0, -0.8, 0.0, 0.0, 0.0, -1.2, 0.0, 0.0, 1.5, 0.0],  # [x, v_x, θ_x, ω_x, y, v_y, θ_y, ω_y, z, v_z]
            'state_labels': ['x', 'v_x', 'θ_x', 'ω_x', 'y', 'v_y', 'θ_y', 'ω_y', 'z', 'v_z'],
            'x_axis_idx': 9,  # x
            'y_axis_idx': 8,  # y
            'z_axis_idx': 0,  # z
        }
    
    # # #For horizontal box plotting
    # def plot_config(self):
    #     return {
    #         'state_slices': [0.0, -0.8, 0.0, 0.0, 0.0, -1.2, 0.0, 0.0, 1.5, 0.0],  # [x, v_x, θ_x, ω_x, y, v_y, θ_y, ω_y, z, v_z]
    #         'state_labels': ['x', 'v_x', 'θ_x', 'ω_x', 'y', 'v_y', 'θ_y', 'ω_y', 'z', 'v_z'],
    #         'x_axis_idx': 0,  # x
    #         'y_axis_idx': 4,  # y
    #         'z_axis_idx': 8,  # z
    #     }

class DronePursuitEvasion20D(Dynamics):
    """
    20D Drone pursuit-evasion system: 1 evader, 1 pursuer.
    State: [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z,  # Drone 1 (evader)
            x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z] # Drone 2 (pursuer)
    Control: [S1_x, S1_y, T1_z] (evader control)
    Disturbance: [S2_x, S2_y, T2_z] (pursuer control)
    """
    disturbance_dim = 3

    def __init__(self, thrust_max: float, max_angle: float, max_torque: float, capture_radius: float, set_mode: str):
        self.state_dim = 20  # 10D for each drone
        self.control_dim = 3  # 3 controls for evader
        self.disturbance_dim = 3  # 3 controls for pursuer

        self.control_max = 1.0  # u_max (normalized control bound)
        self.max_torque = max_torque
        self.Gz = -9.81
        self.max_v = 2.0
        self.max_omega = 1.0  # Maximum angular velocity
        self.max_theta = max_angle  # Maximum angle (radians)
        self.capture_radius = capture_radius

        # Drone dynamics parameters
        self.d0 = 10
        self.d1 = 8
        self.n0 = 10
        self.k_T = thrust_max
        self.mass = 1.0

        # State: [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z,  # Drone 1 (evader)
        #         x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z] # Drone 2 (pursuer)
        
        # State ranges for both drones (same as Drone10DWithDist)
        drone_state_range = torch.tensor([
            [-2.5, 2.5], [-self.max_v, self.max_v], [-self.max_theta, self.max_theta], [-self.max_omega, self.max_omega],  # x, v_x, θ_x, ω_x
            [-2.5, 2.5], [-self.max_v, self.max_v], [-self.max_theta, self.max_theta], [-self.max_omega, self.max_omega],  # y, v_y, θ_y, ω_y
            [-0.5, 3.5], [-self.max_v, self.max_v],  # z, v_z
        ])
        
        # Combine state ranges for both drones
        state_range_ = torch.cat([drone_state_range, drone_state_range], dim=0)
        
        control_range_ = torch.tensor([
            [-self.control_max, self.control_max],  # S1_x
            [-self.control_max, self.control_max],  # S1_y
            [0.0, self.control_max],  # T1_z
        ])
        disturbance_range_ = torch.tensor([
            [-self.control_max, self.control_max],  # S2_x
            [-self.control_max, self.control_max],  # S2_y
            [0.0, self.control_max],  # T2_z
        ])

        box_bounds_ = torch.tensor([
        [-2.0, 2.0], [-self.max_v, self.max_v],
        [-2.0, 2.0], [-self.max_v, self.max_v],
        [0.0, 3.0],  [-self.max_v, self.max_v],
        ])
        
        

        state_mean_ = (state_range_[:, 0] + state_range_[:, 1]) / 2.0
        state_var_ = (state_range_[:, 1] - state_range_[:, 0]) / 2.0

        super().__init__(
            name="DronePursuitEvasion20D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=20, input_dim=25, control_dim=self.control_dim, disturbance_dim=self.disturbance_dim,
            state_mean=state_mean_.tolist(),
            state_var=state_var_.tolist(),
            value_mean=0.2,
            value_var=0.5,
            value_normto=0.02,
            deepReach_model='exact'
        )
        
        self.box_bounds_ = box_bounds_.to(device)

        self.state_range_ = state_range_.to(device)
        self.control_range_ = control_range_.to(device)
        self.disturbance_range_ = disturbance_range_.to(device)

        self.control_init = torch.tensor([0, 0, -self.Gz / self.k_T]).to(device)
        self.disturbance_init = torch.tensor([0, 0, -self.Gz / self.k_T]).to(device)

        self.eps_var_control = torch.tensor([self.max_torque, self.max_torque, self.k_T]).to(device)
        self.eps_var_disturbance = torch.tensor([self.max_torque, self.max_torque, self.k_T]).to(device)

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        
        # State: [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z,  # Drone 1 (evader)
        #         x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z] # Drone 2 (pursuer)
        # Control: [S1_x, S1_y, T1_z] (evader)
        # Disturbance: [S2_x, S2_y, T2_z] (pursuer)
        
        # Drone 1 (evader) dynamics - indices 0-9
        # Position derivatives
        dsdt[..., 0] = state[..., 1]  # x1_dot = v1_x
        dsdt[..., 4] = state[..., 5]  # y1_dot = v1_y
        dsdt[..., 8] = state[..., 9]  # z1_dot = v1_z
        
        # Velocity derivatives (with evader control)
        dsdt[..., 1] = -self.Gz * torch.tan(state[..., 2])  # v̇1_x = g * tan(θ1_x)
        dsdt[..., 5] = -self.Gz * torch.tan(state[..., 6])  # v̇1_y = g * tan(θ1_y)
        dsdt[..., 9] = self.k_T / self.mass * control[..., 2] + self.Gz  # v1_z_dot = T1_z - g
        
        # Angle derivatives
        dsdt[..., 2] = state[..., 3] - self.d1 * state[..., 2]  # θ1_x_dot = ω1_x - d1 * θ1_x
        dsdt[..., 6] = state[..., 7] - self.d1 * state[..., 6]  # θ1_y_dot = ω1_y - d1 * θ1_y
        
        # Angular velocity derivatives
        dsdt[..., 3] = -self.d0 * state[..., 2] + self.n0 * self.max_torque * control[..., 0]  # ω̇1_x
        dsdt[..., 7] = -self.d0 * state[..., 6] + self.n0 * self.max_torque * control[..., 1]  # ω̇1_y
        
        # Drone 2 (pursuer) dynamics - indices 10-19
        # Position derivatives
        dsdt[..., 10] = state[..., 11]  # x2_dot = v2_x
        dsdt[..., 14] = state[..., 15]  # y2_dot = v2_y
        dsdt[..., 18] = state[..., 19]  # z2_dot = v2_z
        
        # Velocity derivatives (with pursuer control)
        dsdt[..., 11] = -self.Gz * torch.tan(state[..., 12])  # v̇2_x = g * tan(θ2_x)
        dsdt[..., 15] = -self.Gz * torch.tan(state[..., 16])  # v̇2_y = g * tan(θ2_y)
        dsdt[..., 19] = self.k_T / self.mass * disturbance[..., 2] + self.Gz  # v2_z_dot = T2_z - g
        
        # Angle derivatives
        dsdt[..., 12] = state[..., 13] - self.d1 * state[..., 12]  # θ2_x_dot = ω2_x - d1 * θ2_x
        dsdt[..., 16] = state[..., 17] - self.d1 * state[..., 16]  # θ2_y_dot = ω2_y - d1 * θ2_y
        
        # Angular velocity derivatives
        dsdt[..., 13] = -self.d0 * state[..., 12] + self.n0 * self.max_torque * disturbance[..., 0]  # ω̇2_x
        dsdt[..., 17] = -self.d0 * state[..., 16] + self.n0 * self.max_torque * disturbance[..., 1]  # ω̇2_y
        
        return dsdt

    def hamiltonian(self, state, dvds):
        # Extract velocities and gradients for both drones
        v1 = state[..., [1, 5, 9]]  # [v1_x, v1_y, v1_z]
        v2 = state[..., [11, 15, 19]]  # [v2_x, v2_y, v2_z]
        omega1 = state[..., [3, 7]]  # [ω1_x, ω1_y]
        omega2 = state[..., [13, 17]]  # [ω2_x, ω2_y]
        theta1 = state[..., [2, 6]]  # [θ1_x, θ1_y]
        theta2 = state[..., [12, 16]]  # [θ2_x, θ2_y]

        # Gradients for drone 1 (evader)
        dVdp1 = dvds[..., [0, 4, 8]]  # [dV/dx1, dV/dy1, dV/dz1]
        dVdv1 = dvds[..., [1, 5, 9]]  # [dV/dv1_x, dV/dv1_y, dV/dv1_z]
        dVdtheta1 = dvds[..., [2, 6]]  # [dV/dθ1_x, dV/dθ1_y]
        dVdomega1 = dvds[..., [3, 7]]  # [dV/dω1_x, dV/dω1_y]
        
        # Gradients for drone 2 (pursuer)
        dVdp2 = dvds[..., [10, 14, 18]]  # [dV/dx2, dV/dy2, dV/dz2]
        dVdv2 = dvds[..., [11, 15, 19]]  # [dV/dv2_x, dV/dv2_y, dV/dv2_z]
        dVdtheta2 = dvds[..., [12, 16]]  # [dV/dθ2_x, dV/dθ2_y]
        dVdomega2 = dvds[..., [13, 17]]  # [dV/dω2_x, dV/dω2_y]
        
        # Drone 1 (evader) terms
        ham = (v1 * dVdp1).sum(-1)  # Position derivatives
        ham += dVdv1[..., 0] * (-self.Gz * torch.tan(theta1[..., 0]))  # v1_x term
        ham += dVdv1[..., 1] * (-self.Gz * torch.tan(theta1[..., 1]))  # v1_y term
        ham += dVdv1[..., 2] * self.Gz  # v1_z gravity term
        ham += (omega1 * dVdtheta1).sum(-1)  # Angle derivatives
        ham += dVdtheta1[..., 0] * (-self.d1 * theta1[..., 0])  # θ1_x damping
        ham += dVdtheta1[..., 1] * (-self.d1 * theta1[..., 1])  # θ1_y damping
        ham += dVdomega1[..., 0] * (-self.d0 * theta1[..., 0])  # ω1_x damping
        ham += dVdomega1[..., 1] * (-self.d0 * theta1[..., 1])  # ω1_y damping
        
        # Drone 2 (pursuer) terms
        ham += (v2 * dVdp2).sum(-1)  # Position derivatives
        ham += dVdv2[..., 0] * (-self.Gz * torch.tan(theta2[..., 0]))  # v2_x term
        ham += dVdv2[..., 1] * (-self.Gz * torch.tan(theta2[..., 1]))  # v2_y term
        ham += dVdv2[..., 2] * self.Gz  # v2_z gravity term
        ham += (omega2 * dVdtheta2).sum(-1)  # Angle derivatives
        ham += dVdtheta2[..., 0] * (-self.d1 * theta2[..., 0])  # θ2_x damping
        ham += dVdtheta2[..., 1] * (-self.d1 * theta2[..., 1])  # θ2_y damping
        ham += dVdomega2[..., 0] * (-self.d0 * theta2[..., 0])  # ω2_x damping
        ham += dVdomega2[..., 1] * (-self.d0 * theta2[..., 1])  # ω2_y damping
        
        # Control and disturbance terms
        if self.set_mode == 'avoid':
            # Evader tries to avoid capture (minimize value function)
            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega1[..., 0])  # S1_x
            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega1[..., 1])  # S1_y
            ham += self.k_T / self.mass * self.control_max * torch.relu(dVdv1[..., 2])  # T1_z
            
            # Pursuer tries to capture (maximize value function)
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega2[..., 0])  # S2_x
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega2[..., 1])  # S2_y
            ham -= self.k_T / self.mass * self.control_max * torch.relu(dVdv2[..., 2])  # T2_z

        elif self.set_mode == 'reach':
            # Evader tries to reach target (maximize value function)
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega1[..., 0])  # S1_x
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega1[..., 1])  # S1_y
            ham -= self.k_T / self.mass * self.control_max * torch.relu(dVdv1[..., 2])  # T1_z
            
            # Pursuer tries to prevent reaching (minimize value function)
            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega2[..., 0])  # S2_x
            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega2[..., 1])  # S2_y
            ham += self.k_T / self.mass * self.control_max * torch.relu(dVdv2[..., 2])  # T2_z
        
        return ham

    def boundary_fn(self, state):

        # Extract positions of both drones
        p1 = torch.stack([state[..., 0], state[..., 4], state[..., 8]], dim=-1)  # [x1, y1, z1] - evader
        p2 = torch.stack([state[..., 10], state[..., 14], state[..., 18]], dim=-1)  # [x2, y2, z2] - pursuer

        height = 1.5

        horizontal_dist = torch.sqrt((p1[..., 0] - p2[..., 0])**2 + (p1[..., 1] - p2[..., 1])**2) - self.capture_radius
        vertical_dist = torch.abs(p1[..., 2] - p2[..., 2]) - (height / 2)

        # Case 1: Outside in both directions
        outside_both = (horizontal_dist > 0) & (vertical_dist > 0)
        dist_outside = torch.sqrt(horizontal_dist**2 + vertical_dist**2)

        # Case 2: Outside horizontally, inside vertically
        outside_horiz = (horizontal_dist > 0) & (vertical_dist <= 0)

        # Case 3: Inside horizontally, outside vertically
        outside_vert = (horizontal_dist <= 0) & (vertical_dist > 0)

        # Case 4: Inside both (inside the cylinder)
        inside_both = (horizontal_dist <= 0) & (vertical_dist <= 0)
        dist_inside = torch.maximum(horizontal_dist, vertical_dist)  # most negative

        # Combine all cases
        inter_drone_dist = torch.where(
            outside_both, dist_outside,
            torch.where(
                outside_horiz, horizontal_dist,
                torch.where(
                    outside_vert, vertical_dist,
                    dist_inside
                )
            )
        )

        capture_constraint = inter_drone_dist

        px, py, pz = state[..., 0], state[..., 4], state[..., 8]  # evader positions

        box_bounds = self.box_bounds_.to(state.device)
        
        x_min, x_max = box_bounds[0, 0], box_bounds[0, 1]
        y_min, y_max = box_bounds[2, 0], box_bounds[2, 1]
        z_min, z_max = box_bounds[4, 0], box_bounds[4, 1]

        # Compute per-dimension signed distances to box faces
        dx_min = px - x_min
        dx_max = x_max - px
        dy_min = py - y_min
        dy_max = y_max - py
        dz_min = pz - z_min
        dz_max = z_max - pz

        # Inside: minimum distance to any face (negative inside, zero on surface)
        inside_dist = torch.min(torch.stack([dx_min, dx_max, dy_min, dy_max, dz_min, dz_max], dim=-1), dim=-1).values

        # For outside: compute the per-dimension "over" (how far outside the box in each dim)
        over_x = torch.clamp(px - x_max, min=0) + torch.clamp(x_min - px, min=0)
        over_y = torch.clamp(py - y_max, min=0) + torch.clamp(y_min - py, min=0)
        over_z = torch.clamp(pz - z_max, min=0) + torch.clamp(z_min - pz, min=0)
        # Norm of the "over" vector gives Euclidean distance outside
        outside_dist = torch.norm(torch.stack([over_x, over_y, over_z], dim=-1), dim=-1)

        # If all inside (all distances to faces > 0), use inside_dist; else use outside_dist
        is_inside = (dx_min > 0) & (dx_max > 0) & (dy_min > 0) & (dy_max > 0) & (dz_min > 0) & (dz_max > 0)
        inside_constraint = torch.where(is_inside, inside_dist, -outside_dist)

        
        if self.set_mode == 'avoid':
            # Safe if outside capture radius AND inside bounds
            return torch.minimum(capture_constraint, inside_constraint)
        elif self.set_mode == 'reach':
            return torch.minimum(-capture_constraint, inside_constraint)
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")

    def optimal_control(self, state, dvds):
        # Extract gradients for evader controls
        dVdomega1 = dvds[..., [3, 7]]  # [dV/dω1_x, dV/dω1_y] for torque controls S1_x, S1_y
        dVdv1 = dvds[..., [1, 5, 9]]  # [dV/dv1_x, dV/dv1_y, dV/dv1_z] for thrust control T1_z
        
        control = torch.zeros_like(dVdv1)

        if self.set_mode == 'avoid':
            # Evader tries to avoid capture (minimize value function)
            control[..., 0] = self.control_max * torch.sign(dVdomega1[..., 0])  # S1_x
            control[..., 1] = self.control_max * torch.sign(dVdomega1[..., 1])  # S1_y
            control[..., 2] = self.control_max * torch.clamp((torch.sign(dVdv1[..., 2])), min=0.0)  # T1_z
        elif self.set_mode == 'reach':
            # Evader tries to reach target (maximize value function)
            control[..., 0] = -self.control_max * torch.sign(dVdomega1[..., 0])  # S1_x
            control[..., 1] = -self.control_max * torch.sign(dVdomega1[..., 1])  # S1_y
            control[..., 2] = self.control_max * torch.clamp((-torch.sign(dVdv1[..., 2])), min=0.0)  # T1_z
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")
        return control

    def optimal_disturbance(self, state, dvds):
        # Extract gradients for pursuer controls
        dVdomega2 = dvds[..., [13, 17]]  # [dV/dω2_x, dV/dω2_y] for torque controls S2_x, S2_y
        dVdv2 = dvds[..., [11, 15, 19]]  # [dV/dv2_x, dV/dv2_y, dV/dv2_z] for thrust control T2_z
        
        disturbance = torch.zeros_like(dVdv2)

        if self.set_mode == 'avoid':
            # Pursuer tries to capture (maximize value function)
            disturbance[..., 0] = -self.control_max * torch.sign(dVdomega2[..., 0])  # S2_x
            disturbance[..., 1] = -self.control_max * torch.sign(dVdomega2[..., 1])  # S2_y
            disturbance[..., 2] = self.control_max * torch.clamp((-torch.sign(dVdv2[..., 2])), min=0.0)  # T2_z
        elif self.set_mode == 'reach':
            # Pursuer tries to prevent reaching (minimize value function)
            disturbance[..., 0] = self.control_max * torch.sign(dVdomega2[..., 0])  # S2_x
            disturbance[..., 1] = self.control_max * torch.sign(dVdomega2[..., 1])  # S2_y
            disturbance[..., 2] = self.control_max * torch.clamp((torch.sign(dVdv2[..., 2])), min=0.0)  # T2_z
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")

        return disturbance

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # Wrap θ1_x, θ1_y, θ2_x, θ2_y (indices 2, 6, 12, 16)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        wrapped_state[..., 6] = (wrapped_state[..., 6] + math.pi) % (2 * math.pi) - math.pi
        wrapped_state[..., 12] = (wrapped_state[..., 12] + math.pi) % (2 * math.pi) - math.pi
        wrapped_state[..., 16] = (wrapped_state[..., 16] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        # Transform periodic angles θ1_x, θ1_y, θ2_x, θ2_y to sin/cos components
        # Input: [..., 21] - [t, x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
        #                     x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z]
        # Output: [..., 25] - [t, x1, v1_x, sin(θ1_x), cos(θ1_x), ω1_x, y1, v1_y, sin(θ1_y), cos(θ1_y), ω1_y, z1, v1_z,
        #                      x2, v2_x, sin(θ2_x), cos(θ2_x), ω2_x, y2, v2_y, sin(θ2_y), cos(θ2_y), ω2_y, z2, v2_z]
        
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] + 4  # Add 4 dimensions for sin/cos transforms
        transformed_input = torch.zeros(output_shape, device=input.device)
        
        # Copy non-periodic variables for drone 1
        transformed_input[..., 0] = input[..., 0]  # t (time)
        transformed_input[..., 1] = input[..., 1]  # x1
        transformed_input[..., 2] = input[..., 2]  # v1_x
        transformed_input[..., 5] = input[..., 4]  # ω1_x
        transformed_input[..., 6] = input[..., 5]  # y1
        transformed_input[..., 7] = input[..., 6]  # v1_y
        transformed_input[..., 10] = input[..., 8]  # ω1_y
        transformed_input[..., 11] = input[..., 9]  # z1
        transformed_input[..., 12] = input[..., 10]  # v1_z
        
        # Transform θ1_x, θ1_y to sin/cos
        theta1_x = input[..., 3]
        transformed_input[..., 3] = torch.sin(theta1_x * self.state_var[2])  # sin(θ1_x)
        transformed_input[..., 4] = torch.cos(theta1_x * self.state_var[2])  # cos(θ1_x)
        
        theta1_y = input[..., 7]
        transformed_input[..., 8] = torch.sin(theta1_y * self.state_var[6])  # sin(θ1_y)
        transformed_input[..., 9] = torch.cos(theta1_y * self.state_var[6])  # cos(θ1_y)
        
        # Copy non-periodic variables for drone 2
        transformed_input[..., 13] = input[..., 11]  # x2
        transformed_input[..., 14] = input[..., 12]  # v2_x
        transformed_input[..., 17] = input[..., 14]  # ω2_x
        transformed_input[..., 18] = input[..., 15]  # y2
        transformed_input[..., 19] = input[..., 16]  # v2_y
        transformed_input[..., 22] = input[..., 18]  # ω2_y
        transformed_input[..., 23] = input[..., 19]  # z2
        transformed_input[..., 24] = input[..., 20]  # v2_z
        
        # Transform θ2_x, θ2_y to sin/cos
        theta2_x = input[..., 13]
        transformed_input[..., 15] = torch.sin(theta2_x * self.state_var[12])  # sin(θ2_x)
        transformed_input[..., 16] = torch.cos(theta2_x * self.state_var[12])  # cos(θ2_x)
        
        theta2_y = input[..., 17]
        transformed_input[..., 20] = torch.sin(theta2_y * self.state_var[16])  # sin(θ2_y)
        transformed_input[..., 21] = torch.cos(theta2_y * self.state_var[16])  # cos(θ2_y)
        
        return transformed_input

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()
    
    def control_range(self, state):
        return self.control_range_.tolist()

    def disturbance_range(self, state):
        return self.disturbance_range_.tolist()

    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[:, 0], self.control_range_[:, 1])

    def bound_disturbance(self, disturbance):
        return torch.clamp(disturbance, self.disturbance_range_[:, 0], self.disturbance_range_[:, 1])
    
    def clamp_control(self, state, control):
        return self.bound_control(control)
    
    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def cost_fn(self, state_traj):
        # Use boundary function for consistency
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def plot_config(self):
        return {
            'state_slices': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0,  # Drone 1
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0], # Drone 2
            'state_labels': ['x1', 'v1_x', 'θ1_x', 'ω1_x', 'y1', 'v1_y', 'θ1_y', 'ω1_y', 'z1', 'v1_z',
                           'x2', 'v2_x', 'θ2_x', 'ω2_x', 'y2', 'v2_y', 'θ2_y', 'ω2_y', 'z2', 'v2_z'],
            'x_axis_idx': 0,  # x1
            'y_axis_idx': 4,  # y1
            'z_axis_idx': 8,  # z1
        }

class Quadrotor(Dynamics):
    disturbance_dim = 0

    def __init__(self, collisionR: float, collective_thrust_max: float,  set_mode: str):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0

        self.collisionR = collisionR
        self.reach_fn_weight = 1.
        self.avoid_fn_weight = 0.3
        self.state_range_ = torch.tensor(
            [
                [-3.0, 3.0],
                [-3.0, 3.0],
                [-3.0, 3.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
            ]
        ).to(device)
        self.control_range_ = torch.tensor(
            [
                [-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max],
            ]
        ).to(device)
        self.eps_var_control = torch.tensor([20, 8, 8, 4]).to(device)
        self.control_init = torch.tensor([-self.Gz * 0.0, 0, 0, 0]).to(device)

        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0
        if set_mode=='reach_avoid':
            l_type='brat_hjivi'
        else:
            l_type='brt_hjivi'
        super().__init__(
            name='Quadrotor', loss_type=l_type, set_mode=set_mode,
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=self.disturbance_dim,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=(math.sqrt(3.0**2 + 3.0**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(3.0**2 + 3.0**2)/2,
            value_normto=0.02,
            deepReach_model='exact',
        )
    def normalize_q(self, x):
        # normalize quaternion
        normalized_x = x*1.0
        q_tensor = x[..., 3:7]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2,dim=-1)  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor
        return normalized_x
    
    def clamp_state_input(self, state_input):
        return self.normalize_q(state_input)

    def control_range(self, state):
        return [[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def periodic_transform_fn(self, input):
        return input.to(device)
    
    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return self.normalize_q(wrapped_state)

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / \
            self.m * f
        dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / \
            self.m * f
        dsdt[..., 9] = self.Gz + (1 - 2 * torch.pow(qx, 2) - 2 *
                                  torch.pow(qy, 2)) * self.CT / self.m * f
        dsdt[..., 10] = (control[..., 1]
                         ) * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 11] = (control[..., 2]
                         ) * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 12] = (control[..., 3]) * 1.0

        return dsdt
    
    def dist_to_cylinder(self, state, a, b):
        '''for cylinder with full body collision'''
        state_=state*1.0
        state_[...,0]=state_[...,0]- a
        state_[...,1]=state_[...,1]- b

        # create normal vector
        v = torch.zeros_like(state_[..., 4:7])
        v[..., 2] = 1
        v = quaternion.quaternion_apply(state_[..., 3:7], v)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # compute vector from center of quadrotor to the center of cylinder
        px = state_[..., 0]
        py = state_[..., 1]

        # get full body distance
        dist = torch.norm(state_[..., :2], dim=-1)
        # return dist- self.collisionR
        dist = dist- torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                           + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR
    
    def reach_fn(self, state):
        state_=state*1.0
        state_[...,0]=state_[...,0] - 0.
        state_[...,1]=state_[...,1]
        return (torch.norm(state[..., :2], dim=-1)-0.3)*self.reach_fn_weight

    def avoid_fn(self, state):
        return self.avoid_fn_weight*torch.minimum(self.dist_to_cylinder(state,0.0,0.75), self.dist_to_cylinder(state,0.0,-0.75))

    def boundary_fn(self, state):
        if self.set_mode=='avoid':
            return self.dist_to_cylinder(state,0.0,0.0)
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))


    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-1, 1]
        target_state_range[1] = [-0.25, 0.25]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])
    
    def cost_fn(self, state_traj):
        if self.set_mode=='avoid':
            return torch.min(self.boundary_fn(state_traj), dim=-1).values
        else:
            # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            return torch.min(torch.clamp(reach_values, min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)),dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode in ['reach', 'reach_avoid']:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham -= torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max

            ham -= torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham += torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max

            ham += torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

        else:
            raise NotImplementedError

        return ham

    def optimal_control(self, state, dvds):
        if self.set_mode in ['reach', 'reach_avoid']:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = -self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = -self.dwx_max * torch.sign(dvds[..., 10])
            u3 = -self.dwy_max * torch.sign(dvds[..., 11])
            u4 = -self.dwz_max * torch.sign(dvds[..., 12])
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)


    def plot_config(self):
        return {
            'state_slices': [0.96,  1.18,  0.54,  0.44, -0.45,  0.27, -0.73, -2.83, -1.07, -3.34, 3.19, -2.80,  3.43],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }
    

class QuadrotorWithDist(Quadrotor):
    disturbance_dim = 6  #  3 for forces 3 for torques

    def __init__(
        self, collisionR: float, collective_thrust_max: float, set_mode: str, disturbance_max: float
    ):
        super().__init__(collisionR, collective_thrust_max, set_mode)
        self.disturbance_max = disturbance_max
        self.wind_force_max = disturbance_max
        self.wind_torque_max = disturbance_max
        # First 3: force, Last 3: torque
        self.disturbance_range_ = torch.tensor(
            [[-self.wind_force_max, self.wind_force_max]] * 3
            + [[-self.wind_torque_max, self.wind_torque_max]] * 3
        ).to(device)
        self.disturbance_init = torch.zeros(self.disturbance_dim).to(device)
        self.name = 'QuadrotorWithDist'
        self.eps_var_control = torch.tensor([20, 8, 8, 4], dtype=torch.float32, device=device)  # Example for control
        self.eps_var_disturbance = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32, device=device)  # Example for disturbance

    def disturbance_range(self, state):
        return self.disturbance_range_.cpu().tolist()

    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def bound_disturbance(self, disturbance):
        return torch.clamp(
            disturbance,
            self.disturbance_range_[..., 0],
            self.disturbance_range_[..., 1],
        )

    def dsdt(self, state, control, disturbance):
        dsdt = super().dsdt(state, control, disturbance)
        
        # Add wind force to translational acceleration (vx, vy, vz)
        dsdt[..., 7] += disturbance[..., 0]  # vx
        dsdt[..., 8] += disturbance[..., 1]  # vy
        dsdt[..., 9] += disturbance[..., 2]  # vz
        # Add wind torque to rotational acceleration (wx, wy, wz)
        dsdt[..., 10] += disturbance[..., 3]  # wx
        dsdt[..., 11] += disturbance[..., 4]  # wy
        dsdt[..., 12] += disturbance[..., 5]  # wz
        return dsdt
    
    def hamiltonian(self, state, dvds):
        # Call the parent Hamiltonian
        ham = super().hamiltonian(state, dvds)

        # Add adversarial disturbance terms
        dist_terms = (
            torch.abs(dvds[..., 7]) * self.wind_force_max +
            torch.abs(dvds[..., 8]) * self.wind_force_max +
            torch.abs(dvds[..., 9]) * self.wind_force_max +
            torch.abs(dvds[..., 10]) * self.wind_torque_max +
            torch.abs(dvds[..., 11]) * self.wind_torque_max +
            torch.abs(dvds[..., 12]) * self.wind_torque_max
        )

        if self.set_mode in ['avoid', 'reach_avoid']:
            ham -= dist_terms
        elif self.set_mode == 'reach':
            ham += dist_terms
        else:
            raise NotImplementedError

        return ham

    def optimal_disturbance(self, state, dvds):
        """
        Set the optimal disturbance direction for both 'avoid' and 'reach' modes.
        """
        grad_disturb = dvds[..., 7:13]
        if self.set_mode == 'avoid':
            sign = torch.sign(grad_disturb)
        elif self.set_mode == 'reach':
            sign = -torch.sign(grad_disturb)
        else:
            raise ValueError(f"Unknown set_mode: {self.set_mode}")

        force = sign[..., :3] * self.wind_force_max
        torque = sign[..., 3:] * self.wind_torque_max
        disturbance = torch.cat((force, torque), dim=-1)
        return disturbance


class F1tenth(Dynamics):
    def __init__(self):
        # variable for dynamics
        self.mu = 1.0489
        self.C_Sf = 4.718
        self.C_Sr = 5.4562
        self.lf = 0.15875
        self.lr = 0.17145
        self.h = 0.074
        self.m = 3.74
        self.I = 0.04712
        self.s_min = -0.4189
        self.s_max = 0.4189
        self.sv_min = -3.2
        self.sv_max = 3.2
        self.v_switch = 7.319
        self.a_max = 9.51
        self.v_min = 0.1
        self.v_max = 10.0
        self.omega_max= 6.0
        self.delta_t = 0.01
        self.g = 9.81
        self.lwb = self.lf + self.lr

        self.v_mean = (self.v_min + self.v_max) / 2
        self.v_var = (self.v_max - self.v_min) / 2

        # map info
        # self.dt = np.load(map_path)
        self.origin = [-78.21853769831466, -44.37590462453829]
        self.resolution = 0.062500
        self.width = 1600
        self.height = 1600


        # control constraints
        self.input_steering_v_max = self.sv_max
        self.input_acceleration_max = self.a_max

        self.xmean=62.5/2
        self.xvar=62.5/2
        self.ymean=25
        self.yvar=25


        self.x_min=self.xmean-self.xvar
        self.x_max=self.xmean+self.xvar
        self.y_min=self.ymean-self.yvar
        self.y_max=self.ymean+self.yvar

        self.state_range_ = torch.tensor(
            [
                [self.x_min, self.x_max],
                [self.y_min, self.y_max],
                [-0.4189, 0.4189],
                [self.v_min, self.v_max],
                [-math.pi, math.pi],
                [-self.omega_max, self.omega_max],
                [-1, 1],
            ]
        ).to(device)
        self.control_range_ = torch.tensor(
            [[self.sv_min, self.sv_max], [-self.a_max, self.a_max]]
        ).to(device)
        self.eps_var_control = torch.tensor([self.sv_max**2, self.a_max**2]).to(device)
        self.control_init = torch.tensor([0.0, 0.0]).to(device)

        # for the track
        self.obstaclemap_file = 'dynamics/F1_map_obstaclemap.mat'
        self.pixel2world = 0.0625
        self.obstacle_map = spio.loadmat(self.obstaclemap_file)
        self.obstacle_map = self.obstacle_map['obs_map']
        self.obstacle_map[self.obstacle_map == -0.] = 1
        self.obstacle_map = self.obstacle_map[int(self.y_min/self.pixel2world):int(self.y_max/self.pixel2world)+1,
                                            int(self.x_min/self.pixel2world):int(self.x_max/self.pixel2world)+1]
        self.obstacle_map = torch.tensor(self.obstacle_map) 


        self.world_range = self.state_range_.cpu().numpy()
        self.x_rangearray = torch.arange(self.obstacle_map.shape[0])
        self.y_rangearray = torch.arange(self.obstacle_map.shape[1])

        
        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0

        super().__init__(
            name='F1tenth', loss_type='brt_hjivi', set_mode='avoid',
            state_dim=7, input_dim=9, control_dim=2, disturbance_dim=0,
            
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=0.5, # mean of expected value function
            value_var=1.5, # (max - min)/2.0 of expected value function
            value_normto=0.02,
            deepReach_model='exact'
        )

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return [
            [self.x_min, self.x_max], 
            [self.y_min, self.y_max],                      # y
            [-0.4189, 0.4189],               # steering angle
            [0.1, 8.0],                   # velocity
            [-math.pi, math.pi],                  # pose theta
            [-4.5, 4.5],                        # pose theta rate
            [-0.8, 0.8],                       # slip angle
        ]
    
    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :5] = input[..., :5]
        transformed_input[..., 5] = torch.sin(input[..., 5]*self.state_var[4])
        transformed_input[..., 6] = torch.cos(input[..., 5]*self.state_var[4])
        transformed_input[..., 7:] = input[..., 6:]
        return transformed_input.to(device)
    
    def dsdt(self, state, control, disturbance):
        # here the control is steering angle v and acceleration
        f = torch.zeros_like(state)
        current_vel = state[..., 3] # [1, 65000]
        kinematic_mask = torch.abs(current_vel) < 0.5
        # switch to kinematic model for small velocities
        if torch.any(kinematic_mask):
            # print(f"kinematic_mask is {kinematic_mask.shape}")
            if len(kinematic_mask.shape)==1:
                sample_idx = kinematic_mask.nonzero(as_tuple=True)[0]
                x_ks = state[kinematic_mask][..., 0:5]
                u_ks = control[kinematic_mask]
                f_ks = torch.zeros_like(x_ks)
                f_ks[..., 0] = x_ks[..., 3]*torch.cos(x_ks[..., 4])
                f_ks[..., 1] = x_ks[..., 3]*torch.sin(x_ks[..., 4])
                f_ks[..., 2] = u_ks[..., 0]
                f_ks[..., 3] = u_ks[..., 1]
                f_ks[..., 4] = x_ks[..., 3]/self.lwb*torch.tan(x_ks[..., 2])
                f[sample_idx, :5] = f_ks
                f[sample_idx, 5] = u_ks[..., 1]/self.lwb*torch.tan(state[kinematic_mask][..., 2])+state[kinematic_mask][..., 3]/(self.lwb*torch.cos(state[kinematic_mask][..., 2])**2)*u_ks[..., 0]
                f[sample_idx, 6] = 0.
            else:
                batch_idx, sample_idx = kinematic_mask.nonzero(as_tuple=True)
                x_ks = state[kinematic_mask][..., 0:5]
                u_ks = control[kinematic_mask]
                f_ks = torch.zeros_like(x_ks)
                f_ks[..., 0] = x_ks[..., 3]*torch.cos(x_ks[..., 4])
                f_ks[..., 1] = x_ks[..., 3]*torch.sin(x_ks[..., 4])
                f_ks[..., 2] = u_ks[..., 0]
                f_ks[..., 3] = u_ks[..., 1]
                f_ks[..., 4] = x_ks[..., 3]/self.lwb*torch.tan(x_ks[..., 2])
                f[batch_idx, sample_idx, :5] = f_ks
                f[batch_idx, sample_idx, 5] = u_ks[..., 1]/self.lwb*torch.tan(state[kinematic_mask][..., 2])+state[kinematic_mask][..., 3]/(self.lwb*torch.cos(state[kinematic_mask][..., 2])**2)*u_ks[..., 0]
                f[batch_idx, sample_idx, 6] = 0.

        dynamic_mask = ~kinematic_mask
        if torch.any(dynamic_mask):
            if len(kinematic_mask.shape)==1:
                sample_idx = dynamic_mask.nonzero(as_tuple=True)[0]
                f[sample_idx, 0] = state[dynamic_mask][..., 3]*torch.cos(state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4])
                f[sample_idx, 1] = state[dynamic_mask][..., 3]*torch.sin(state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4])
                f[sample_idx, 2] = control[dynamic_mask][..., 0]
                f[sample_idx, 3] = control[dynamic_mask][..., 1]
                f[sample_idx, 4] = state[dynamic_mask][..., 5]
                f[sample_idx, 5] = -self.mu*self.m/(state[dynamic_mask][..., 3]*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h) + self.lr**2*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 5] \
                        +self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) - self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                        +self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*state[dynamic_mask][..., 2]
                f[sample_idx, 6] = (self.mu/(state[dynamic_mask][..., 3]**2*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h)*self.lr - self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*self.lf)-1)*state[dynamic_mask][..., 5] \
                        -self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) + self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                        +self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 2]
            else:
                batch_idx, sample_idx = dynamic_mask.nonzero(as_tuple=True)
                f[batch_idx, sample_idx, 0] = state[dynamic_mask][..., 3]*torch.cos(state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4])
                f[batch_idx, sample_idx, 1] = state[dynamic_mask][..., 3]*torch.sin(state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4])
                f[batch_idx, sample_idx, 2] = control[dynamic_mask][..., 0]
                f[batch_idx, sample_idx, 3] = control[dynamic_mask][..., 1]
                f[batch_idx, sample_idx, 4] = state[dynamic_mask][..., 5]
                f[batch_idx, sample_idx, 5] = -self.mu*self.m/(state[dynamic_mask][..., 3]*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h) + self.lr**2*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 5] \
                        +self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) - self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                        +self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*state[dynamic_mask][..., 2]
                f[batch_idx, sample_idx, 6] = (self.mu/(state[dynamic_mask][..., 3]**2*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h)*self.lr - self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*self.lf)-1)*state[dynamic_mask][..., 5] \
                        -self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) + self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                        +self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 2]
        #------------------------------OPT CTRL--------------------------------

        return f

    def clamp_state_input(self, state_input):
        full_input=torch.cat((torch.ones(state_input.shape[0],1).to(state_input),state_input),dim=-1)
        state=self.input_to_coord(full_input)[...,1:]
        lx=self.boundary_fn(state)
        return state_input[lx>=-1]

    def clamp_verification_state(self, state):
        lx=self.boundary_fn(state)
        return state[lx>=0]

    def clamp_control(self, state, control):
        control_clamped=control*1.0
        
        smax_mask = state[...,2]>self.s_max-0.01
        smin_mask = state[...,2]<-self.s_max+0.01
        if len(smax_mask.shape)==1:
            max_sample_idx = smax_mask.nonzero(as_tuple=True)[0]
            control_clamped[max_sample_idx, 0] = torch.clamp(control_clamped[max_sample_idx, 0],max=0)
            min_sample_idx = smin_mask.nonzero(as_tuple=True)[0]
            control_clamped[min_sample_idx, 0] = torch.clamp(control_clamped[min_sample_idx, 0],min=0)
            
        else:
            max_batch_idx, max_sample_idx = smax_mask.nonzero(as_tuple=True)
            control_clamped[max_batch_idx, max_sample_idx, 0] = torch.clamp(control_clamped[max_batch_idx, max_sample_idx, 0],max=0)
            min_batch_idx, min_sample_idx = smin_mask.nonzero(as_tuple=True)
            control_clamped[min_batch_idx, min_sample_idx, 0] = torch.clamp(control_clamped[min_batch_idx, min_sample_idx, 0],min=0)

        accelerate_upper = torch.ones(state.shape[:-1], device=state.device) * self.input_acceleration_max
        accelerate_upper[state[..., 3] > self.v_switch] = self.input_acceleration_max * self.v_switch / state[state[..., 3] > self.v_switch][..., 3]
        
        acc_mask=control_clamped[...,1]>accelerate_upper
        if len(acc_mask.shape)==1:
            sample_idx = acc_mask.nonzero(as_tuple=True)[0]
            control_clamped[sample_idx,1]=accelerate_upper[sample_idx]
        else:
            batch_idx, sample_idx = acc_mask.nonzero(as_tuple=True)
            control_clamped[batch_idx, sample_idx,1]=accelerate_upper[batch_idx, sample_idx]

        assert ((accelerate_upper-control_clamped[...,1])>=0.0).all()
        return control_clamped
    
    def interpolation(self, state_pixel_coords):
        self.obstacle_map=self.obstacle_map.to(state_pixel_coords)
        # Find the indices surrounding the query points
        x0 = torch.floor(state_pixel_coords[..., 0]).long()
        x1 = x0 + 1
        y0 = torch.floor(state_pixel_coords[..., 1]).long()
        y1 = y0 + 1
        # Ensure indices are within bounds
        x0 = torch.clamp(x0, 0, self.x_rangearray.size(0) - 1)
        x1 = torch.clamp(x1, 0, self.x_rangearray.size(0) - 1)
        y0 = torch.clamp(y0, 0, self.y_rangearray.size(0) - 1)
        y1 = torch.clamp(y1, 0, self.y_rangearray.size(0) - 1)

        # Gather the values at the corner points for each query point
        v00 = self.obstacle_map[x0, y0]
        v01 = self.obstacle_map[x0, y1]
        v10 = self.obstacle_map[x1, y0]
        v11 = self.obstacle_map[x1, y1]
        # Compute the fractional part for each query point
        x_frac = state_pixel_coords[..., 0] - x0.float()
        y_frac = state_pixel_coords[..., 1] - y0.float()
        # Bilinear interpolation for each query point
        v0 = v00 * (1 - x_frac) + v10 * x_frac
        v1 = v01 * (1 - x_frac) + v11 * x_frac
        # Interpolated value
        interp_values = v0 * (1 - y_frac) + v1 * y_frac
        return interp_values
    
    def boundary_fn(self, state): 
        # MPC: state = B * N * H * 7
        # DeepReach: state = B * 7
        # Takes the cordinates in the real world and returns the lx for the obstacles at those coords
        # shift the origin so that the min is 0
        shiftedCoords = state - torch.tensor(self.world_range[...,0].reshape(self.state_dim,), device = state.device) # num states involve time as well
        # extract and flip the x and y pos for image space query
        if shiftedCoords.shape[0] == 1:
            shiftedCoords_pos_world = np.squeeze(shiftedCoords[...,0:2])
        else:
            shiftedCoords_pos_world = shiftedCoords[...,0:2]
        if len(shiftedCoords_pos_world.shape)==2:
            shiftedCoords_pos_image =  torch.fliplr(shiftedCoords_pos_world) # B*2 for deepreach, B*N*H*2 for MPC 
        else:
            shiftedCoords_pos_image=torch.flip(shiftedCoords_pos_world, [-1])
        # convert the world coordinates to pixel coordinates
        shiftedCoords_pos_pixel = shiftedCoords_pos_image/self.pixel2world # note this does not have to be integers due to the regularGridInterpolator
        # query the generator
        obstacle_value = self.interpolation(shiftedCoords_pos_pixel)  # obstacle value only depends on pos
        # obstacle_value = obstacle_value.reshape([obstacle_value.shape[0],1]) # this should be the lx
        return obstacle_value
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            opt_control=self.optimal_control(state,dvds)
            dsdt_=self.dsdt(state,opt_control,None)
            ham=torch.sum(dvds*dsdt_,dim=-1)
        return ham

    def optimal_control(self, state, dvds):

        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            unsqueeze_u=False
            if state.shape[0]==1:
                state=state.squeeze(0)
                dvds=dvds.squeeze(0)
                unsqueeze_u=True
            batch_dims = state.shape[:-1]
            u = torch.zeros(*batch_dims, 2, device=state.device)

            
            kinematic_mask = torch.abs(state[..., 3]) < 0.5
            # if the number of kinematic_mask's dimensional is greater than 1, print
            
            if torch.any(kinematic_mask):
                if len(kinematic_mask.shape)==1:
                    sample_idx = kinematic_mask.nonzero(as_tuple=True)[0]
                    u[sample_idx, 0] = self.input_steering_v_max * torch.sign(dvds[kinematic_mask][..., 2] + dvds[kinematic_mask][..., 5] * state[kinematic_mask][..., 3] / (self.lwb * torch.cos(state[kinematic_mask][..., 2])**2))
                    u[sample_idx, 1] = self.input_acceleration_max * torch.sign(dvds[kinematic_mask][..., 3] + dvds[kinematic_mask][..., 5] / self.lwb * torch.tan(state[kinematic_mask][..., 2]))
                else:
                    batch_idx, sample_idx = kinematic_mask.nonzero(as_tuple=True)
                    u[batch_idx, sample_idx, 0] = self.input_steering_v_max * torch.sign(dvds[kinematic_mask][..., 2] + dvds[kinematic_mask][..., 5] * state[kinematic_mask][..., 3] / (self.lwb * torch.cos(state[kinematic_mask][..., 2])**2))
                    u[batch_idx, sample_idx, 1] = self.input_acceleration_max * torch.sign(dvds[kinematic_mask][..., 3] + dvds[kinematic_mask][..., 5] / self.lwb * torch.tan(state[kinematic_mask][..., 2]))

            dynamic_mask = ~kinematic_mask
            if torch.any(dynamic_mask):
                if len(kinematic_mask.shape)==1:
                    sample_idx = dynamic_mask.nonzero(as_tuple=True)[0]
                    u[sample_idx, 0] = self.input_steering_v_max * torch.sign(dvds[dynamic_mask][..., 2])

                    u[sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[dynamic_mask][..., 3] \
                        + dvds[dynamic_mask][..., 5] * ((-self.mu * self.m / (state[dynamic_mask][..., 3] * self.I * (self.lr + self.lf))*(-self.lf**2*self.C_Sf*self.h + self.lr**2*self.C_Sr*self.h))*state[dynamic_mask][..., 5] \
                            + self.mu * self.m / (self.I * (self.lr + self.lf)) * (self.lr*self.C_Sr*self.h + self.lf*self.C_Sf*self.h)*state[dynamic_mask][..., 6] \
                            - self.mu * self.m / (self.I * (self.lr + self.lf)) * self.lf*self.C_Sf*self.h*state[dynamic_mask][..., 2])\
                        + dvds[dynamic_mask][..., 6] * ((self.mu / (state[dynamic_mask][..., 3]**2 * (self.lr + self.lf)) * (self.C_Sr*self.h*self.lr + self.C_Sf*self.h*self.lf))*state[dynamic_mask][..., 5]
                            - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * (self.C_Sr*self.h - self.C_Sf*self.h)*state[dynamic_mask][..., 6]
                            - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * self.C_Sf*self.h*state[dynamic_mask][..., 2])
                    )
                else:
                    batch_idx, sample_idx = dynamic_mask.nonzero(as_tuple=True)

                    u[batch_idx, sample_idx, 0] = self.input_steering_v_max * torch.sign(dvds[dynamic_mask][..., 2])

                    u[batch_idx, sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[dynamic_mask][..., 3] \
                        + dvds[dynamic_mask][..., 5] * ((-self.mu * self.m / (state[dynamic_mask][..., 3] * self.I * (self.lr + self.lf))*(-self.lf**2*self.C_Sf*self.h + self.lr**2*self.C_Sr*self.h))*state[dynamic_mask][..., 5] \
                            + self.mu * self.m / (self.I * (self.lr + self.lf)) * (self.lr*self.C_Sr*self.h + self.lf*self.C_Sf*self.h)*state[dynamic_mask][..., 6] \
                            - self.mu * self.m / (self.I * (self.lr + self.lf)) * self.lf*self.C_Sf*self.h*state[dynamic_mask][..., 2])\
                        + dvds[dynamic_mask][..., 6] * ((self.mu / (state[dynamic_mask][..., 3]**2 * (self.lr + self.lf)) * (self.C_Sr*self.h*self.lr + self.C_Sf*self.h*self.lf))*state[dynamic_mask][..., 5]
                            - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * (self.C_Sr*self.h - self.C_Sf*self.h)*state[dynamic_mask][..., 6]
                            - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * self.C_Sf*self.h*state[dynamic_mask][..., 2])
                    )
            u=self.clamp_control(state,u)
            if unsqueeze_u:
                u=u[None,...]
        return u

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 4] = (
            wrapped_state[..., 4] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state
    
    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 8.0, 0, 0, 0],
            'state_labels': ['x', 'y', 'sangle', 'v', 'posetheta', 'poserate', 'slipangle'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }
    
class LessLinearND(Dynamics):
    def __init__(self, N:int, gamma:float, mu:float, alpha:float, goalR:float):
        u_max, set_mode = 0.5, "reach" # TODO: unfix

        self.N = N 
        self.u_max = u_max
        self.input_center = torch.zeros(N-1)
        self.input_shape = "box"
        self.game = set_mode
        self.A = (
            -0.5 * torch.eye(N)
            - torch.cat(
                (
                    torch.cat((torch.zeros(1, 1), torch.ones(N - 1, 1)), 0),
                    torch.zeros(N, N - 1),
                ),
                1,
            )
        ).to(device)
        self.B = torch.cat((torch.zeros(1, N - 1), 0.4 * torch.eye(N - 1)), 0).to(device)
        self.Bumax = u_max * torch.matmul(self.B, torch.ones(self.N - 1).to(device)).unsqueeze(
            0
        ).unsqueeze(0).to(device)
        self.C = torch.cat((torch.zeros(1,N-1), 0.1*torch.eye(N-1)), 0)
        self.gamma, self.mu, self.alpha = gamma, mu, alpha
        self.gamma_orig, self.mu_orig, self.alpha_orig = gamma, mu, alpha

        self.goalR_2d = goalR
        self.goalR = ((N-1) ** 0.5) * self.goalR_2d # accounts for N-dimensional combination
        self.ellipse_params = torch.cat((((N-1) ** 0.5) * torch.ones(1), torch.ones(N-1) / 1.), 0) # accounts for N-dimensional combination

        self.state_range_ = torch.tensor([[-1, 1] for _ in range(self.N)]).to(device)
        self.control_range_ = torch.tensor([[-u_max, u_max] for _ in range(self.N - 1)]).to(device)
        self.eps_var_control = torch.tensor([u_max for _ in range(self.N - 1)]).to(device)
        self.control_init = torch.tensor([0.0 for _ in range(self.N - 1)]).to(device)

        super().__init__(
            name='50D system', loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=N, input_dim=N+1, control_dim=N-1, disturbance_dim=N-1,
            state_mean=[0 for _ in range(N)], 
            state_var=[1 for _ in range(N)],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepReach_model="exact",
        )

    def vary_nonlinearity(self, epsilon):
        self.gamma = epsilon * self.gamma_orig
        self.mu = epsilon * self.mu_orig
        # self.alpha = epsilon * self.alpha_orig #shouldn't be varied since its not a scalar (1-\lambda) l(\cdot) +  \lambda f(\cdot)

    def state_test_range(self):
        return [[-1, 1] for _ in range(self.N)]
    
    def state_verification_range(self):
        return [[-1, 1] for _ in range(self.N)]
    
    def control_range(self, state):
        return self.control_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state
        
    # LessLinear dynamics
    # \dot xN    = (aN \cdot x) + (no ctrl or dist) + mu * sin(alpha * xN) * xN^2
    # \dot xi    = (ai \cdot x) + bi * ui + ci * di - gamma * xi * xN^2
    # i.e.
    # \dot x = Ax + Bu + Cd + NLterm(x, gamma, mu, alpha)
    # def dsdt(self, state, control, disturbance):
    #     dsdt = torch.zeros_like(state)
    #     nl_term_N = self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 0] * state[..., 0]
    #     nl_term_i = torch.multiply(-self.gamma * state[..., 0] * state[..., 0], state[..., 1:])
    #     dsdt[..., :] = torch.matmul(self.A, state[..., :]) + torch.matmul(self.B, control[..., :]) + torch.cat((nl_term_N, nl_term_i), 0)
    #     return dsdt
    def dsdt(self, state, control, disturbance):
        x0 = state[..., 0]  # shape: (...)
        x_rest = state[..., 1:]  # shape: (..., n-1)

        # Nonlinear terms
        nl_term_N = self.mu * torch.sin(self.alpha * x0) * x0 * x0  # shape: (...)
        nl_term_N = nl_term_N.unsqueeze(-1)  # shape: (..., 1)

        x0_squared = (x0 ** 2).unsqueeze(-1)  # shape: (..., 1)
        nl_term_i = -self.gamma * x0_squared * x_rest  # broadcasted: (..., n-1)

        nl_term = torch.cat([nl_term_N, nl_term_i], dim=-1)  # shape: (..., n)

        # Linear terms
        linear_term = torch.matmul(state, self.A.T) + torch.matmul(control, self.B.T)

        return linear_term + nl_term

    
    def periodic_transform_fn(self, input):
        return input.to(device)
    
    def boundary_fn(self, state):
        if self.ellipse_params.device != state.device:  # FIXME: Patch to cover de/attached state bug
            self.ellipse_params = self.ellipse_params.to(device)

        return 0.5 * (torch.square(torch.norm(self.ellipse_params * state[..., :], dim=-1)) - (self.goalR ** 2))
        # return 0.5 * (torch.square(torch.norm(torch.cat((((self.N-1)**0.5)*torch.ones(1),torch.ones(self.N-1)),0) * state[..., :], dim=-1)) - (self.goalR ** 2))

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):

        nl_term_N = (self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 0] * state[..., 0]).unsqueeze(-1)
        nl_term_i = (-self.gamma * state[..., 0] * state[..., 0]).t() * state[..., 1:]
        pAx = (dvds * (torch.matmul(state, self.A.t()) + torch.cat((nl_term_N, nl_term_i), 2))).sum(2)
        pBumax = (torch.abs(dvds) * self.Bumax).sum(2)

        if self.set_mode == 'reach':
            return pAx - pBumax
        elif self.set_mode == 'avoid':
            return pAx + pBumax

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return -self.u_max * torch.sign(dvds[..., 1:])
        elif self.set_mode == 'avoid':
            return self.u_max * torch.sign(dvds[..., 1:])

    def optimal_disturbance(self, state, dvds):
        return 0.0
    
    def plot_config(self): # FIXME
        return {
            'state_slices': [0 for _ in range(self.N)],
            'state_labels': ['xN'] + ['x' + str(i) for i in range(1, self.N)],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

   