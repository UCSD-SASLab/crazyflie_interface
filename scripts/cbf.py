#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.msg import Float64MultiArray
import numpy as np
import rowan
import cvxpy as cp
import jax
import jax.numpy as jnp

class ExtendedCBF:  # Barrier function for safety enforcement
    def __init__(self):
        self.ndim = 4
        self.control_dim = 2
        self.umin = np.array([-0.2, 4.0])
        self.umax = np.array([0.2, 16.0])

    def vf(self, state):
        scaling_pos_z = 1.0
        scaling_vel_z = 0.1
        scaling_pos_y = 1.0
        scaling_vel_y = 0.05
        state = jnp.array(state)
        # circle_constraint = 1.5 - scaling_pos*(jnp.sqrt(state[0]**2 + state[1]**2)) - scaling_vel*(jnp.sqrt(state[2]**2 + state[3]**2))
        z_constraint = 1.5 - scaling_pos_z*state[1] - scaling_vel_z*state[3]
        y_constraint = 1.5 - scaling_pos_y*state[0] - scaling_vel_y*state[2]
        #square_constraint_y = 1.5 - jnp.abs(state[0])  # Bound at y = ±5
        #square_constraint_z = 1.5 - jnp.abs(state[1])  # Bound at z = ±5
        #return jnp.minimum(circle_constraint, jnp.minimum(square_constraint_y, square_constraint_z))
        #return z_constraint
        return y_constraint

    def grad_vf(self, state):
        state = jnp.array(state)
        # if np.isclose(state[0]**2 + state[1]**2, 0.0):
        #     return jnp.zeros(4)
        return jax.grad(self.vf)(state)

    def control_jacobian(self, state):
        out = np.zeros((self.ndim, self.control_dim))
        out[2, 0] = -9.81  # Roll affects y_dot
        out[3, 1] = 1.0   # Thrust affects z_dot
        return out

    def open_loop_dynamics(self, state):
        out = np.zeros_like(state)
        out[0] = state[2]  # y_dot
        out[1] = state[3]  # z_dot
        out[3] = -9.81     # gravity
        return out

class SafetyFilter:
    def __init__(self, cbf, logger):
        self.cbf = cbf
        self.dynamics = self.cbf  # Now it directly refers to ExtendedCBF
        self.gamma = 1.0  #0.6 (3.0 for z)
        self.logger = logger  # Logging for debugging

    def __call__(self, state, nominal_control):
        try:
            #logger for vf_value which is for debug
            #vf_value = self.cbf.vf(state)

            vf = self.cbf.vf(state)
            self.logger.info(f"vf(state) = {vf}")
            grad_vf = self.cbf.grad_vf(state)
            Lg_v = grad_vf @ self.dynamics.control_jacobian(state)
            Lf_v = grad_vf @ self.dynamics.open_loop_dynamics(state)

            u = cp.Variable(2)  # Control input is [roll, thrust]
            weights = np.diag([12.0, 1.0])  # Less aggressive on thrust, more on roll (12,1)
            obj = cp.Minimize(cp.quad_form(u - nominal_control, weights))

            constraints = [
                Lf_v + Lg_v @ u >= -self.gamma * vf,
                # u[0] >= -1.0, u[0] <= 1.0,  # Roll bounds
                # u[1] >= 5.0, u[1] <= 14.0   # Thrust bounds
            ]

            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3)
            # self.logger.info(f"Optimization status: {prob.status}")
            # self.logger.info(f"QP control solution: {u.value if u.value is not None else 'No solution'}")
            u_out = u.value
            res = Lf_v + Lg_v @ u_out + self.gamma * vf
            self.logger.info("result: {:.3f}".format(res))
            # self.logger.info(f"Lf_v: {Lf_v}, Lg_v: {Lg_v}, Constraint: {Lf_v + Lg_v @ u.value}")
            if prob.status in ['optimal', 'optimal_inaccurate']:
                u_safe = u.value
                u_actual = np.clip(u_safe, self.cbf.umin, self.cbf.umax)
                if not np.isclose(u_safe, u_actual).all():
                    self.logger.info("u_safe: {}".format(u_safe))
                #self.logger.error("it works?")
                return u_actual, True
            else:
                self.logger.warn(f"QP failed with status: {prob.status}")
                return nominal_control, False
        except Exception as e:
            self.logger.error(f"Safety filter error: {str(e)}")
            return nominal_control, False
        # return nominal_control, True

class CBF(Node):
    def __init__(self):
        super().__init__('CBF')

        self.cbf_state = None
        self.cbf_model = ExtendedCBF()  # Using ExtendedCBF
        self.safety_filter = SafetyFilter(self.cbf_model, self.get_logger())

        self.create_subscription(Float64MultiArray, 'cf_interface/state', self.state_callback, 10)
        self.create_subscription(Float64MultiArray, 'cf_interface/control', self.publish_cbf_control, 10)
        self.cbf_control_pub = self.create_publisher(Float64MultiArray, 'cbf/control', 10)

    def state_callback(self, msg):
        '''
        self.cbf_state = np.array(msg.data)
        self.get_logger().info(f"CBF state: {self.cbf_state}")
        '''
        self.cbf_state = np.array(msg.data)  # Store the state
        euler_angles = rowan.to_euler(([self.cbf_state[9], self.cbf_state[6], self.cbf_state[7], self.cbf_state[8]]), "xyz")
        yaw = euler_angles[2]
        near_hover_state = np.concatenate([self.cbf_state[0:6], np.array([yaw])])
        #self.get_logger().info(f"Received state: {self.state}")
        #self.get_logger().info(f"Received near hover state: {near_hover_state}")

        state_safety_idis = [1, 2, 4, 5]
        self.only_state_safety_output = near_hover_state[state_safety_idis]
        #self.get_logger().info(f"Received only state safety output: {self.only_state_safety_output}")
        #self.get_logger().info("Received state:")


    def publish_cbf_control(self, msg):
        if self.cbf_state is None:
            #self.get_logger().warn("No state received yet, passing through control")
            self.cbf_control_pub.publish(msg)
            return
        control_safety_idis = [0, 3]
        u_nominal = np.array(msg.data)
        u_nominal_safety = u_nominal[control_safety_idis]
        # self.get_logger().info(f"Nominal control: {u_nominal}")

        u_safe, success = self.safety_filter(self.only_state_safety_output, u_nominal_safety)

        safe_msg = Float64MultiArray()
        u_nominal[control_safety_idis] = u_safe
        safe_msg.data = u_nominal.tolist()
        self.cbf_control_pub.publish(safe_msg)

def main():
    rclpy.init()
    cbf = CBF()
    rclpy.spin(cbf)
    cbf.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
'''
#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from crazyflie_interface.srv import Command
from crazyflie_interfaces.srv import Takeoff, Land, NotifySetpointsStop
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from example_interfaces.msg import Float64MultiArray
import rowan
from crazyflie_interface_py.template_controller import TemplateController


class CBF_Node(Node):
    def __init__(self, node_name = 'CBF'):  
        super().__init__(node_name)
        self.state = None 
        self.cbf_cf_pub = self.create_publisher(Float64MultiArray, 'cbf/control', 1)
        self.cf_cbf_sub = self.create_subscription(Float64MultiArray, 'cf_interface/state', self.callback_state, 1)
        self.lqr_cf_sub = self.create_subscription(Float64MultiArray, 'cf_interface/control', self.callback_control, 1)
        

    def callback_control(self, msg):
        self.lqr_info(msg)

    def callback_state(self, msg):
        self.cf_info(msg)

    def cbf_control_publish(self, control_data): #change to control_data
        control_msg = Float64MultiArray()
        control_msg.data = control_data
        self.cbf_cf_pub.publish(control_msg)

    def lqr_info(self,msg): 
        lqr_output = msg.data
        self.get_logger().info(f'Received from LQR : {lqr_output}')
        self.cbf_control_publish(lqr_output)

    def cf_info(self,msg): 
        cf_output = np.array(msg.data)
        self.state = cf_output
        self.get_logger().info(f'Received from cf interface : {cf_output}')

def main(args=None):
    rclpy.init(args=args)
    cbf_node = CBF_Node()
    rclpy.spin(cbf_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

'''
'''
## CODE2p (CONSTANT)
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp
import matplotlib.pyplot as plt

## CODE3p
class DoubleInt: # this is where the euler approximation of the state is computed. 
    ndim = 4 # (VARIABLE)
    control_dim = 2     # (VARIABLE)
    def __init__(self):
        self.umin = np.array([-1.0, 5.0])  # (VARIABLE)
        self.umax = np.array([1.0, 14.0])    # (VARIABLE)
        self.dt = 0.01  # Time step    
    
    def step(self, state, control):
        print("Control dimensions inside step:", control.shape)
        return state + self.dt * self(state, control)
        print("State dimensions after dynamics computation (step):", result.shape)
    
    def __call__(self, state, control):
        print("State dimensions in __call__:", state.shape)
        print("Control dimensions in __call__:", control.shape)
        result = self.open_loop_dynamics(state) + self.control_jacobian(state) @ control  # (VARIABLE)
        print("Resulting dimensions in __call__:", result.shape)
        return result

    def open_loop_dynamics(self, state):
        out = np.zeros_like(state)
        out[..., 0] = state[..., 2]  # (VARIABLE)
        out[..., 1] = state[..., 3]  # (VARIABLE)
        out[..., 3] = -9.81
        print("Open-loop dynamics output dimensions:", out.shape)
        return out

    def control_jacobian(self, state):
        out = np.zeros((*state.shape[:-1], self.ndim, self.control_dim))
        out[..., 2, 0] = 9.81  # (VARIABLE) Roll affects y_dot 
        out[..., 3, 1] = 1.0  # (VARIABLE) Thrust affects z_dot
        print("Control Jacobian shape:", out.shape)
        return out
    
    ## CODE4p (UPDATED)
class SimpleCBF:  # Safety function is defined here
    def __init__(self, dynamics):
        self.dynamics = dynamics
    
    def vf(self, state):
        state = jnp.array(state)
        # Circular safety boundary
        circle_constraint = 5 - jnp.sqrt(state[0]**2 + state[1]**2)
        # Square boundary constraints
        square_constraint_y = 5 - jnp.abs(state[0])  # y-bound
        square_constraint_z = 5 - jnp.abs(state[1])  # z-bound
        # Combine both constraints
        return jnp.minimum(circle_constraint, jnp.minimum(square_constraint_y, square_constraint_z))
    
    def grad_vf(self, state):
        state = jnp.array(state)
        # Check for near-zero norm to avoid numerical issues
        if np.isclose(state[0]**2 + state[1]**2, 0.0):
            return jnp.zeros(4)
        return jax.grad(self.vf)(state)


class ExtendedCBF(SimpleCBF):  # Extends the safety function to include velocity
    def vf(self, state):
        scaling_factor = 0.72
        state = jnp.array(state)
        # Circular safety boundary with velocity
        circle_constraint = 5 - scaling_factor*(jnp.sqrt(state[0]**2 + state[1]**2)) - scaling_factor*(jnp.sqrt(state[2]**2 + state[3]**2))
        # Square boundary constraints
        square_constraint_y = 5 - jnp.abs(state[0])  # y-bound
        square_constraint_z = 5 - jnp.abs(state[1])  # z-bound
        # Combine both constraints
        return jnp.minimum(circle_constraint, jnp.minimum(square_constraint_y, square_constraint_z))
    
    ## CODE5p (CONSTANT)
class SafetyFilter:
    def __init__(self, cbf):
        self.cbf = cbf
        self.dynamics = self.cbf.dynamics
        self.gamma = 0.6  # Even gentler gamma for smoother approach to wall
        
    def __call__(self, state, nominal_control):
        vf = self.cbf.vf(state)
        grad_vf = self.cbf.grad_vf(state)
        Lg_v = grad_vf @ self.dynamics.control_jacobian(state)
        Lf_v = grad_vf @ self.dynamics.open_loop_dynamics(state)
        
        u = cp.Variable(nominal_control.shape)
        # Adjust weights to prioritize position convergence
        weights = np.diag([12.0, 1.0])  # Less aggressive on thrust, more on pitchNah wha
        obj = cp.Minimize(cp.quad_form(u - nominal_control, weights))
        constraints = [Lf_v + Lg_v @ u >= -self.gamma * vf]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3)
        if u.value is not None:
            return u.value, True
        else:
            return nominal_control, False
        

    ## CODE6p (CONSTANT)
## This is where we call on the previously defined classes to initialize
dynamics = DoubleInt()
cbf = SimpleCBF(dynamics)
safety_filter = SafetyFilter(cbf)

## CODE7p (VARIABLE)
## This is where we use the tager goal with the safety function, a velocity damping, and a clipping funciton 
## to ensure velocity doesn't exceed limitation
nominal_control = lambda state: np.clip(
    np.array([
        - 1.0 * (state[0] - 6.0) - 2.0 * state[2],  # Kept original y-target (6.0), adjusted gains
        9.81 - 0.75 * (state[1] - 6.0) - 1.5 * state[3]  # Kept original z-target (6.0), increased damping
    ]),
    dynamics.umin,
    dynamics.umax
)

## CODE8p (CONSTANT)
## It initializes the starting state to be zero with a vector of same size as predetermined state vector. Calibration. 
## Makes sure that code8 and code10 iterates from a proper initial state. 
starting_state = np.zeros((dynamics.ndim))

## CODE14p
## Initializes for the Smarter CBF 
dynamics = DoubleInt()
cbf = ExtendedCBF(dynamics)
safety_filter = SafetyFilter(cbf)

## CODE15p
## Essentially same thing as CODE8 and COE10 but now just filters the nominal control using the Smart CBF
states = []
vfs = []
controls = []
qp_feasibles = []
state = starting_state
for t in range(1000):
    states.append(state)
    control, qp_feasible = safety_filter(state, nominal_control(state))
    controls.append(control)
    qp_feasibles.append(qp_feasible)
    vf = cbf.vf(state)
    vfs.append(vf)
    state = dynamics.step(state, control)
states = np.array(states)
vfs = np.array(vfs)
controls = np.array(controls)
'''