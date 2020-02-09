import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        self.init_x = init_pose[0]
        self.init_y = init_pose[1]
        self.init_z = init_pose[2]

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses time in air, current pose vs target pose to return reward."""
        
        ## time in air, increases with self.sim.time
        if self.sim.time < 2.5:
            tia = self.sim.time ** 2
        else:
            tia = (self.sim.time / 5) * 3
        
        ## x_pos reward
        if self.sim.pose[0] > self.target_pos[0]:
            x_pos = (self.target_pos[0] - self.sim.pose[0]) / 10
        else:
            if self.target_pos[0] - self.init_x == 0:
                x_pos = 0
            else:
                x_pos = self.sim.pose[0] / (self.target_pos[0] - self.init_x)
        
        #increased reward for getting close to the target_pos
        if x_pos > 0.5:
            x_pos += 0.5
        
        if x_pos > 0.9:
            x_pos += 0.9
            
        if x_pos == 1:
            x_pos += 5
        
        ## y_pos reward
        if self.sim.pose[1] > self.target_pos[1]:
            y_pos = (self.target_pos[1] - self.sim.pose[1]) / 10
        else:
            if self.target_pos[1] - self.init_y == 0:
                y_pos = 0
            else:
                y_pos = self.sim.pose[1] / (self.target_pos[1] - self.init_y)
        
        #increased reward for getting close to the target_pos
        if y_pos > 0.5:
            y_pos += 0.5
        
        if y_pos > 0.9:
            y_pos += 0.9       
        
        if y_pos == 1:
            y_pos += 5        
        
        ## z pos reward
        if self.sim.pose[2] > self.target_pos[2]:
            z_pos = (self.target_pos[2] - self.sim.pose[2]) / 10
        else:
            if self.target_pos[2] - self.init_z == 0:
                z_pos = 0
            else:
                z_pos = self.sim.pose[2] / (self.target_pos[2] - self.init_z)
        
        #increased reward for getting close to the target_pos
        if z_pos > 0.5:
            z_pos += 0.5
        
        if z_pos > 0.9:
            z_pos += 0.9        

        if z_pos == 1:
            z_pos += 5
            
        ## su m of distance from target
        dist = x_pos + y_pos + z_pos
        
        ## reward for completing the task
        if x_pos and y_pos and z_pos == 0.95:
            comp = 1000
            
        else:
            comp = 0
        
        ## sum of reward components
        reward = tia + dist + comp
        
        ## punishment components 
        punishment = 0
        
        # not completing the task
        if (self.sim.time < 5.) and (self.sim.done == True):
            punishment += 100
                  
        return reward - punishment

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state