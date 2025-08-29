## global variables
import numpy as np

n = 8  ## system states
m = 3 + 4  ## input position and quaternion (0,0,0,1)
data_length = 20
duration = 120.0
## controller gains and limits
################### Position controller
P_kp = 1.0
P_kd = 0.0
v_max = 0.25

## room dimensions
x_max = 0.9
x_min = -0.6
y_max = 0.9
y_min = -0.6
z_max = 1.5
z_min = 0.0


## quad
R_quad = 0.2

## static obs
R_obs = np.array([0.2,0.2])
num_obs = len(R_obs)
# R_obs = 0.3
X_obs = np.array([[0.5, 0.7, 0.4],
                  [0.3, .0, .5]])




## CBF parameters
a = 1.2
