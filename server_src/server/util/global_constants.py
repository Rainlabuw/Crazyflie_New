## global variables
import numpy as np

n = 8  ## system states
m = 3 + 4  ## input position and quaternion (0,0,0,1)
data_length = 20
duration = 120.0
## controller gains and limits
################### Position controller
P_kp = 0.7
P_kd = 0.0
v_max = 0.5

## room dimensions
x_max = 1
x_min = -1
y_max = 1
y_min = -0.1
z_max = 1.5
z_min = 0.1

## static obstacles
R_obs = 0.5
X_obs = np.array([0.5, 0.5, 0.5])

## CBF parameters
a = 1.2
