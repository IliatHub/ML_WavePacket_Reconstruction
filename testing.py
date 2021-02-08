
from main import single_state_prop
import math
import matplotlib.pyplot as plt
import numpy as np

Jmax = 30
N = (Jmax+1)**2
I, dalpha = (5.3901e+05, 34.35)
# Conversion factor between atomic time units and ps.
psauconv = 2.418884*10 ** (-5)
E0, tau = (0.037, 0.02/psauconv)
P = 0.25*dalpha*math.sqrt(math.pi/math.log(16))*E0**2*tau
print(P)
# Define time grid in picoseconds
tdef = (0, 120, 500)  # (initial time, final time, number of time steps)
J0M0 = (0, 0)
# Time grid in picosecodns
time = np.linspace(tdef[0], tdef[1], num=tdef[2])
signal1 = single_state_prop(Jmax, J0M0, I, P, tdef)
signal2 = single_state_prop(Jmax+5, J0M0, I, P, tdef)

plt.plot(time, signal1)
plt.plot(time, signal2)
plt.show()
