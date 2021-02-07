
from main import single_state_prop
import math
import matplotlib.pyplot as plt
import numpy as np

Jmax = 40
N = (Jmax+1)**2
I, dalpha = (5.3901e+05, 34.35)
psauconv = 2.418884*10 ** (-5)
E0, tau = (0.05, 0.02/psauconv)
P = 0.25*dalpha*math.sqrt(math.pi/math.log(16))*E0**2*tau

tinit, tmax, tsteps = (0, 12, 200)
J0, M0 = (0, 0)
psauconv = 2.418884*10 ** (-5)
time = np.linspace(tinit, tmax, num=200)/psauconv
signal = single_state_prop(Jmax, J0, M0, I, P, tinit, tmax, tsteps)

plt.plot(time*psauconv, signal)
plt.show()
