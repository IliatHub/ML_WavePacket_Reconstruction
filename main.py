import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import math


def superindex(Jmax):
    v = np.zeros((1, 2))
    for m in range(-Jmax, Jmax+1):
        Js = np.arange(Jmax, abs(m)-1, -1)
        vtemp = np.repeat(m, Js.size)
        v = np.vstack((v, np.vstack((Js, vtemp)).T))
    return np.delete(v, 0, 0)


def initialstate(J, M, N, v):
    psi0 = np.zeros(N)
    index = np.where((v == (0, 0)).all(axis=1))[0]
    psi0[index] = 1
    return psi0


def costheta(N, v):
    LD = np.zeros(N-2)
    DD = np.zeros(N)
    for n in range(0, N):
        J, M = (v[n, 0], v[n, 1])
        DD[n] = 1/3 - (2/3)*(3*M**2-J*(J+1))/((2*J+3)*(2*J-1))
        if v[n, 1] == v[n-2, 1] and v[n, 0] == v[n-2, 0]-2:
            LD[n-2] = np.sqrt(((J+2)**2-M**2)*(J+1)**2-M**2) /\
                ((2*J+3)*np.sqrt((2*J+5)*(2*J+1)))
    return sparse.diags([LD, DD, LD], [-2, 0, 2], format="csc")


def intpropagation(N, v, psi0, observable):
    energy = (1/(2*I))*v[:, 0]*(v[:, 0]+1)
    interaction = linalg.expm(1j*P*costheta(N, v))
    psiplus = interaction.dot(psi0)
    wfafotime = np.exp(-1j*np.outer(energy, time))*psiplus[..., np.newaxis]
    return np.real(np.multiply(np.conj(wfafotime), observable*wfafotime).sum(axis=0))


Jmax = 20
N = (Jmax+1)**2
psauconv = 2.418884*10 ** (-5)
time = np.linspace(0, 12, num=200)/psauconv
I, dalpha = (5.3901e+05, 34.35)
E0, tau = (0.05, 0.02/(2.418884*10 ** (-5)))
P = 0.25*dalpha*math.sqrt(math.pi/math.log(16))*E0**2*tau

v = superindex(Jmax)
psi0 = initialstate(0, 0, N, v)
observable = costheta(N, v)
signal = intpropagation(N, v, psi0, observable)

plt.plot(time*psauconv, signal)
plt.show()
