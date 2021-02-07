def single_state_prop(Jmax, J0, M0, I, P, tint, tmax, tsteps):

    import math
    import numpy as np
    import scipy.sparse as sparse
    import scipy.linalg as linalg

    def superindex(Jmax):
        """
        Generates a "superindex" matrix v. The first 
        column of v contains all the J's, while the 
        second one all the M's. The order is discussed
        in the pdf document.
        """
        v = np.zeros((1, 2))
        for m in range(-Jmax, Jmax+1):
            Js = np.arange(Jmax, abs(m)-1, -1)
            vtemp = np.repeat(m, Js.size)
            v = np.vstack((v, np.vstack((Js, vtemp)).T))
        return np.delete(v, 0, 0)

    def initialstate(J0, M0, N, v):
        """
        Generates a vector representing
        the initial state defined by J0 and M0.
        """
        psi0 = np.zeros(N)
        index = np.where((v == (0, 0)).all(axis=1))[0]
        psi0[index] = 1
        return psi0

    def costheta(N, v):
        """
        Generates the matrix representation 
        of cos^2(theta).
        """
        LD = np.zeros(N-2)
        DD = np.zeros(N)
        for n in range(0, N):
            J, M = (v[n, 0], v[n, 1])
            DD[n] = 1/3 - (2/3)*(3*M**2-J*(J+1))/((2*J+3)*(2*J-1))
            if v[n, 1] == v[n-2, 1] and v[n, 0] == v[n-2, 0]-2:
                LD[n-2] = np.sqrt(((J+2)**2-M**2)*(J+1)**2-M**2) /\
                    ((2*J+3)*np.sqrt((2*J+5)*(2*J+1)))
        return sparse.diags([LD, DD, LD], [-2, 0, 2], format="csc")

    def intpropagation(N, v, psi0, P, observable, time):
        """
        Using matrix exponential a single kick is applied.
        Then, the wave function is propagated in time.
        """
        energy = (1/(2*I))*v[:, 0]*(v[:, 0]+1)
        interaction = linalg.expm(1j*P*costheta(N, v))
        psiplus = interaction.dot(psi0)
        wfafotime = np.exp(-1j*np.outer(energy, time))*psiplus[..., np.newaxis]
        return np.real(np.multiply(np.conj(wfafotime), observable*wfafotime).sum(axis=0))

    # The basis size
    N = (Jmax+1)**2
    # Conversion factor between atomic time units and ps.
    psauconv = 2.418884*10 ** (-5)
    # Time grid in atomic units
    time = np.linspace(tint, tmax, num=tsteps)/psauconv
    v = superindex(Jmax)
    psi0 = initialstate(J0, M0, N, v)
    # Define the observable.
    observable = costheta(N, v)
    return intpropagation(N, v, psi0, P, observable, time)
