import numpy as np
import scipy.sparse as sp


class GeneralSpinBosonEnv:
    def __init__(self, SB_params):
        #
        # k = The momenta
        # wk = Distribution of mode frequencies (degeneracies allowed)
        # gk = Coupling of the system to each mode
        #

        """PARAMETROS DEL ENTORNO SPIN-BOSON"""
        self.k, self.wk, self.gk = self.SubSubOhmicModel(
            SB_params["Nk"],
            SB_params["wc"],
            SB_params["w0"],
            SB_params["g"],
        )

        """ HAMILTONIANO MAPPING """
        # Mapping of the photonic Hamiltonian
        H, self.basis = self.Hmapping(self.wk, self.gk)
        Nmap = H.shape[0]

        # Create the full mapped Hamiltonian
        Hmap = np.zeros((Nmap + 1, Nmap + 1))
        Hmap[1:, 1:] = H.toarray()[:, :]
        Hmap[0, 0] = SB_params["delta"]
        Hmap[0, 1] = np.linalg.norm(self.gk)
        Hmap[1, 0] = np.linalg.norm(self.gk)

        self.wlist = np.diagonal(Hmap[1:, 1:], offset=0)
        self.Jlist = np.diagonal(Hmap[1:, 1:], offset=1)

        # print(self.wlist,'\n\n')
        # print(self.Jlist,'\n\n')

        self.Hmap = Hmap
        # print('Dimension del mapping: ', Nmap)
        self.Nmap = Nmap

        """ HAMILTONIANO MOMENTOS"""
        Hk = np.zeros((len(self.wk) + 1, len(self.wk) + 1))
        Hk[1:, 1:] = np.diag(self.wk)
        Hk[0, 0] = SB_params["delta"]
        Hk[0, 1:] = self.gk
        Hk[1:, 0] = self.gk

        self.Hk = Hk

    def SubSubOhmicModel(self, Nk, wc, w0, g):

        k = np.linspace(-np.pi, np.pi, Nk)
        wk = w0 * wc / np.sqrt(w0**2 + 2 * wc**2 * (1 - np.cos(k)))
        gk = np.array([g / np.sqrt(Nk)])
        gk = np.broadcast_to(gk, (Nk,))

        return (k, wk, gk)

    def Hmapping(self, wks, gks, tol=1e-10):
        """
        Function that transforms a Spin-Boson respresentation of a single arbitrary system coupled to a bosonic bath
        into a tight-binding Hamiltonian where the first site is coupled to the external system.

        The coupling of the first site is given by the norm of the mode coupling vectors.

        Inputs:

        wks: Hamiltonian of the bare photons, allows for diagonal (dispersion relation) or non-diagonal entries.
        gks: A list of each mode coupling gk. Needs to be ordered like the dispersion relation wks.
        tol: Tolerance of the method, we will stop introducing vectors in the new basis with norm < tol.

        Output:
        H: Transformed Hamiltonian of the bare photons
        """

        if len(wks.shape) == 1:
            # If the input is an iterable with the dispersion relation (Hph diagonal) we construct its corresponding matrix
            Hph = np.diag(wks)
        else:
            # Otherwise we take the incoming matrix
            Hph = wks

        L = len(gks)

        # The first vector of our basis is given by the normalized mode couplings.
        # This ensures that we obtain the expected coupling only at cavity 0.
        g0 = np.linalg.norm(gks)
        print("g0", g0)
        self.g0 = g0
        v0 = gks / g0

        basis = [v0]

        for i in range(0, L - 1):
            # For each vector of the basis v_i (starting from behind so that the coupling is at position 0)
            v = basis[len(basis) - 1 - i]

            # Because we are looking for a basis that turns our matrix into a tridiagonal one we want:
            # Hv_i = beta_{i-1}v_{i-1} + alpha_i v_i + beta_{i+1}v_{i+1}

            u = Hph @ v

            # Theoretically we could just say u_i = H·v_i - beta_{i-1}v_{i-1}  = alpha_i v_i + beta_{i+1} v_{i+1}
            # Then we would take out the linear dependence on v_i by substracting (u_i·v_i)*v_i

            # Numerically we have to check that it is orthogonal not only to v_i but to the entirety of the basis

            # u_i will be our next basis vector which is constructed from H·v_i minus its projections
            # to the computed basis v_i

            for v0 in basis:
                # substract all projections to the rest of the basis
                u = u - np.dot(v0, u) * v0

            # Now u_i is orthonormal to all {v_{j=<i}} we just need to normalize it
            norm = np.linalg.norm(u)

            if norm < tol:
                # if the norm is too small means that w can be constructed as a linear combination
                # of all other vectors in the basis.
                # Therefore the basis is enough to reproduce Hph and we stop the computation
                # print('break since norm = ', norm)
                break

            # The vector u is normalized and included in the basis
            u = u / norm

            basis = [u] + basis

        # The columns of 'basis' are the modes 'b_k' of the new model
        basis = np.array(list(reversed(basis))).T
        # We diagonalize the Hamiltonian on the projected subspace
        H = basis.T.conjugate() @ (Hph @ basis)
        # We extract the only relevant items, neglecting small values
        H = sum(sp.diags([np.diag(H, i)], [i]) for i in [-1, 0, 1])
        # print(H.toarray()[0:5,0:5])
        # print(f'Lanczos basis {basis.shape}')
        return H, basis
