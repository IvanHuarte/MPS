import matplotlib.pyplot as plt
import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.models.lattice import IrregularLattice, Lattice
from tenpy.models.model import CouplingModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import BosonSite, GroupedSite, SpinSite, set_common_charges


class CavityArrayAtom:

    def __init__(self, model_params, DMRG_options):
        """PARAMETROS"""
        self.GS = model_params["GS"]
        RWA = model_params["RWA"]
        self.L = model_params["L"]
        self.w, self.delta, self.g, self.J = (
            model_params["w"],
            model_params["delta"],
            model_params["g"],
            model_params["J"],
        )
        self.Q, self.atpos_lat, self.bc_mps, self.bc = (
            model_params["Q"],
            model_params["atpos_lat"],
            model_params["bc_mps"],
            model_params["bc"],
        )
        if self.GS:
            self.L -= self.Q

        """SITES"""
        self.bs = BosonSite(Nmax=model_params["N_max"])
        self.sp = SpinSite(S=model_params["S_max"], conserve="parity", sort_charge=True)
        self.add_projectors(self.bs)
        self.add_projectors(self.sp)
        set_common_charges([self.bs, self.sp], new_charges=[])
        if self.GS:
            self.gs = GroupedSite(
                [self.bs, self.sp], labels=None, charges="independent"
            )

        """ADD OPERATORS"""
        if self.GS:
            self.gs.add_op("RW", self.gs.multiply_operators([self.gs.B0, self.gs.Sp1]))
            self.gs.add_op(
                "RWd", self.gs.multiply_operators([self.gs.Bd0, self.gs.Sm1])
            )
            self.gs.add_op("CR", self.gs.multiply_operators([self.gs.B0, self.gs.Sm1]))
            self.gs.add_op(
                "CRd", self.gs.multiply_operators([self.gs.Bd0, self.gs.Sp1])
            )

        """LATTICE"""
        self.lat = Lattice(
            [self.L], unit_cell=[self.bs], bc_MPS=self.bc_mps, bc=self.bc
        )
        if self.GS:
            self.lat = IrregularLattice(
                self.lat,
                add=([at for at in self.atpos_lat], [None for i in range(self.Q)]),
                add_unit_cell=[self.gs],
            )
        else:
            self.lat = IrregularLattice(
                self.lat,
                add=([at for at in self.atpos_lat], [None for i in range(self.Q)]),
                add_unit_cell=[self.sp],
            )
        self.lat.unit_cell_positions = np.array([[0.0], [-0.5]])
        self.lat.pairs["nearest_neighbors"] = [
            (0, 0, np.array([1])),
            (0, 1, np.array([0])),
        ]

        """ ORDERING """
        self.lat.order = self.ordenMPS()
        self.atpos_idx = []
        self.bs_idx = [i for i in range(self.L + self.Q)]
        if self.GS:
            for i, site in enumerate(self.lat.mps_sites()):
                if site == self.gs:
                    self.atpos_idx.append(i)
        else:
            for i, site in enumerate(self.lat.mps_sites()):
                if site == self.sp:
                    self.atpos_idx.append(i)
                    self.bs_idx.remove(i)

        """print('SITES: ','\n',self.lat.mps_sites(),'\n\n')
        orderlat=self.lat.order
        ordermps=[self.lat.lat2mps_idx(j) for j in orderlat]
        print('Orden de MPS en lat_idx: ',orderlat,'\n\n')
        print('Orden de MPS en MPS_idx: ',ordermps,'\n\n')
        """

        """COUPLINGS"""

        self.Cm = CouplingModel(self.lat)
        self.coupling_config(RWA)

        # print('Coupling Terms: \n\n ',Caa.Cm.all_coupling_terms().to_TermList(),'\n\n')
        # print('Onsite Terms: \n\n ',Caa.Cm.all_onsite_terms().to_TermList(),'\n\n')
        """BONDS AND MPO"""

        self.H_MPO = self.Cm.calc_H_MPO(tol_zero=1e-10)
        if not self.GS:
            self.H_bond = self.Cm.calc_H_bond(tol_zero=1e-10)

        """ GROUND STATE AND ENERGY """

        # self.E_gr, self.psi_gr = self.GroundState(options=DMRG_options)
        # print(
        #     "Ground state energy (self.E_gr) and MPS (self.psi_gr) has been calculated \n"
        # )
        # print("E_gr: ", self.E_gr, " \n")

        """ LATTICE PLOT """

        if model_params["plot_lattice"]:
            plt.figure(figsize=(20, 10))
            ax = plt.gca()
            self.lat.plot_coupling(ax, lw=2)
            self.lat.plot_order(ax, linestyle=":", color="r", lw=2)
            self.lat.plot_sites(ax)
            self.lat.plot_basis(ax, origin=(-1, -1))
            ax.set_aspect("equal")
            # Nax.set_xlim(-2,L+1)
            # ax.set_ylim(-2,2)

    def bond_energies(self, psi):
        if self.lat.bc_MPS == "infinite":
            return psi.expectation_value(
                self.H_bond, axes=(["p0", "p1"], ["p0*", "p1*"])
            )

        return psi.expectation_value(
            self.H_bond[1:], axes=(["p0", "p1"], ["p0*", "p1*"])
        )

    def ordenMPS(self):
        orden = [[i, 0] for i in range(self.L)]
        if self.atpos_lat[0][0] == 0:
            orden.insert(0, self.atpos_lat[0])
        else:
            orden.insert(self.atpos_lat[0][0] + 1, self.atpos_lat[0])

        if len(self.atpos_lat) > 1:
            if self.atpos_lat[-1][0] == self.L - 1:
                orden.append(self.atpos_lat[-1])
            else:
                orden.insert(self.atpos_lat[-1][0] + 2, self.atpos_lat[-1])

        for i in range(1, self.Q - 1):
            orden.insert(self.atpos_lat[i][0] + i + 1, self.atpos_lat[i])

        return np.array(orden)  # .reshape(self.L+self.Q,2)

    def coupling_config(self, RWA):
        if self.GS:
            N = self.OperatorChain("N")
            B = self.OperatorChain("B")
            Bd = self.OperatorChain("Bd")

            for i in range(self.L):  # CAVITY-PHOTON
                self.Cm.add_local_term(self.w[i], [(N[i], self.lat.order[i])])
                ### HOPPING ###
            for i in range(self.L - 1):
                # print(i)
                self.Cm.add_local_term(
                    -self.J[i],
                    [(B[i], self.lat.order[i]), (Bd[i + 1], self.lat.order[i + 1])],
                    plus_hc=True,
                )  # horizontal #

            for at in self.atpos_lat:
                self.Cm.add_local_term(self.delta, [("Sz1", at)])  # CAVITY-ATOM
                self.Cm.add_local_term(self.g, [("RW", at)])  # RW
                self.Cm.add_local_term(self.g, [("RWd", at)])
                if not RWA:
                    self.Cm.add_local_term(self.g, [("CR", at)])  # CRW
                    self.Cm.add_local_term(self.g, [("CRd", at)])
        else:
            # Cavity Energy
            for i in range(self.L):
                self.Cm.add_local_term(self.w[i], [("N", [i, 0])])

            # HOPPING
            for i in range(self.L - 1):
                self.Cm.add_local_term(
                    -self.J[i], [("B", [i, 0]), ("Bd", [i + 1, 0])], plus_hc=True
                )
            # CAVITY-ATOM
            for at in self.atpos_lat:
                # Atom Energy
                self.Cm.add_local_term(self.delta/2, [("Sz", at)])
                # Atom coupling
                self.Cm.add_local_term(
                    self.g, [("B", [at[0], 0]), ("Sp", at)], plus_hc=True
                )  # RW
                if not RWA:
                    self.Cm.add_local_term(
                        self.g, [("B", [at[0], 0]), ("Sm", at)], plus_hc=True
                    )  # CRW

    def InitialState(self, config, GS=False, adv_config=False, show_pstate=False):
        """
        Set the initial wavefunction in all lattice. 2 modes:

        (adv_config=False)(default) Sets all BosonSite's to |0> ([1,0,...]) and config=[[spin1_state],[spin2_state],...] specifies
        the local initial state of the SpinSite's in the lattice.

        (adv_config=True) Free wavefunction settings. Sets all sites to their fundamental state (|0> for BosonSite, |-S> for SpinSite)
        and config==[ [[lat_idx_1],[local_state_1]] , [[lat_idx_2],[local_state_2]], ....], i.e., each element specifies first the lattice
        index ([i,j,u]) of the site, and then sets the local initial state in it.

        Also, implemented a GroupedSite option GS=True

        """
        SN = self.L + self.Q
        bosdim, spindim = self.bs.dim, self.sp.dim
        boson = [[0 for i in range(bosdim)]]
        boson[0][0] = 1
        spin = [[0 for i in range(spindim)]]
        if GS:
            spin = [0 for i in range(spindim * bosdim)]
        spin[0][0] = 1
        """if (len(config)!=0 and GS):
            spin=config[0]+[0]*(spindim*(bosdim-1))"""

        p = boson * SN
        for sp in self.atpos_idx:
            p[sp] = spin[0]

        if adv_config:
            for conf in config:
                mps_idx = self.lat.lat2mps_idx(conf[0])
                p[mps_idx] = conf[1]

        else:
            for at, conf in zip(self.atpos_idx, config):
                p[at] = conf

        if show_pstate:
            print("Initial State has been generated\n")
            print("p_state: ", p, "\n\n")

        # print(f"Initial state:\n {p}")
        # print(f"Sites: {len(p)}")

        if len(config) == 0:
            return MPS.from_product_state(
                self.lat.mps_sites(), p, bc=self.lat.bc_MPS, permute=False
            )
        else:
            self.psi0 = MPS.from_product_state(
                self.lat.mps_sites(), p, bc=self.lat.bc_MPS, permute=False
            )

    def GroundState(self, options):
        initial_state = self.InitialState(config=[], GS=self.GS)
        eng = TwoSiteDMRGEngine(initial_state, self, options)

        return eng.run()

    def measurement(self, eng, data):

        if data is None:
            keys = ["t", "EXC", "N", "trunc_err"]
            data = dict([(k, []) for k in keys])

        data["t"].append(eng.evolved_time)
        Pes = eng.psi.expectation_value("Sz1", self.atpos_idx) + 0.5
        data["EXC"].append(Pes)
        data["N"].append(eng.psi.expectation_value(self.OperatorChain("N")))

        data["trunc_err"].append(eng.trunc_err.eps)

        print("t: ", eng.evolved_time)
        print(Pes, "\n")

        return data

    def measure_projector(self, eng, data, field=False):
        if data is None:
            keys = ["t", "EXC", "N"]
            data = dict([(k, []) for k in keys])
            data["N"] = [[] for i in range(self.bs.dim)]
            data["EXC"] = [[] for i in range(self.sp.dim)]
            print(type(data["EXC"]))

        if self.GS:
            for i in range(self.sp.dim):
                data["EXC"][i].append(
                    eng.psi.expectation_value(
                        ["P" + str(i) + "1"] * self.Q, self.atpos_idx
                    )[0]
                )
            if field:
                for i in range(self.bs.dim):
                    data["N"][i].append(
                        eng.psi.expectation_value(
                            self.OperatorChain("P" + str(i)), self.bs_idx
                        )
                    )
        else:
            for i in range(self.sp.dim):
                data["EXC"][i].append(
                    eng.psi.expectation_value(["P" + str(i)] * self.Q, self.atpos_idx)[
                        0
                    ]
                )
            if field:
                for i in range(self.bs.dim):
                    data["N"][i].append(
                        eng.psi.expectation_value("P" + str(i), self.bs_idx)
                    )
        return data

    def OperatorChain(self, op):
        op0 = op + "0"
        opchain = [op] * (self.L + self.Q)
        for at in self.atpos_idx:
            opchain[at] = op0
        return opchain

    def add_projectors(self, site):
        dim = site.dim
        op = "P"
        for i in range(dim):
            p = np.zeros([dim])
            p[i] = 1.0
            proj = npc.diag(p, site.leg, labels=["p", "p*"])
            site.add_op(op + str(i), proj)

    """def state_probability(psi,ops):
        
        psi.apply_local_op(0, proj, unitary=True)
        S=psi0.expectation_value('Sz',0)[0]
        np.abs(S)"""
