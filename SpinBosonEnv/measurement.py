from .Basis import Map2Xcontraction


def measurement_TEBD_observables(eng, data=None):

    if data is None:
        keys = ["t", "E", "Sz", "trunc_err"]
        data = dict([(k, []) for k in keys])

    psi = eng.psi
    model = eng.model

    if "t" in data:
        data["t"].append(eng.evolved_time)
    if "trunc_error" in data:
        data["trunc_error"].append(eng.trunc_err.eps)
    if "E" in data:
        data["E"].append(model.H_MPO.expectation_value(psi))
    if "S_bond" in data:
        data["S_bond"].append(psi.entanglement_entropy(n=1)[0])
    if "parity" in data:
        data["parity"].append(model.calc_mps_parity(psi))
    if "Sx" in data:
        data["Sx"].append(psi.expectation_value("Sx", model.atpos_idx))
    if "Sy" in data:
        data["Sy"].append(psi.expectation_value("Sy", model.atpos_idx))
    if "Sz" in data:
        data["Sz"].append(psi.expectation_value("Sz", model.atpos_idx))
        
    return data
    
def measurement_TEBD_field(eng, data=None, basis=None):
    if data is None:
        keys = ["t", "N"]
        data = dict([(k, []) for k in keys])

    psi = eng.psi
    model = eng.model

    if "t" in data:
        data["t"].append(eng.evolved_time)
    if "N" in data:
        data["N"].append(psi.expectation_value(model.OperatorChain("N"), model.bs_idx))
    if "Nmap" in data:
        data["Nmap"].append(psi.expectation_value(model.OperatorChain("N"), model.bs_idx))
    if "Nx" in data:
        assert basis is not None, f"Nmap basis needed to perform the basis change MAP --> X"
        C = psi.correlation_function(
            model.OperatorChain("Bd"), model.OperatorChain("B"), model.bs_idx, model.bs_idx
        )
        Nx = Map2Xcontraction(C, basis)
        data["Nx"].append(Nx)

    return data