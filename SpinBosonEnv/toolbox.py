import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

# ---------- Lectura de información ----------


@lru_cache(maxsize=None)
def load_artifact(path):
    """Carga el json una única vez."""
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def get_params(path):
    """Extrae los parámetros del nombre del archivo."""
    stem = Path(path).stem

    params = {k: float(v) for k, v in re.findall(r"(w0|wc|Nk|delta)_([0-9.]+)", stem)}

    uuid = re.search(r"UUID_([a-f0-9]+)", stem)
    if uuid:
        params["UUID"] = uuid.group(1)

    return params


def get_value(path, criterion):
    """
    Devuelve el valor de 'criterion' para un archivo.
    """

    if criterion == "parity":
        pstate = load_artifact(path)["SIM"]["model_params"]["pstate"]

        if pstate == [[1, 0]]:
            return "even"
        elif pstate == [[0, 1]]:
            return "odd"

        raise ValueError(f"Estado inicial desconocido: {pstate}")

    return get_params(path)[criterion]


# ---------- Agrupación ----------


def group_by(paths, criterion):
    """
    Agrupa una lista de archivos según criterion.

    criterion puede ser:
        "wc"
        "parity"
        "delta-parity"
        "w0-wc-Nk"
    """

    groups = defaultdict(list)

    criteria = criterion.split("-")

    for path in paths:

        if len(criteria) == 1:
            key = get_value(path, criteria[0])
        else:
            key = tuple(get_value(path, c) for c in criteria)

        groups[key].append(path)

    try:
        keys = sorted(groups)
    except TypeError:
        keys = groups.keys()

    return {k: groups[k] for k in keys}


# ---------- Clasificación recursiva ----------


def classify(paths, mode):

    if mode == "":
        return paths

    if "/" in mode:
        criterion, remaining = mode.split("/", 1)
    else:
        criterion, remaining = mode, ""

    groups = group_by(paths, criterion)

    return {key: classify(group, remaining) for key, group in groups.items()}


# -------------- Print dictionary --------------


def print_tree(tree, prefix="", values=True):
    for key, val in tree.items():

        if isinstance(val, dict):
            print(prefix + str(key))
            print_tree(val, prefix + "   ", values=values)
        else:
            if values:
                print(prefix + f"{str(key)}: {str(val)}")
            else:
                print(prefix + str(key))


# ------------- PLOTTING -------------- #

param2latex = {
    "w0": r"\omega_0",
    "wc": r"\omega_c",
    "delta": r"\Delta",
    "parity": r"P",
    "Nk": r"N_k"

}

def join_modes(modes, values):

    assert len(modes) == len(values)
    values = [f"{v:g}" if not isinstance(v, str) else v for v in values]
    equals = [rf"{param2latex[m]}\;=\;{v}"  for m, v in zip(modes, values)]
    return r"\;\;".join(equals)

def join_static_modes(static_modes):

    static_modes = [(m, f"{v:g}") if not isinstance(v, str) else (m,v) for m, v in static_modes]
    equals = [rf"${param2latex[m]}\;=\;{v}$"  for m, v in static_modes]
    return "\n".join(equals)
    
