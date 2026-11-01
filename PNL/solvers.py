import os
import shutil
from pathlib import Path

import pyomo.environ as pyo


def _localizar_executavel(nome, caminhos_relativos=()):
    """Localiza um solver no PATH ou na pasta local de solvers do Windows."""
    executavel = shutil.which(nome)
    if executavel:
        return Path(executavel)

    raiz = Path(os.environ.get("FPO_SOLVERS_DIR", r"C:\Solvers"))
    for caminho in caminhos_relativos:
        candidato = raiz / caminho
        if candidato.is_file():
            return candidato

    if raiz.is_dir():
        encontrados = sorted(raiz.rglob(nome))
        if encontrados:
            return encontrados[0]

    return None


def localizar_ipopt():
    executavel = _localizar_executavel(
        "ipopt.exe",
        (Path("Ipopt") / "bin" / "ipopt.exe",),
    )
    if executavel:
        return executavel

    try:
        import idaes

        executavel = Path(idaes.bin_directory) / "ipopt.exe"
        if executavel.exists():
            return executavel
    except ImportError:
        pass

    raise FileNotFoundError("IPOPT nao foi encontrado.")


def configurar_ambiente():
    ipopt = localizar_ipopt()
    glpk = _localizar_executavel(
        "glpsol.exe",
        (
            Path("Glpk") / "w64" / "glpsol.exe",
            Path("Glpk") / "w32" / "glpsol.exe",
        ),
    )
    if glpk is None:
        raise FileNotFoundError(
            "GLPK nao foi encontrado no PATH nem em C:\\Solvers."
        )

    pastas = [str(ipopt.parent), str(glpk.parent)]
    os.environ["PATH"] = os.pathsep.join(pastas + [os.environ.get("PATH", "")])

    if not pyo.SolverFactory("ipopt").available(False):
        raise RuntimeError("O Pyomo nao conseguiu acessar o IPOPT.")
    if not pyo.SolverFactory("glpk").available(False):
        raise RuntimeError("O Pyomo nao conseguiu acessar o GLPK.")
    if pyo.SolverFactory("appsi_highs").available(False):
        mip = "appsi_highs"
    else:
        mip = "glpk"
    os.environ["FPO_MIP_SOLVER"] = mip
    if not pyo.SolverFactory("mindtpy").available(False):
        raise RuntimeError("O MindtPy nao esta disponivel no Pyomo.")

    return {"ipopt": str(ipopt), "glpk": str(glpk), "mip": mip}

