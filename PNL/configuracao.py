import importlib
import os


SISTEMAS = {
    "B57": "config_b57",
    "B118": "config_b118",
    "B57_TAP_FIXO": "config_b57_tap_fixo",
    "B118_TAP_FIXO": "config_b118_tap_fixo",
}


def carregar_config(sistema=None):
    nome = str(sistema or os.environ.get("FPO_SISTEMA", "B9")).upper()
    if nome not in SISTEMAS:
        opcoes = ", ".join(SISTEMAS)
        raise ValueError(f"Sistema {nome!r} invalido. Opcoes: {opcoes}.")
    return importlib.import_module(SISTEMAS[nome])
