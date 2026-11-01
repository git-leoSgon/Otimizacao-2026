import numpy as np
import pandas as pd


def carregar_sistema(arquivo, PB):
    DBAR = pd.read_excel(arquivo, sheet_name="DBAR")
    DLIN = pd.read_excel(arquivo, sheet_name="DLIN")
    DGER = pd.read_excel(arquivo, sheet_name="DGER")

    custos = DGER["Custo"].to_numpy(dtype=float)
    DDEF = DGER[custos >= 100].copy().reset_index(drop=True)
    DGER = DGER[custos < 100].copy().reset_index(drop=True)

    barras = DBAR["Barra"].astype(int).to_numpy()
    mapa = {barra: i for i, barra in enumerate(barras)}

    DBAR["Barra_original"] = barras
    DBAR["Barra"] = np.arange(len(DBAR))
    DBAR["Pd"] = DBAR["Pd"].astype(float) / PB

    DLIN["De_original"] = DLIN["De"].astype(int)
    DLIN["Para_original"] = DLIN["Para"].astype(int)
    DLIN["De"] = DLIN["De"].map(mapa).astype(int)
    DLIN["Para"] = DLIN["Para"].map(mapa).astype(int)
    DLIN["R"] = DLIN["R"].astype(float) / PB
    DLIN["X"] = DLIN["X"].astype(float) / PB
    DLIN["Flim"] = DLIN["Flim"].astype(float) / PB
    DLIN["Tap"] = DLIN["Tap"].replace(0, 1).fillna(1).astype(float)
    DLIN["Phs"] = DLIN["Phs"].fillna(0).astype(float)

    for tabela in (DGER, DDEF):
        tabela["Barra_original"] = tabela["Barra"].astype(int)
        tabela["Barra"] = tabela["Barra"].map(mapa).astype(int)
        tabela["Pmin"] = tabela["Pmin"].astype(float) / PB
        tabela["Pmax"] = tabela["Pmax"].astype(float) / PB

    return DBAR, DLIN, DGER, DDEF, mapa


def montar_cenario(arquivo, DBAR, mapa, config):
    carga = pd.read_excel(arquivo, sheet_name="Carga")["GERAL"].iloc[:config.DT].to_numpy(dtype=float)
    solar = pd.read_excel(arquivo, sheet_name="MMGD")["GERAL"].iloc[:config.DT].to_numpy(dtype=float)
    carga /= carga.max()
    solar /= solar.max()

    rng = np.random.default_rng(config.SEED)
    fator_carga = np.ones(len(DBAR))
    barras_carga = np.flatnonzero(DBAR["Pd"].to_numpy() > 0)
    fator_carga[barras_carga] = rng.uniform(
        *config.FAIXA_CARGA, size=len(barras_carga)
    )
    fator_mmgd = rng.uniform(
        *config.FAIXA_MMGD, size=len(config.BARRAS_MMGD)
    )

    return {
        "carga": carga,
        "solar": solar,
        "Fcar": carga[:, None] * fator_carga[None, :],
        "Fcap": solar[:, None] * fator_mmgd[None, :],
        "fator_carga": fator_carga,
        "fator_mmgd": fator_mmgd,
        "barras_mmgd": np.array(
            [mapa[barra] for barra in config.BARRAS_MMGD], dtype=int
        ),
    }
