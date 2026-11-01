import math

import numpy as np
import pandas as pd


def carregar_sistema(arquivo, config):
    dbar = pd.read_excel(arquivo, sheet_name="DBAR")
    dlin = pd.read_excel(arquivo, sheet_name="DLIN")
    dger_raw = pd.read_excel(arquivo, sheet_name="DGER")

    barras = dbar["Barra"].astype(int).to_numpy()
    mapa = {barra: indice for indice, barra in enumerate(barras)}

    dbar["Barra_original"] = barras
    dbar["Barra"] = np.arange(len(dbar))
    dbar["Vm"] = dbar["Vm"].astype(float) / 1000.0
    dbar["Va"] = np.radians(dbar["Va"].astype(float))
    dbar["Pd"] = dbar["Pd"].astype(float)
    if hasattr(config, "DEMANDA_REATIVA_MVAR"):
        dbar["Qd"] = dbar["Barra_original"].map(
            config.DEMANDA_REATIVA_MVAR
        ).fillna(0.0)
    else:
        dbar["Qd"] = dbar["Pd"] * math.tan(
            math.acos(config.FATOR_POTENCIA_CARGA)
        )
    dbar["Sh_MVAr"] = dbar["Barra_original"].map(
        getattr(config, "SHUNTS_BARRAS_MVAR", {})
    ).fillna(0.0)
    dbar["Qg_fixo_MVAr"] = dbar["Barra_original"].map(
        getattr(config, "INJECOES_REATIVAS_FIXAS_MVAR", {})
    ).fillna(0.0)
    for barra, tensao in config.TENSAO_GERADORES.items():
        dbar.loc[dbar["Barra_original"] == barra, "Vm"] = tensao

    dlin["De_original"] = dlin["De"].astype(int)
    dlin["Para_original"] = dlin["Para"].astype(int)
    dlin["De"] = dlin["De"].map(mapa).astype(int)
    dlin["Para"] = dlin["Para"].map(mapa).astype(int)
    dlin["R"] = dlin["R"].astype(float) / config.PB
    dlin["X"] = dlin["X"].astype(float) / config.PB
    dlin["Bsh"] = dlin["Bsh"].fillna(0.0).astype(float) / config.PB
    dlin["Tap"] = dlin["Tap"].replace(0, 1).fillna(1).astype(float)
    dlin["TapMin"] = dlin["Tn"].fillna(0.0).astype(float)
    dlin["TapMax"] = dlin["Tm"].fillna(0.0).astype(float)
    dlin["Phs"] = np.radians(dlin["Phs"].fillna(0.0).astype(float))
    dlin["Flim"] = (
        dlin["Flim"].astype(float)
        * float(getattr(config, "FATOR_LIMITE_TERMICO_ESTUDO", 1.0))
    )
    for (de, para), tap in getattr(config, "TAPS_INICIAIS", {}).items():
        linhas = (
            (dlin["De_original"] == de)
            & (dlin["Para_original"] == para)
        )
        dlin.loc[linhas, "Tap"] = float(tap)
    dlin["Barra_controle"] = 0
    dlin["V_controle"] = 0.0
    dlin["Orientacao_controle"] = 0.0
    for (de, para), (barra, tensao, orientacao) in config.CONTROLES_TAP.items():
        linhas = (
            (dlin["De_original"] == de)
            & (dlin["Para_original"] == para)
            & (dlin["TapMin"] > 0.0)
            & (dlin["TapMax"] > 0.0)
        )
        dlin.loc[linhas, "Barra_controle"] = mapa[barra]
        dlin.loc[linhas, "V_controle"] = tensao
        dlin.loc[linhas, "Orientacao_controle"] = orientacao

    dger = dger_raw[dger_raw["Custo"].astype(float) < 100.0].copy()
    dger = dger.reset_index(drop=True)
    dger["Barra_original"] = dger["Barra"].astype(int)
    dger["Barra"] = dger["Barra"].map(mapa).astype(int)
    for coluna in ["Pmin", "Pmax", "Custo"]:
        dger[coluna] = dger[coluna].astype(float)

    dbar_original = dbar.set_index("Barra_original")
    dger["Qmin"] = dger["Barra_original"].map(dbar_original["Qn"]).astype(float)
    dger["Qmax"] = dger["Barra_original"].map(dbar_original["Qm"]).astype(float)
    dger["Pg0"] = dger["Barra_original"].map(dbar_original["Pg"]).astype(float)
    dger["Qg0"] = dger["Barra_original"].map(dbar_original["Qg"]).astype(float)

    # A unidade da barra de referencia representa a interface com a rede a
    # montante. Despacho negativo nessa unidade e exportacao liquida, necessaria
    # para estudar reversao de fluxo com MMGD acima da demanda instantanea.
    if getattr(config, "PERMITIR_EXPORTACAO_REFERENCIA", False):
        barra_referencia = int(
            dbar.loc[dbar["Tipo"].astype(int) == 2, "Barra_original"].iloc[0]
        )
        mascara_referencia = dger["Barra_original"] == barra_referencia
        dger.loc[mascara_referencia, "Pmin"] = -float(dbar["Pd"].sum()) * 2.0

    carga = pd.read_excel(arquivo, sheet_name="Carga")["GERAL"].iloc[:24].to_numpy(float, copy=True)
    solar = pd.read_excel(arquivo, sheet_name="MMGD")["GERAL"].iloc[:24].to_numpy(float, copy=True)
    carga /= carga.max()
    solar /= solar.max()

    return dbar, dlin, dger, carga, solar, mapa


def montar_cenarios_24h(dbar, carga, solar, mapa, config, penetracao):
    rng = np.random.default_rng(config.SEED)
    fator_carga = np.ones(len(dbar))
    barras_carga = np.flatnonzero(dbar["Pd"].to_numpy(float) > 0.0)
    fator_carga[barras_carga] = rng.uniform(
        *config.FAIXA_CARGA, size=len(barras_carga)
    )
    fator_mmgd = rng.uniform(*config.FAIXA_MMGD, size=len(config.BARRAS_MMGD))

    tabela_mmgd = dbar.set_index("Barra_original").loc[
        config.BARRAS_MMGD, "Pd"
    ]
    if not np.all(
        dbar.set_index("Barra_original").loc[config.BARRAS_MMGD, "Tipo"].to_numpy(int)
        == 0
    ):
        raise ValueError("Todas as barras com MMGD/BESS devem ser barras de carga (Tipo=0).")
    carga_local = tabela_mmgd.to_numpy(float)
    if np.any(carga_local <= 0.0):
        raise ValueError("Todas as barras com MMGD/BESS devem possuir carga ativa positiva.")

    # Penetracao = potencia MMGD instalada / demanda ativa maxima do sistema.
    # A capacidade total e distribuida proporcionalmente a carga das barras
    # hospedeiras. Isto evita que uma penetracao de 100% seja interpretada como
    # 100% apenas do pequeno subconjunto de cargas que recebeu MMGD.
    demanda_base_mw = float(dbar["Pd"].sum())
    capacidade_total_mw = demanda_base_mw * float(penetracao) / 100.0
    capacidade = capacidade_total_mw * carga_local / carga_local.sum()
    barras_mmgd = np.array([mapa[barra] for barra in config.BARRAS_MMGD])
    cenarios = []
    for hora in range(24):
        pd = dbar["Pd"].to_numpy(float) * carga[hora] * fator_carga
        qd = (
            dbar["Qd"].to_numpy(float) * carga[hora] * fator_carga
            - dbar["Qg_fixo_MVAr"].to_numpy(float)
        )
        cenarios.append(
            {
                "hora": hora,
                "penetracao": float(penetracao),
                "Pd": pd,
                "Qd": qd,
                "barras_mmgd": barras_mmgd,
                "barras_mmgd_originais": np.array(config.BARRAS_MMGD),
                "capacidade_mmgd": capacidade,
                "disponibilidade_mmgd": capacidade * solar[hora] * fator_mmgd,
                "fator_carga": fator_carga,
                "fator_mmgd": fator_mmgd,
                "curva_carga": carga,
                "curva_solar": solar,
            }
        )
    return cenarios

