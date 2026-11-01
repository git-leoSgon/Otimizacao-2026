from pathlib import Path
import sys

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import config_b9 as config
from config_b9 import (
    ARQUIVO,
    BARRAS_MMGD,
    FRACAO_BESS,
    PB,
    PENETRACOES,
    RESULTADOS_COM_BESS,
)
from DC_OPF.dados import carregar_sistema, montar_cenario
from DC_OPF.fpo_dc_baterias import resolver_com_perdas


def main():
    RESULTADOS_COM_BESS.mkdir(exist_ok=True)

    DBAR, DLIN, DGER, DDEF, mapa = carregar_sistema(ARQUIVO, PB)
    cenario = montar_cenario(ARQUIVO, DBAR, mapa, config)
    carga_local = DBAR.set_index("Barra_original").loc[BARRAS_MMGD, "Pd"].to_numpy() * PB

    solucoes = []
    capacidades_mmgd = []
    capacidades_bess = []

    for penetracao in PENETRACOES:
        pmax_mmgd = carga_local * penetracao / 100.0 / PB
        capacidade_bess = FRACAO_BESS * pmax_mmgd
        solucao = resolver_com_perdas(
            DBAR, DLIN, DGER, DDEF, cenario, pmax_mmgd,
            capacidade_bess, config,
        )
        solucoes.append(solucao)
        capacidades_mmgd.append(pmax_mmgd * PB)
        capacidades_bess.append(capacidade_bess * PB)

        ocupacao = 100.0 * np.max(
            np.maximum(np.abs(solucao["Fd"]), np.abs(solucao["Fc"]))
            / DLIN["Flim"].to_numpy()[:, None]
        )
        print(
            f"MMGD {penetracao:2d}% | {pmax_mmgd.sum() * PB:6.2f} MW | "
            f"baterias {capacidade_bess.sum() * PB:6.2f} MWh | "
            f"perdas {solucao['Ploss'].sum() * PB:7.3f} MWh | linhas {ocupacao:6.2f}%"
        )

    capacidades_mmgd = np.asarray(capacidades_mmgd)
    capacidades_bess = np.asarray(capacidades_bess)
    Pg = np.stack([solucao["Pg"] for solucao in solucoes]) * PB
    Fd = np.stack([solucao["Fd"] for solucao in solucoes])
    Fc = np.stack([solucao["Fc"] for solucao in solucoes])
    Pcut = np.stack([solucao["Pcut"] for solucao in solucoes]) * PB
    Pdef = np.stack([solucao["Pdef"] for solucao in solucoes]) * PB
    Ploss = np.stack([solucao["Ploss"] for solucao in solucoes]) * PB
    Pbin = np.stack([solucao["Pbin"] for solucao in solucoes]) * PB
    Pbout = np.stack([solucao["Pbout"] for solucao in solucoes]) * PB
    Soc = np.stack([solucao["Soc"] for solucao in solucoes])
    Pbin[np.abs(Pbin) < 1e-8] = 0.0
    Pbout[np.abs(Pbout) < 1e-8] = 0.0

    limites = DLIN["Flim"].to_numpy() * PB
    fluxo_pct = 100.0 * np.maximum(np.abs(Fd), np.abs(Fc)) * PB / limites[None, :, None]
    carga_h = np.sum(DBAR["Pd"].to_numpy()[None, :] * cenario["Fcar"], axis=1) * PB
    mmgd_disp = capacidades_mmgd[:, :, None] * cenario["Fcap"].T[None, :, :]
    mmgd_inj = mmgd_disp - Pcut

    barras_texto = ", ".join(str(barra) for barra in BARRAS_MMGD)
    fatores_texto = " | ".join(
        f"B{barra}={fator:.4f}"
        for barra, fator in zip(BARRAS_MMGD, cenario["fator_mmgd"])
    )
    relatorio = [
        f"B9 - MMGD E BATERIAS NAS BARRAS {barras_texto}",
        "",
        f"Fatores de MMGD: {fatores_texto}",
        "",
        "Pen.  MMGD MW  Bat. MWh  MMGD inj.  Ger. conv.  Carga bat.  Desc. bat.  Perdas  Corte  Deficit",
    ]

    for k, penetracao in enumerate(PENETRACOES):
        relatorio.append(
            f"{penetracao:3d}%  {capacidades_mmgd[k].sum():7.2f} "
            f"{capacidades_bess[k].sum():9.2f} {mmgd_inj[k].sum():10.2f} "
            f"{Pg[k].sum():11.2f} {Pbin[k].sum():10.2f} {Pbout[k].sum():10.2f} "
            f"{Ploss[k].sum():7.2f} {Pcut[k].sum():6.2f} {Pdef[k].sum():8.2f}"
        )

    texto = "\n".join(relatorio)
    print("\n" + texto)
    (RESULTADOS_COM_BESS / "B9_MMGD_Baterias_relatorio.txt").write_text(
        texto, encoding="utf-8"
    )

    np.savez_compressed(
        RESULTADOS_COM_BESS / "B9_MMGD_Baterias_resultados.npz",
        penetracoes=np.asarray(PENETRACOES),
        barras_mmgd=np.asarray(BARRAS_MMGD),
        capacidades_mmgd_MW=capacidades_mmgd,
        curva_carga=cenario["carga"],
        curva_solar=cenario["solar"],
        fatores_carga=cenario["fator_carga"],
        fatores_mmgd=cenario["fator_mmgd"],
        carga_h_MW=carga_h,
        mmgd_disponivel_MW=mmgd_disp,
        mmgd_injetada_MW=mmgd_inj,
        geracao_conv_MW=Pg.sum(axis=1),
        despacho_geradores_MW=Pg,
        capacidade_baterias_MWh=capacidades_bess,
        potencia_baterias_max_MW=capacidades_bess,
        carga_baterias_MW=Pbin,
        descarga_baterias_MW=Pbout,
        potencia_baterias_MW=Pbout - Pbin,
        soc_baterias=Soc,
        perdas_MW=Ploss.sum(axis=1),
        corte_MW=Pcut.sum(axis=1),
        deficit_MW=Pdef.sum(axis=1),
        fluxo_pct=fluxo_pct,
        fluxo_de_para_MW=Fd * PB,
        fluxo_para_de_MW=Fc * PB,
        ocupacao_max_pct=fluxo_pct.max(axis=(1, 2)),
        linhas_de=DLIN["De_original"].to_numpy(),
        linhas_para=DLIN["Para_original"].to_numpy(),
        iteracoes_perdas=np.asarray([solucao["iteracoes"] for solucao in solucoes]),
        erro_perdas_MW=np.asarray([solucao["erro_perdas_MW"] for solucao in solucoes]),
        tempo_s=np.asarray([solucao["tempo"] for solucao in solucoes]),
        funcao_objetivo=np.asarray([solucao["Fob"] for solucao in solucoes]),
    )


if __name__ == "__main__":
    main()
