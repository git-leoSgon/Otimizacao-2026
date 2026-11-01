from pathlib import Path
import sys

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import config_b107 as config
from config_b107 import (
    ARQUIVO,
    BARRAS_MMGD,
    PB,
    PENETRACOES,
    RESULTADOS_SEM_BESS,
)
from DC_OPF.dados import carregar_sistema, montar_cenario
from DC_OPF.fpo_dc import resolver_com_perdas


def main():
    RESULTADOS_SEM_BESS.mkdir(exist_ok=True)

    DBAR, DLIN, DGER, DDEF, mapa = carregar_sistema(ARQUIVO, PB)
    cenario = montar_cenario(ARQUIVO, DBAR, mapa, config)
    carga_local = DBAR.set_index("Barra_original").loc[BARRAS_MMGD, "Pd"].to_numpy() * PB

    solucoes = []
    capacidades = []

    for penetracao in PENETRACOES:
        pmax_mmgd = carga_local * penetracao / 100.0 / PB
        solucao = resolver_com_perdas(
            DBAR, DLIN, DGER, DDEF, cenario, pmax_mmgd, config
        )
        solucoes.append(solucao)
        capacidades.append(pmax_mmgd * PB)

        ocupacao = 100.0 * np.max(
            np.maximum(np.abs(solucao["Fd"]), np.abs(solucao["Fc"]))
            / DLIN["Flim"].to_numpy()[:, None]
        )
        print(
            f"MMGD {penetracao:2d}% | {pmax_mmgd.sum() * PB:6.2f} MW | "
            f"perdas {solucao['Ploss'].sum() * PB:7.3f} MWh | linhas {ocupacao:6.2f}%"
        )

    capacidades = np.asarray(capacidades)
    Pg = np.stack([solucao["Pg"] for solucao in solucoes]) * PB
    Fd = np.stack([solucao["Fd"] for solucao in solucoes])
    Fc = np.stack([solucao["Fc"] for solucao in solucoes])
    Pcut = np.stack([solucao["Pcut"] for solucao in solucoes]) * PB
    Pdef = np.stack([solucao["Pdef"] for solucao in solucoes]) * PB
    Ploss = np.stack([solucao["Ploss"] for solucao in solucoes]) * PB

    limites = DLIN["Flim"].to_numpy() * PB
    fluxo_pct = 100.0 * np.maximum(np.abs(Fd), np.abs(Fc)) * PB / limites[None, :, None]
    carga_h = np.sum(DBAR["Pd"].to_numpy()[None, :] * cenario["Fcar"], axis=1) * PB
    mmgd_disp = capacidades[:, :, None] * cenario["Fcap"].T[None, :, :]
    mmgd_inj = mmgd_disp - Pcut

    barras_texto = ", ".join(str(barra) for barra in BARRAS_MMGD)
    fatores_texto = " | ".join(
        f"B{barra}={fator:.4f}"
        for barra, fator in zip(BARRAS_MMGD, cenario["fator_mmgd"])
    )
    cabecalho_pmax = "  ".join(f"Pmax B{barra}" for barra in BARRAS_MMGD)
    relatorio = [
        f"B107 - MMGD NAS BARRAS {barras_texto}",
        "",
        f"Fatores de MMGD: {fatores_texto}",
        "",
        f"Pen.  {cabecalho_pmax}  MMGD inj.  Ger. conv.  Perdas  Corte  Deficit  Linha max.",
    ]

    for k, penetracao in enumerate(PENETRACOES):
        pmax_texto = " ".join(f"{valor:8.2f}" for valor in capacidades[k])
        relatorio.append(
            f"{penetracao:3d}%  {pmax_texto} "
            f"{mmgd_inj[k].sum():10.2f} {Pg[k].sum():11.2f} {Ploss[k].sum():7.2f} "
            f"{Pcut[k].sum():6.2f} {Pdef[k].sum():8.2f} {fluxo_pct[k].max():10.2f}%"
        )

    texto = "\n".join(relatorio)
    print("\n" + texto)
    (RESULTADOS_SEM_BESS / "B107_MMGD_relatorio.txt").write_text(texto, encoding="utf-8")

    np.savez_compressed(
        RESULTADOS_SEM_BESS / "B107_MMGD_resultados.npz",
        penetracoes=np.asarray(PENETRACOES),
        barras_mmgd=np.asarray(BARRAS_MMGD),
        capacidades_mmgd_MW=capacidades,
        curva_carga=cenario["carga"],
        curva_solar=cenario["solar"],
        fatores_carga=cenario["fator_carga"],
        fatores_mmgd=cenario["fator_mmgd"],
        carga_h_MW=carga_h,
        mmgd_disponivel_MW=mmgd_disp,
        mmgd_injetada_MW=mmgd_inj,
        geracao_conv_MW=Pg.sum(axis=1),
        despacho_geradores_MW=Pg,
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
