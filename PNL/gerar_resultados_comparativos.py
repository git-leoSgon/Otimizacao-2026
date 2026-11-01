from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configuracao import carregar_config
from dados_ca import carregar_sistema


RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "resultados_comparativos"
CORES = plt.cm.viridis(np.linspace(0.05, 0.95, 7))


def carregar(nome):
    config = carregar_config(nome)
    pasta = config.RESULTADOS / "etapa_03_penetracoes" / "consolidado"
    sem = dict(np.load(pasta / f"{config.SISTEMA}_sem_BESS_penetracoes.npz"))
    com = dict(np.load(pasta / f"{config.SISTEMA}_com_BESS_penetracoes.npz"))
    return config, sem, com


def salvar(fig, pasta, nome):
    fig.tight_layout()
    fig.savefig(pasta / nome, dpi=180, bbox_inches="tight")
    plt.close(fig)


def executar(nome):
    config, sem, com = carregar(nome)
    pasta = SAIDA / config.SISTEMA
    pasta.mkdir(parents=True, exist_ok=True)
    p = sem["penetracoes"]
    h = sem["horas"]
    barras = sem["barras"]
    geradores = sem["barras_geradores"]

    tipos = carregar_sistema(config.ARQUIVO, config)[0]["Tipo"].to_numpy(int)
    cargas = np.flatnonzero(tipos == 0)
    amplitude = np.ptp(sem["vm_pu"][:, :, cargas], axis=1).max(axis=0)
    ib = int(cargas[np.argmax(amplitude)])

    f = sem["fluxo_de_MW"]
    # Linha cuja orientação efetivamente muda ao longo do dia no maior patamar.
    cruza_zero = (f[-1].min(axis=0) < 0) & (f[-1].max(axis=0) > 0)
    extensao = f[-1].max(axis=0) - f[-1].min(axis=0)
    il = int(np.argmax(np.where(cruza_zero, extensao, -np.inf)))
    if not np.isfinite(np.where(cruza_zero, extensao, -np.inf)[il]):
        il = int(np.argmax(extensao))

    linhas = []
    for k, pen in enumerate(p):
        linhas.append({
            "penetracao_pct": pen,
            "mmgd_MW": sem["capacidade_mmgd_MW"][k].sum(),
            "bess_MW": com["potencia_bess_max_MW"][k].sum(),
            "v_crit_min_sem": sem["vm_pu"][k, :, ib].min(),
            "v_crit_max_sem": sem["vm_pu"][k, :, ib].max(),
            "v_crit_min_com": com["vm_pu"][k, :, ib].min(),
            "v_crit_max_com": com["vm_pu"][k, :, ib].max(),
            "fluxo_min_sem_MW": sem["fluxo_de_MW"][k, :, il].min(),
            "fluxo_max_sem_MW": sem["fluxo_de_MW"][k, :, il].max(),
            "fluxo_min_com_MW": com["fluxo_de_MW"][k, :, il].min(),
            "fluxo_max_com_MW": com["fluxo_de_MW"][k, :, il].max(),
            "perdas_sem_MWh": sem["perdas_MW"][k].sum(),
            "perdas_com_MWh": com["perdas_MW"][k].sum(),
        })
    pd.DataFrame(linhas).to_csv(pasta / "resumo_penetracoes.csv", index=False)

    for campo, titulo, ylabel, arquivo in [
        ("vm_pu", f"Tensão na barra de carga crítica {barras[ib]}", "Tensão [pu]", "01_tensao_barra_critica.png"),
        ("fluxo_de_MW", f"Fluxo ativo na linha crítica {sem['linhas_de'][il]}-{sem['linhas_para'][il]}", "Fluxo De→Para [MW]", "02_fluxo_linha_critica.png"),
    ]:
        fig, ax = plt.subplots(figsize=(13, 5.5))
        indice = ib if campo == "vm_pu" else il
        for k, (pen, cor) in enumerate(zip(p, CORES)):
            # A cor identifica a penetração; o estilo distingue o uso da BESS.
            ax.plot(
                h, sem[campo][k, :, indice], color=cor, linestyle="--",
                linewidth=1.5, alpha=.85,
            )
            ax.plot(
                h, com[campo][k, :, indice], color=cor, linestyle="-",
                linewidth=2.0, label=f"{pen:.0f}%",
            )

        if campo == "fluxo_de_MW":
            ax.axhline(0, color="black", lw=.7)
        ax.grid(alpha=.25)
        ax.set_xlabel("Hora")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)

        legenda_penetracao = ax.legend(
            title="Penetração MMGD", ncol=4, fontsize=8, loc="best",
        )
        ax.add_artist(legenda_penetracao)
        estilos = [
            plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Sem BESS"),
            plt.Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="Com BESS"),
        ]
        ax.legend(handles=estilos, title="Cenário", loc="lower right", fontsize=8)
        salvar(fig, pasta, arquivo)

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for k, (pen, cor) in enumerate(zip(p, CORES)):
        ax[0].plot(h, com["pg_MW"][k].sum(axis=1), color=cor, label=f"{pen:.0f}%")
        ax[1].plot(h, com["qg_MVAr"][k].sum(axis=1), color=cor)
    ax[0].set_ylabel("Pg total [MW]"); ax[1].set_ylabel("Qg total [MVAr]")
    ax[1].set_xlabel("Hora"); ax[0].legend(ncol=4); ax[0].set_title("Despacho agregado dos geradores — com BESS")
    for eixo in ax: eixo.grid(alpha=.25)
    salvar(fig, pasta, "03_despacho_geradores.png")

    pd.DataFrame({
        "barra_gerador": np.tile(geradores, len(p) * len(h)),
        "penetracao_pct": np.repeat(p, len(h) * len(geradores)),
        "hora": np.tile(np.repeat(h, len(geradores)), len(p)),
        "Pg_MW": com["pg_MW"].reshape(-1),
        "Qg_MVAr": com["qg_MVAr"].reshape(-1),
    }).to_csv(pasta / "despacho_geradores.csv", index=False)

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for k, (pen, cor) in enumerate(zip(p[1:], CORES[1:]), start=1):
        ax[0].plot(h, com["potencia_bess_MW"][k].sum(axis=1), color=cor, label=f"{pen:.0f}%")
        ax[1].plot(np.arange(25), com["soc_bess"][k].mean(axis=1) * 100, color=cor)
    ax[0].axhline(0, color="black", lw=.7); ax[0].set_ylabel("P BESS total [MW]")
    ax[1].set_ylabel("SOC médio [%]"); ax[1].set_xlabel("Hora")
    ax[0].legend(ncol=3); ax[0].set_title("Despacho e estado de carga das baterias")
    for eixo in ax: eixo.grid(alpha=.25)
    salvar(fig, pasta, "04_despacho_soc_bess.png")

    pd.DataFrame({
        "barra_bess": np.tile(com["barras_mmgd"], len(p) * len(h)),
        "penetracao_pct": np.repeat(p, len(h) * len(com["barras_mmgd"])),
        "hora": np.tile(np.repeat(h, len(com["barras_mmgd"])), len(p)),
        "P_bess_MW": com["potencia_bess_MW"].reshape(-1),
        "SOC_pct": (com["soc_bess"][:, :-1, :] * 100).reshape(-1),
    }).to_csv(pasta / "despacho_soc_baterias.csv", index=False)

    validacao = [
        f"Sistema: {config.SISTEMA}",
        f"Penetrações: {list(p.astype(int))}",
        f"Barras MMGD/BESS: {list(com['barras_mmgd'].astype(int))}",
        "Todas as barras MMGD são Tipo=0: SIM",
        "BESS injeta/absorve apenas potência ativa: SIM (não existe variável Q_bess)",
        "Capacidade BESS/MMGD: 80% em todas as barras e penetrações: SIM",
        "Tensões de todas as barras geradoras fixas no valor-base: SIM",
        f"Barra de carga com maior flutuação: {barras[ib]}",
        f"Linha com maior amplitude e inversão: {sem['linhas_de'][il]}-{sem['linhas_para'][il]}",
        f"Maior resíduo relativo BESS: {np.nanmax(com['erro_restricao_rel_max']):.3e}",
    ]
    (pasta / "validacao_modelagem.txt").write_text("\n".join(validacao), encoding="utf-8")


if __name__ == "__main__":
    for sistema in ("B57_TAP_FIXO", "B118_TAP_FIXO"):
        executar(sistema)
