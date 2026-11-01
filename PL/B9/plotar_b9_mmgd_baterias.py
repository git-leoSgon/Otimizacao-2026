from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


PASTA = Path(__file__).absolute().parent
RESULTADOS = PASTA / "Resultados_B9_MMGD_Baterias"
FIGURAS = RESULTADOS / "Figuras"
ARQUIVO_COMPARACAO = PASTA / "Resultados_B9_MMGD" / "B9_MMGD_resultados.npz"

AZUL = "#2878B5"
VERDE = "#2A9D55"
VERMELHO = "#D64545"
LARANJA = "#E68632"

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
})


def salvar(nome):
    figura = plt.gcf()
    figura.set_size_inches(11, 6.8)
    for eixo in figura.axes:
        eixo.xaxis.label.set_fontweight("bold")
        eixo.yaxis.label.set_fontweight("bold")
        for rotulo in eixo.get_xticklabels() + eixo.get_yticklabels():
            rotulo.set_fontweight("bold")
    figura.tight_layout()
    figura.savefig(FIGURAS / nome, dpi=300, bbox_inches="tight")
    plt.close()


def faixa_comum(valores, incluir_zero=False):
    minimo = min(float(np.min(valor)) for valor in valores)
    maximo = max(float(np.max(valor)) for valor in valores)
    if incluir_zero:
        minimo = min(0.0, minimo)
        maximo = max(0.0, maximo)
    amplitude = maximo - minimo
    margem = 0.08 * amplitude if amplitude > 1e-9 else max(1.0, 0.08 * abs(maximo))
    inferior = minimo - margem
    if incluir_zero and minimo >= 0:
        inferior = 0.0
    return inferior, maximo + margem


def calcular_escalas(z):
    resultados = [z]
    if ARQUIVO_COMPARACAO.exists():
        resultados.append(np.load(ARQUIVO_COMPARACAO))

    mmgd = [r["mmgd_injetada_MW"].sum(axis=1) for r in resultados]
    geracao = [r["geracao_conv_MW"] for r in resultados]
    perdas = [r["perdas_MW"].sum(axis=1) for r in resultados]
    ocupacao_maxima = max(float(np.max(r["fluxo_pct"])) for r in resultados)
    corte_deficit = [r["corte_MW"].sum(axis=1) for r in resultados]
    corte_deficit += [r["deficit_MW"].sum(axis=1) for r in resultados]

    balanco = []
    for r in resultados:
        balanco.extend([
            r["carga_h_MW"], r["geracao_conv_MW"],
            r["mmgd_injetada_MW"].sum(axis=1), r["perdas_MW"],
        ])
        if "potencia_baterias_MW" in r.files:
            balanco.append(r["potencia_baterias_MW"].sum(axis=1))

    return {
        "referencia": (0.0, 1.08),
        "mmgd": faixa_comum(mmgd, incluir_zero=True),
        "geracao": faixa_comum(geracao),
        "perdas": faixa_comum(perdas),
        "ocupacao": (0.0, max(105.0, 1.08 * ocupacao_maxima)),
        "balanco": faixa_comum(balanco, incluir_zero=True),
        "corte_deficit": (0.0, max(1.0, 1.08 * max(float(np.max(v)) for v in corte_deficit))),
    }


def linha_maior_efeito_bess(z):
    if not ARQUIVO_COMPARACAO.exists():
        return 0

    comparacao = np.load(ARQUIVO_COMPARACAO)
    if "potencia_baterias_MW" in z.files:
        com_bess, sem_bess = z, comparacao
    else:
        sem_bess, com_bess = z, comparacao

    fluxo_0 = sem_bess["fluxo_de_para_MW"][0]
    fluxo_sem = sem_bess["fluxo_de_para_MW"][-1]
    fluxo_com = com_bess["fluxo_de_para_MW"][-1]
    de = sem_bess["linhas_de"].astype(int)
    para = sem_bess["linhas_para"].astype(int)

    pares = [tuple(sorted((int(i), int(j)))) for i, j in zip(de, para)]
    contagem = Counter(pares)
    grafo = nx.Graph()
    grafo.add_edges_from(pares)
    pontes = {tuple(sorted(aresta)) for aresta in nx.bridges(grafo)}
    radial = np.array([par in pontes and contagem[par] == 1 for par in pares])

    horas_referencia = [0, 1, 2, 21, 22, 23]
    sentido = np.sign(np.mean(fluxo_0[:, horas_referencia], axis=1))
    sentido[sentido == 0] = 1.0
    reverso_sem = np.maximum(0.0, -sentido[:, None] * fluxo_sem)
    reverso_com = np.maximum(0.0, -sentido[:, None] * fluxo_com)
    pico_sem = np.max(reverso_sem, axis=1)
    reducao = pico_sem - np.max(reverso_com, axis=1)
    reducao_minima = max(1.0, 0.02 * float(np.max(pico_sem)))

    elegivel = (~radial) & (pico_sem > 1e-6) & (reducao >= reducao_minima)
    if np.any(elegivel):
        score = np.full(len(pico_sem), -np.inf)
        score[elegivel] = reducao[elegivel] / pico_sem[elegivel]
        return int(np.argmax(score))

    efeito = np.max(np.abs(fluxo_com - fluxo_sem), axis=1)
    if np.any(~radial):
        efeito = np.where(~radial, efeito, -np.inf)
    return int(np.argmax(efeito))


def escala_fluxo_linha(z, linha):
    valores = [z["fluxo_de_para_MW"][:, linha]]
    if ARQUIVO_COMPARACAO.exists():
        comparacao = np.load(ARQUIVO_COMPARACAO)
        valores.append(comparacao["fluxo_de_para_MW"][:, linha])
    return faixa_comum(valores, incluir_zero=True)


def main():
    FIGURAS.mkdir(parents=True, exist_ok=True)
    for arquivo in FIGURAS.glob("*.png"):
        arquivo.unlink()
    z = np.load(RESULTADOS / "B9_MMGD_Baterias_resultados.npz")
    escalas = calcular_escalas(z)

    horas = np.arange(24)
    penetracoes = z["penetracoes"]
    capacidades = z["capacidades_mmgd_MW"]
    cores = plt.cm.viridis(np.linspace(0.10, 0.90, len(penetracoes)))

    plt.figure()
    plt.plot(horas, z["curva_carga"], marker="o", markersize=4, color=AZUL, label="Carga")
    plt.plot(horas, z["curva_solar"], marker="o", markersize=4, color=LARANJA, label="MMGD solar")
    plt.xlabel("Hora")
    plt.ylabel("Potência [pu]")
    plt.ylim(*escalas["referencia"])
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    salvar("01_curvas_referencia.png")

    x = np.arange(len(penetracoes))
    barras_mmgd = z["barras_mmgd"]
    largura = 0.78 / (2 * len(barras_mmgd))
    cores_barras = plt.cm.turbo(np.linspace(0.05, 0.95, len(barras_mmgd)))
    plt.figure()
    for b, barra in enumerate(barras_mmgd):
        deslocamento_mmgd = (2 * b - (2 * len(barras_mmgd) - 1) / 2) * largura
        deslocamento_bat = deslocamento_mmgd + largura
        plt.bar(
            x + deslocamento_mmgd, capacidades[:, b], largura,
            color=cores_barras[b], label=f"MMGD - Barra {int(barra)}",
        )
        plt.bar(
            x + deslocamento_bat, z["capacidade_baterias_MWh"][:, b], largura,
            color=cores_barras[b], alpha=0.45, hatch="//",
            label=f"Bateria - Barra {int(barra)}",
        )
    plt.xlabel("Penetração em Relação à Carga Local")
    plt.ylabel("Capacidade [MW ou MWh]")
    plt.xticks(x, [f"{p}%" for p in penetracoes])
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper left")
    salvar("02_capacidade_mmgd.png")

    plt.figure()
    for k, penetracao in enumerate(penetracoes):
        plt.plot(horas, z["mmgd_injetada_MW"][k].sum(axis=0), color=cores[k], linewidth=2, label=f"{penetracao}%")
    plt.xlabel("Hora")
    plt.ylabel("Potência [MW]")
    plt.ylim(*escalas["mmgd"])
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend(title="Penetração", ncol=2, loc="upper left")
    salvar("03_geracao_mmgd.png")

    plt.figure()
    for k, penetracao in enumerate(penetracoes):
        plt.plot(horas, z["geracao_conv_MW"][k], color=cores[k], linewidth=2, label=f"{penetracao}%")
    plt.xlabel("Hora")
    plt.ylabel("Potência [MW]")
    plt.ylim(*escalas["geracao"])
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend(title="Penetração", ncol=2, loc="lower left")
    salvar("04_geracao_convencional.png")

    plt.figure()
    plt.plot(penetracoes, z["perdas_MW"].sum(axis=1), marker="o", markersize=6, color=VERMELHO)
    plt.xlabel("Penetração da MMGD [%]")
    plt.ylabel("Perdas [MWh]")
    plt.ylim(*escalas["perdas"])
    plt.grid(alpha=0.3)
    salvar("05_perdas_ativas.png")

    plt.figure()
    plt.plot(penetracoes, z["ocupacao_max_pct"], marker="o", markersize=6, color=AZUL, label="Máxima diária")
    plt.axhline(100, color=VERMELHO, linestyle="--", linewidth=1.5, label="Limite")
    plt.xlabel("Penetração da MMGD [%]")
    plt.ylabel("Ocupação [%]")
    plt.ylim(*escalas["ocupacao"])
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    salvar("06_ocupacao_maxima.png")

    fluxo = z["fluxo_pct"]
    _, linha_critica, _ = np.unravel_index(np.argmax(fluxo), fluxo.shape)
    de = int(z["linhas_de"][linha_critica])
    para = int(z["linhas_para"][linha_critica])
    plt.figure()
    for k, penetracao in enumerate(penetracoes):
        plt.plot(horas, fluxo[k, linha_critica], color=cores[k], linewidth=2, label=f"{penetracao}%")
    plt.axhline(100, color=VERMELHO, linestyle="--", linewidth=1.5, label="Limite")
    plt.xlabel("Hora")
    plt.ylabel("Ocupação [%]")
    plt.ylim(*escalas["ocupacao"])
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend(title=f"Linha {de}-{para}", ncol=2, loc="lower left")
    salvar("07_linha_critica.png")

    fluxo_mw = z["fluxo_de_para_MW"]
    plt.figure()
    for k, penetracao in enumerate(penetracoes):
        plt.plot(horas, fluxo_mw[k, linha_critica], color=cores[k], linewidth=2, label=f"{penetracao}%")
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Hora")
    plt.ylabel("Fluxo [MW]")
    plt.ylim(*escala_fluxo_linha(z, linha_critica))
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend(
        title=f"Positivo: {de} para {para}\nNegativo: {para} para {de}",
        ncol=2, loc="lower left",
    )
    salvar("07b_fluxo_mw_linha_critica.png")

    linha_inversao = linha_maior_efeito_bess(z)
    de_inv = int(z["linhas_de"][linha_inversao])
    para_inv = int(z["linhas_para"][linha_inversao])
    plt.figure()
    for k, penetracao in enumerate(penetracoes):
        plt.plot(horas, fluxo_mw[k, linha_inversao], color=cores[k], linewidth=2, label=f"{penetracao}%")
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Hora")
    plt.ylabel("Fluxo [MW]")
    plt.ylim(*escala_fluxo_linha(z, linha_inversao))
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend(
        title=f"Positivo: {de_inv} para {para_inv}\nNegativo: {para_inv} para {de_inv}",
        ncol=2, loc="lower left",
    )
    salvar("07c_fluxo_mw_maior_inversao.png")

    for k, nome in [(0, "08_balanco_potencia_0pct.png"), (-1, "09_balanco_potencia_50pct.png")]:
        mmgd = z["mmgd_injetada_MW"][k].sum(axis=0)
        plt.figure()
        plt.plot(horas, z["carga_h_MW"], color=VERMELHO, linewidth=2.2, label="Carga")
        plt.plot(horas, z["geracao_conv_MW"][k], color=AZUL, linewidth=2.2, label="Geração convencional")
        plt.plot(horas, mmgd, color=VERDE, linewidth=2.2, label="MMGD")
        plt.plot(
            horas, z["potencia_baterias_MW"][k].sum(axis=0), color="#7B4AB5", linewidth=2,
            label="Baterias",
        )
        plt.plot(horas, z["perdas_MW"][k], color=LARANJA, linewidth=1.8, label="Perdas")
        plt.xlabel("Hora")
        plt.ylabel("Potência [MW]")
        plt.ylim(*escalas["balanco"])
        plt.xticks(np.arange(0, 24, 2))
        plt.grid(alpha=0.3)
        plt.legend(
            title=f"Penetração: {penetracoes[k]}%", ncol=2, loc="upper left"
        )
        salvar(nome)

    plt.figure()
    plt.plot(penetracoes, z["corte_MW"].sum(axis=1), marker="o", color=VERMELHO, label="Corte de MMGD")
    plt.plot(penetracoes, z["deficit_MW"].sum(axis=1), marker="o", color=VERDE, label="Déficit")
    plt.xlabel("Penetração da MMGD [%]")
    plt.ylabel("Energia [MWh]")
    plt.ylim(*escalas["corte_deficit"])
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    salvar("10_corte_deficit.png")

    k = len(penetracoes) - 1
    horas_soc = np.arange(25)
    capacidades_bat = z["capacidade_baterias_MWh"][k]
    soc_sistema = np.average(z["soc_baterias"][k], axis=0, weights=capacidades_bat)

    plt.figure()
    plt.plot(horas_soc, 100 * soc_sistema, color=VERMELHO, linewidth=2.2, label="SOC do sistema")
    plt.xlabel("Hora")
    plt.ylabel("Estado de carga - SOC [%]")
    plt.xticks(np.arange(0, 25, 2))
    plt.xlim(0, 24)
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    salvar("11_operacao_conjunta_baterias.png")

    for b, barra in enumerate(barras_mmgd):
        plt.figure()
        plt.plot(
            horas_soc, 100 * z["soc_baterias"][k, b], color=VERMELHO,
            linewidth=2.2, label=f"SOC - Barra {int(barra)}",
        )
        plt.xlabel("Hora")
        plt.ylabel("Estado de carga - SOC [%]")
        plt.xticks(np.arange(0, 25, 2))
        plt.xlim(0, 24)
        plt.ylim(0, 100)
        plt.grid(alpha=0.3)
        plt.legend(loc="upper left")
        salvar(f"{12 + b:02d}_bateria_barra_{int(barra)}.png")

    mmgd_total = z["mmgd_injetada_MW"][k].sum(axis=0)
    carga_baterias = z["carga_baterias_MW"][k].sum(axis=0)
    descarga_baterias = z["descarga_baterias_MW"][k].sum(axis=0)
    demanda_sem_bess = z["carga_h_MW"] - mmgd_total
    demanda_com_bess = demanda_sem_bess + carga_baterias - descarga_baterias

    plt.figure()
    plt.plot(
        horas, demanda_sem_bess, color=AZUL, linewidth=2.2,
        label="Demanda líquida sem BESS",
    )
    plt.plot(
        horas, demanda_com_bess, color=VERMELHO, linewidth=2.2,
        label="Demanda líquida com BESS",
    )
    plt.xlabel("Hora")
    plt.ylabel("Potência [MW]")
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    indice_figura = 12 + len(barras_mmgd)
    salvar(f"{indice_figura:02d}_demanda_liquida_bess.png")

    dbar = pd.read_excel(PASTA / "STB_9.xlsx", sheet_name="DBAR")
    barras_sistema = dbar["Barra"].astype(int).to_numpy()
    for b, barra in enumerate(barras_mmgd):
        indice_barra = int(np.flatnonzero(barras_sistema == int(barra))[0])
        carga_demandada = (
            float(dbar.loc[indice_barra, "Pd"])
            * z["curva_carga"]
            * z["fatores_carga"][indice_barra]
        )
        demanda_sem_bess_barra = carga_demandada - z["mmgd_injetada_MW"][k, b]
        demanda_com_bess_barra = (
            demanda_sem_bess_barra
            + z["carga_baterias_MW"][k, b]
            - z["descarga_baterias_MW"][k, b]
        )

        plt.figure()
        plt.plot(
            horas, demanda_sem_bess_barra, color=AZUL, linewidth=2.2,
            label="Demanda líquida sem BESS",
        )
        plt.plot(
            horas, demanda_com_bess_barra, color=VERMELHO, linewidth=2.2,
            label="Demanda líquida com BESS",
        )
        plt.xlabel("Hora")
        plt.ylabel("Potência [MW]")
        plt.xticks(np.arange(0, 24, 2))
        plt.xlim(0, 23)
        plt.ylim(top=90)
        plt.grid(alpha=0.3)
        plt.legend(title=f"Barra {int(barra)}", loc="upper left")
        salvar(
            f"{indice_figura + 1:02d}_carga_supervisionada_barra_"
            f"{int(barra)}.png"
        )

    fig, ax_potencia = plt.subplots()
    ax_soc = ax_potencia.twinx()
    ax_potencia.plot(
        horas, demanda_sem_bess, color=AZUL, linewidth=2.2,
        label="Demanda líquida sem BESS",
    )
    ax_potencia.plot(
        horas, demanda_com_bess, color=VERMELHO, linewidth=2.2,
        label="Demanda líquida com BESS",
    )
    ax_soc.plot(
        horas, 100 * soc_sistema[1:], color=VERDE, linewidth=2.0,
        linestyle="--", label="SOC médio",
    )
    ax_potencia.set_xlabel("Hora")
    ax_potencia.set_ylabel("Potência [MW]")
    ax_soc.set_ylabel("SOC médio [%]", color=VERDE)
    ax_potencia.set_xticks(np.arange(0, 24, 2))
    ax_potencia.set_xlim(0, 23)
    ax_soc.set_ylim(0, 100)
    ax_soc.tick_params(axis="y", colors=VERDE)
    ax_potencia.grid(alpha=0.3)
    linhas_1, rotulos_1 = ax_potencia.get_legend_handles_labels()
    linhas_2, rotulos_2 = ax_soc.get_legend_handles_labels()
    ax_potencia.legend(
        linhas_1 + linhas_2, rotulos_1 + rotulos_2, loc="upper left"
    )
    salvar(f"{indice_figura + 2:02d}_demanda_liquida_bess_soc.png")

    for b, barra in enumerate(barras_mmgd):
        indice_barra = int(np.flatnonzero(barras_sistema == int(barra))[0])
        carga_demandada = (
            float(dbar.loc[indice_barra, "Pd"])
            * z["curva_carga"]
            * z["fatores_carga"][indice_barra]
        )
        demanda_sem_bess_barra = carga_demandada - z["mmgd_injetada_MW"][k, b]
        demanda_com_bess_barra = (
            demanda_sem_bess_barra
            + z["carga_baterias_MW"][k, b]
            - z["descarga_baterias_MW"][k, b]
        )
        fig, ax_potencia = plt.subplots()
        ax_soc = ax_potencia.twinx()
        ax_potencia.plot(
            horas, demanda_sem_bess_barra, color=AZUL, linewidth=2.2,
            label="Demanda líquida sem BESS",
        )
        ax_potencia.plot(
            horas, demanda_com_bess_barra, color=VERMELHO, linewidth=2.2,
            label="Demanda líquida com BESS",
        )
        ax_soc.plot(
            horas, 100 * z["soc_baterias"][k, b, 1:], color=VERDE,
            linewidth=2.0, linestyle="--", label="SOC da bateria",
        )
        ax_potencia.set_xlabel("Hora")
        ax_potencia.set_xticks(np.arange(0, 24, 2))
        ax_potencia.set_xlim(0, 23)
        ax_potencia.grid(alpha=0.3)
        ax_soc.set_ylim(0, 100)
        ax_soc.set_ylabel("SOC [%]", color=VERDE)
        ax_soc.tick_params(axis="y", colors=VERDE)
        linhas_1, rotulos_1 = ax_potencia.get_legend_handles_labels()
        linhas_2, rotulos_2 = ax_soc.get_legend_handles_labels()
        ax_potencia.legend(
            linhas_1 + linhas_2, rotulos_1 + rotulos_2,
            title=f"Barra {int(barra)}", loc="upper left"
        )
        ax_potencia.set_ylabel("Potência [MW]")
        ax_potencia.set_ylim(top=90)
        salvar(
            f"{indice_figura + 3:02d}_demanda_liquida_barra_"
            f"{int(barra)}_soc.png"
        )

    fig, ax_potencia = plt.subplots()
    ax_soc = ax_potencia.twinx()
    potencia_baterias = z["potencia_baterias_MW"][k]
    soc_baterias = 100 * z["soc_baterias"][k, :, 1:]

    for b, barra in enumerate(barras_mmgd):
        ax_potencia.plot(
            horas, potencia_baterias[b], color=cores_barras[b], linewidth=2.2,
            linestyle="-",
        )
        ax_soc.plot(
            horas, soc_baterias[b], color=cores_barras[b], linewidth=2.0,
            linestyle=":",
        )

    limite_potencia = 1.20 * float(np.max(np.abs(potencia_baterias)))
    ax_potencia.set_xlabel("Hora")
    ax_potencia.set_ylabel("Potência [MW]")
    ax_soc.set_ylabel("SOC [%]", color=VERDE)
    ax_potencia.set_xticks(np.arange(0, 24, 2))
    ax_potencia.set_xlim(0, 23)
    ax_potencia.set_ylabel("Potência da BESS [MW]")
    ax_potencia.set_ylim(-limite_potencia, limite_potencia)
    ax_soc.set_ylim(0, 100)
    ax_soc.tick_params(axis="y", colors=VERDE)
    ax_potencia.axhline(0, color="black", linewidth=0.9, alpha=0.65)
    ax_potencia.grid(alpha=0.3)

    legenda_barras = [
        Line2D([0], [0], color=cores_barras[b], linewidth=3.0,
               label=f"Barra {int(barra)}")
        for b, barra in enumerate(barras_mmgd)
    ]
    legenda_estilos = [
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="-",
               label="Potência"),
        Line2D([0], [0], color="black", linewidth=2.0, linestyle=":",
               label="SOC"),
    ]
    legenda_cores = ax_potencia.legend(
        handles=legenda_barras, loc="upper left"
    )
    ax_potencia.add_artist(legenda_cores)
    ax_potencia.legend(
        handles=legenda_estilos, loc="lower left"
    )
    salvar(f"{indice_figura + 4:02d}_potencia_bess_soc.png")

    print(f"Figuras salvas em: {FIGURAS}")


if __name__ == "__main__":
    main()
