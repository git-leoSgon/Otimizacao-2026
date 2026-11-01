from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


PASTA = Path(__file__).absolute().parent
RESULTADOS = PASTA / "Resultados_B9_MMGD"
FIGURAS = RESULTADOS / "Figuras"
ARQUIVO_COMPARACAO = PASTA / "Resultados_B9_MMGD_Baterias" / "B9_MMGD_Baterias_resultados.npz"

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
    z = np.load(RESULTADOS / "B9_MMGD_resultados.npz")
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
    largura = 0.72 / len(barras_mmgd)
    cores_barras = [AZUL, VERDE, VERMELHO]
    plt.figure()
    for b, barra in enumerate(barras_mmgd):
        deslocamento = (b - (len(barras_mmgd) - 1) / 2) * largura
        plt.bar(
            x + deslocamento, capacidades[:, b], largura,
            color=cores_barras[b], label=f"Barra {int(barra)}",
        )
    plt.xlabel("Penetração em Relação à Carga Local")
    plt.ylabel("Potência [MW]")
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

    print(f"Figuras salvas em: {FIGURAS}")


if __name__ == "__main__":
    main()
