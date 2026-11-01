import numpy as np

from configuracao import carregar_config


config = carregar_config()

from dados_ca import carregar_sistema
from fluxo_ca import montar_ybarra, preparar_controles_tap


def executar():
    dbar, dlin, dger, _, _, _ = carregar_sistema(config.ARQUIVO, config)
    ybarra = montar_ybarra(dbar, dlin)
    ajustaveis = dlin[(dlin["TapMin"] > 0.0) & (dlin["TapMax"] > 0.0)]
    controles = preparar_controles_tap(dlin)

    linhas = [
        f"INVENTARIO DO SISTEMA {config.SISTEMA}",
        "=" * 80,
        f"Barras: {len(dbar)}",
        f"Linhas e transformadores: {len(dlin)}",
        f"Geradores convencionais e compensadores sincronos: {len(dger)}",
        f"Barras com MMGD: {len(config.BARRAS_MMGD)}",
        f"Carga ativa nominal: {dbar['Pd'].sum():.2f} MW",
        f"Carga reativa nominal: {dbar['Qd'].sum():.2f} MVAr",
        f"Ramos com faixa de tap no arquivo: {len(ajustaveis)}",
        f"Controles de tap ativos no modelo: {len(controles)}",
        f"Barras com shunt fixo: {np.count_nonzero(dbar['Sh_MVAr'])}",
        f"Soma algebrica dos shunts: {dbar['Sh_MVAr'].sum():.2f} MVAr",
        f"Compensador estatico fixo: {dbar['Qg_fixo_MVAr'].sum():.2f} MVAr",
        f"Matriz Ybus finita: {np.isfinite(ybarra).all()}",
        "",
        "GRUPOS DE TAP E POSICOES INICIAIS",
        "-" * 80,
    ]
    for (de, para), grupo in ajustaveis.groupby(["De_original", "Para_original"]):
        taps = ", ".join(f"{valor:.4f}" for valor in grupo["Tap"].to_numpy(float))
        linhas.append(
            f"{int(de)}-{int(para)} | circuitos={len(grupo)} | taps={taps} | "
            f"faixa={grupo['TapMin'].max():.4f} a {grupo['TapMax'].min():.4f}"
        )

    linhas.extend(["", "COMPENSADORES E SHUNTS FIXOS", "-" * 80])
    for _, barra in dbar[dbar["Sh_MVAr"] != 0.0].iterrows():
        linhas.append(
            f"Barra {int(barra['Barra_original'])}: shunt={barra['Sh_MVAr']:.2f} MVAr"
        )
    for _, barra in dbar[dbar["Qg_fixo_MVAr"] != 0.0].iterrows():
        linhas.append(
            f"Barra {int(barra['Barra_original'])}: injecao reativa fixa="
            f"{barra['Qg_fixo_MVAr']:.2f} MVAr"
        )

    texto = "\n".join(linhas)
    pasta = getattr(config, "PASTA_INVENTARIO", config.PASTA)
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / f"inventario_sistema_{config.SISTEMA.lower()}.txt"
    arquivo.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nArquivo: {arquivo}")


if __name__ == "__main__":
    executar()
