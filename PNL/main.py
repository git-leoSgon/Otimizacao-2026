import os
import subprocess
import sys
import time
import csv
from pathlib import Path


# Configuracao principal: altere estas flags e execute este arquivo.
SISTEMA = "B118_TAP_FIXO"
SISTEMAS_PARA_EXECUTAR = ["B57_TAP_FIXO", "B118_TAP_FIXO"]
EXECUTAR_TODOS = True

PENETRACAO = None
MOSTRAR_IPOPT = True
MOSTRAR_MINLP = False
EXECUTAR_ETAPA_1 = False
EXECUTAR_ETAPA_2 = False
EXECUTAR_ETAPA_3 = True
REUTILIZAR_CASOS = False
RECALCULAR_COM_BESS = True
GERAR_INVENTARIO = True


if "FPO_SISTEMA" not in os.environ:
    os.environ["FPO_SISTEMA"] = SISTEMA

from configuracao import SISTEMAS, carregar_config
from simulacao_base import executar_caso_base, salvar_resultados
from simulacao_bess import executar_caso_bess, salvar_resultados as salvar_bess
from simulacao_penetracoes import executar as executar_penetracoes
from solvers import configurar_ambiente


config = carregar_config()


def executar_sistema():
    penetracao = config.PENETRACAO_TESTE if PENETRACAO is None else PENETRACAO
    print("=" * 88)
    print(
        "TRABALHO 3 - FPO-CA COM LIMITES OPERACIONAIS E BESS - "
        f"SISTEMA {config.SISTEMA}"
    )
    print("=" * 88)

    solvers = configurar_ambiente()
    print(f"IPOPT: {solvers['ipopt']}")
    print(f"Mestre MILP: {solvers['mip']}")
    print(f"GLPK alternativo: {solvers['glpk']}")
    print(f"Penetracao de MMGD: {penetracao}%")
    print(f"Barras com MMGD/BESS: {len(config.BARRAS_MMGD)}")
    print(f"Peso do desvio de Pg: {config.PESO_DESVIO_PG:.1f}")
    print("Tempo limite do MINLP: sem limite")
    if config.CONTROLES_TAP:
        print("Taps: automaticos")
    else:
        descricao = getattr(
            config,
            "DESCRICAO_TAPS",
            "fixos nos valores do arquivo de dados",
        )
        print(f"Taps: {descricao}")

    if GERAR_INVENTARIO:
        from inventariar_sistema import executar as inventariar

        inventariar()

    arquivo_base = (
        config.RESULTADOS
        / "etapa_01_fpo_ca_horario"
        / f"{config.SISTEMA}_FPO_CA_{penetracao:02d}pct.npz"
    )
    dados_base = None
    if EXECUTAR_ETAPA_1:
        print("Etapa 1: FPO-CA horario sem BESS e validacao pelo Newton-Raphson\n")
        dados_base = executar_caso_base(config, penetracao, tee=MOSTRAR_IPOPT)
        arquivo, relatorio = salvar_resultados(dados_base, config)
        print("\nEtapa 1 concluida.")
        print(f"Banco: {arquivo}")
        print(f"Relatorio: {relatorio}")
    elif arquivo_base.exists():
        import numpy as np

        dados_base = dict(np.load(arquivo_base, allow_pickle=False))
        print(f"Etapa 1 disponivel: {arquivo_base}")

    if EXECUTAR_ETAPA_2:
        if dados_base is None:
            raise FileNotFoundError(
                "A etapa 2 exige o resultado da etapa 1 para a mesma penetracao."
            )
        print("\nEtapa 2: FPO-CA multiperiodo com BESS")
        dados_bess = executar_caso_bess(
            config, penetracao, dados_base, tee=MOSTRAR_MINLP
        )
        arquivo, relatorio = salvar_bess(dados_bess, config)
        print("\nEtapa 2 concluida.")
        print(f"Banco: {arquivo}")
        print(f"Relatorio: {relatorio}")

    if EXECUTAR_ETAPA_3:
        print("\nEtapa 3: varredura das penetracoes com e sem BESS")
        executar_penetracoes(
            config,
            reutilizar=REUTILIZAR_CASOS,
            tee=MOSTRAR_MINLP,
            recalcular_com_bess=RECALCULAR_COM_BESS,
        )

def executar_todos():
    desconhecidos = [nome for nome in SISTEMAS_PARA_EXECUTAR if nome not in SISTEMAS]
    if desconhecidos:
        raise ValueError(f"Sistemas invalidos: {desconhecidos}")

    tempos = []
    configs = [carregar_config(nome) for nome in SISTEMAS_PARA_EXECUTAR]
    pastas = {item.RESULTADOS.parent for item in configs}
    pasta_tempos = (
        pastas.pop()
        if len(pastas) == 1
        else Path(__file__).resolve().parent / "resultados"
    )
    arquivo_tempos = pasta_tempos / "tempos_execucao_sistemas.csv"
    for numero, sistema in enumerate(SISTEMAS_PARA_EXECUTAR, start=1):
        print("\n" + "#" * 88)
        print(f"SISTEMA {numero}/{len(SISTEMAS_PARA_EXECUTAR)}: {sistema}")
        print("#" * 88)
        ambiente = os.environ.copy()
        ambiente["FPO_SISTEMA"] = sistema
        ambiente["FPO_EXECUCAO_FILHA"] = "1"
        inicio = time.perf_counter()
        situacao = "concluido"
        try:
            subprocess.run([sys.executable, str(__file__)], env=ambiente, check=True)
        except subprocess.CalledProcessError:
            situacao = "erro"
            raise
        finally:
            tempos.append(
                {
                    "sistema": sistema,
                    "tempo_total_parede_s": time.perf_counter() - inicio,
                    "situacao": situacao,
                }
            )
            arquivo_tempos.parent.mkdir(parents=True, exist_ok=True)
            with arquivo_tempos.open("w", newline="", encoding="utf-8-sig") as arquivo:
                escritor = csv.DictWriter(
                    arquivo,
                    fieldnames=["sistema", "tempo_total_parede_s", "situacao"],
                    delimiter=";",
                )
                escritor.writeheader()
                escritor.writerows(tempos)

    print(f"\nRegistro dos tempos: {arquivo_tempos}")


if __name__ == "__main__":
    if EXECUTAR_TODOS and os.environ.get("FPO_EXECUCAO_FILHA") != "1":
        executar_todos()
    else:
        executar_sistema()
