from pathlib import Path

import numpy as np

from simulacao_base import executar_caso_base
from simulacao_bess import executar_caso_bess
from solvers import configurar_ambiente


def _salvar_npz(caminho, dados):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(caminho, **dados)


def _carregar_npz(caminho):
    return dict(np.load(caminho, allow_pickle=False))


def _normalizar_bess(dados):
    resultado = {}
    for chave, valor in dados.items():
        if chave == "componentes_fob":
            for nome, componente in valor.items():
                resultado[f"fob_{nome}"] = np.asarray(componente)
        elif chave not in {"sucesso", "viavel_multistart"}:
            resultado[chave] = np.asarray(valor)
    return resultado


def _versao_bess_atual(dados, config):
    if "versao_solucao_bess" not in dados:
        return False
    versao = str(np.asarray(dados["versao_solucao_bess"]).item())
    return versao == config.VERSAO_SOLUCAO_BESS


def _referencia_sem_bess(base, config):
    nbat = len(base["barras_mmgd"])
    zeros = np.zeros((24, nbat))
    indices = [
        int(np.flatnonzero(base["barras"] == barra)[0])
        for barra in base["barras_mmgd"]
    ]
    referencia = dict(base)
    referencia.update(
        {
            "potencia_bess_max_MW": np.zeros(nbat),
            "energia_bess_max_MWh": np.zeros(nbat),
            "potencia_bess_MW": zeros,
            "carga_bess_MW": zeros,
            "descarga_bess_MW": zeros,
            "modo_carga": np.zeros((24, nbat), dtype=int),
            "energia_bess_MWh": np.zeros((25, nbat)),
            "soc_bess": np.full((25, nbat), config.SOC_INICIAL),
            "pliq_barras_MW": base["carga_MW"][:, indices]
            - base["mmgd_injetada_MW"],
            "partida": np.array("referencia_sem_bess"),
            "estrategia_inicial": np.array("nenhuma"),
            "condicao": np.array("referencia"),
            "fob": np.array(np.nan),
            "fob_fisica": np.array(np.nan),
            "fob_desvio_pg": np.array(0.0),
            "indice_desvio_pg": np.array(0.0),
            "desvio_pg_max_MW": np.array(0.0),
            "desvio_pg_medio_MW": np.array(0.0),
            "tempo_s": np.array(0.0),
            "tempo_multistart_s": np.array(0.0),
            "numero_variaveis": np.array(0),
            "numero_variaveis_binarias": np.array(0),
            "numero_restricoes": np.array(0),
            "erro_restricao_rel_max": np.array(0.0),
            "erro_validacao_vm_pu": np.array(0.0),
            "erro_validacao_pg_MW": np.array(0.0),
            "versao_solucao_bess": np.array(config.VERSAO_SOLUCAO_BESS),
        }
    )
    return referencia


def _consolidar(casos, penetracoes, com_bess):
    fixos = [
        "horas",
        "barras",
        "barras_geradores",
        "barras_mmgd",
        "linhas_de",
        "linhas_para",
        "limites_linhas_MVA",
        "curva_carga_pu",
        "curva_solar_pu",
        "carga_MW",
        "carga_MVAr",
        "numero_variaveis_fpo_horario",
        "numero_restricoes_fpo_horario",
        "numero_modelos_fpo_horario",
    ]
    variaveis = [
        "capacidade_mmgd_MW",
        "mmgd_injetada_MW",
        "pg_MW",
        "qg_MVAr",
        "deficit_MW",
        "vm_pu",
        "va_rad",
        "taps",
        "estado_q",
        "estado_tap",
        "perdas_MW",
        "fluxo_de_MW",
        "fluxo_para_MW",
        "ocupacao_pct",
        "violacao_tensao",
    ]
    if com_bess:
        variaveis.extend(
            [
                "potencia_bess_max_MW",
                "energia_bess_max_MWh",
                "potencia_bess_MW",
                "carga_bess_MW",
                "descarga_bess_MW",
                "modo_carga",
                "energia_bess_MWh",
                "soc_bess",
                "pliq_barras_MW",
                "fob",
                "fob_fisica",
                "fob_desvio_pg",
                "indice_desvio_pg",
                "desvio_pg_max_MW",
                "desvio_pg_medio_MW",
                "tempo_s",
                "tempo_multistart_s",
                "numero_variaveis",
                "numero_variaveis_binarias",
                "numero_restricoes",
                "erro_restricao_rel_max",
                "erro_validacao_vm_pu",
                "erro_validacao_pg_MW",
            ]
        )
    else:
        variaveis.append("tempo_total_s")

    consolidado = {chave: casos[0][chave] for chave in fixos}
    consolidado["penetracoes"] = np.asarray(penetracoes, dtype=float)
    for chave in variaveis:
        consolidado[chave] = np.stack([caso[chave] for caso in casos])
    if com_bess:
        consolidado["partida"] = np.array(
            [str(caso["partida"]) for caso in casos]
        )
        consolidado["condicao"] = np.array(
            [str(caso["condicao"]) for caso in casos]
        )
        consolidado["versao_solucao_bess"] = np.array(
            [str(caso["versao_solucao_bess"]) for caso in casos]
        )
    return consolidado


def _tabela_comparativa(sem, com, config):
    cabecalho = (
        "penetracao_pct;mmgd_instalada_MW;bess_instalada_MW;"
        "geracao_sem_MWh;geracao_com_MWh;perdas_sem_MWh;perdas_com_MWh;"
        "vmin_sem_pu;vmin_com_pu;vmax_sem_pu;vmax_com_pu;"
        "ocupacao_sem_pct;ocupacao_com_pct;violacoes_v_sem;"
        "violacoes_v_com;deficit_MWh;partida"
    )
    linhas = [cabecalho]
    resumo = [
        "FPO-CA MULTIPERIODO - COMPARACAO DAS PENETRACOES",
        "=" * 132,
        "",
        "Pen.  MMGD MW  BESS MW  Perdas sem  Perdas com  Vmax sem  Vmax com  "
        "Linha sem  Linha com  Viol. V sem/com  Deficit  Partida",
    ]
    for k, penetracao in enumerate(sem["penetracoes"]):
        partida = str(com["partida"][k])
        valores = [
            penetracao,
            sem["capacidade_mmgd_MW"][k].sum(),
            com["potencia_bess_max_MW"][k].sum(),
            sem["pg_MW"][k].sum(),
            com["pg_MW"][k].sum(),
            sem["perdas_MW"][k].sum(),
            com["perdas_MW"][k].sum(),
            sem["vm_pu"][k].min(),
            com["vm_pu"][k].min(),
            sem["vm_pu"][k].max(),
            com["vm_pu"][k].max(),
            sem["ocupacao_pct"][k].max(),
            com["ocupacao_pct"][k].max(),
            np.count_nonzero(sem["violacao_tensao"][k]),
            np.count_nonzero(com["violacao_tensao"][k]),
            com["deficit_MW"][k].sum(),
        ]
        linhas.append(";".join(f"{valor:.10f}" for valor in valores) + f";{partida}")
        resumo.append(
            f"{penetracao:4.0f}%  {valores[1]:7.2f}  {valores[2]:7.2f}  "
            f"{valores[5]:10.3f}  {valores[6]:10.3f}  "
            f"{valores[9]:8.5f}  {valores[10]:8.5f}  "
            f"{valores[11]:8.3f}%  {valores[12]:8.3f}%  "
            f"{int(valores[13]):6d}/{int(valores[14]):6d}  "
            f"{valores[15]:7.3f}  {partida}"
        )
    resumo.extend(
        [
            "",
            "A MMGD disponivel foi integralmente injetada em todos os casos.",
            "Os geradores convencionais possuem despacho ativo e reativo livre.",
            "O despacho ativo com BESS possui regularizacao suave em relacao ao caso sem BESS.",
            "As ultrapassagens da faixa operacional de tensao foram mantidas como resultado.",
            (
                "Os modos da BESS sao otimizados pelo MindtPy/HiGHS e os "
                "subproblemas continuos sao resolvidos pelo IPOPT."
                if config.OTIMIZAR_MODOS_BESS
                else
                "Os modos da BESS seguem a partida configurada e as "
                "variaveis continuas sao otimizadas pelo IPOPT."
            ),
        ]
    )
    return "\n".join(linhas), "\n".join(resumo)


def executar(config, reutilizar=True, tee=False, recalcular_com_bess=False):
    configurar_ambiente()
    raiz = config.RESULTADOS / "etapa_03_penetracoes"
    pasta_sem = raiz / "sem_bess" / "casos"
    pasta_com = raiz / "com_bess" / "casos"
    resultados_sem = []
    resultados_com = []

    for numero, penetracao in enumerate(config.PENETRACOES, start=1):
        print("\n" + "-" * 88)
        print(
            f"Penetracao {numero}/{len(config.PENETRACOES)}: "
            f"{penetracao}%"
        )
        nome_sem = f"{config.SISTEMA}_FPO_CA_{penetracao:02d}pct.npz"
        nome_com = (
            f"{config.SISTEMA}_FPO_CA_MINLP_BESS_"
            f"{penetracao:02d}pct.npz"
        )
        arquivo_sem = pasta_sem / f"{penetracao:02d}pct" / nome_sem
        arquivo_com = pasta_com / f"{penetracao:02d}pct" / nome_com

        if reutilizar and arquivo_sem.exists():
            sem = _carregar_npz(arquivo_sem)
            print("  sem BESS: caso existente reutilizado")
        else:
            arquivo_etapa_1 = (
                config.RESULTADOS
                / "etapa_01_fpo_ca_horario"
                / nome_sem
            )
            if reutilizar and arquivo_etapa_1.exists():
                sem = _carregar_npz(arquivo_etapa_1)
                print("  sem BESS: resultado da etapa 1 reutilizado")
            else:
                sem = executar_caso_base(config, penetracao, tee=False)
                print("  sem BESS: caso calculado")
            _salvar_npz(arquivo_sem, sem)
        resultados_sem.append(sem)

        com = None
        partida_bess = sem
        if resultados_com and penetracao != 0:
            # Continuacao parametrica: o patamar anterior fornece uma partida
            # muito melhor para cenarios de alta penetracao que o caso sem BESS.
            partida_bess = dict(resultados_com[-1])
            partida_bess["pg_referencia_MW"] = sem["pg_MW"]
        if reutilizar and arquivo_com.exists() and not recalcular_com_bess:
            candidato = _carregar_npz(arquivo_com)
            if _versao_bess_atual(candidato, config):
                com = candidato
                print("  com BESS: caso existente compativel reutilizado")
            else:
                partida_bess = dict(candidato)
                partida_bess["pg_referencia_MW"] = sem["pg_MW"]
                print("  com BESS: resultado antigo sera recalculado")

        if com is None and penetracao == 0:
            com = _referencia_sem_bess(sem, config)
            _salvar_npz(arquivo_com, com)
            print("  com BESS: referencia sem capacidade instalada")
        elif com is None:
            arquivo_etapa_2 = (
                config.RESULTADOS
                / "etapa_02_minlp_bess"
                / (
                    f"{config.SISTEMA}_FPO_CA_MINLP_BESS_"
                    f"{penetracao:02d}pct.npz"
                )
            )
            if (
                reutilizar
                and arquivo_etapa_2.exists()
                and not recalcular_com_bess
            ):
                candidato = _carregar_npz(arquivo_etapa_2)
                if _versao_bess_atual(candidato, config):
                    com = candidato
                    print("  com BESS: resultado compativel da etapa 2 reutilizado")
                else:
                    partida_bess = dict(candidato)
                    partida_bess["pg_referencia_MW"] = sem["pg_MW"]
            if com is None:
                com = _normalizar_bess(
                    executar_caso_bess(
                        config,
                        penetracao,
                        partida_bess,
                        tee=tee,
                    )
                )
            _salvar_npz(arquivo_com, com)
        resultados_com.append(com)

        print(
            f"  perdas sem/com={sem['perdas_MW'].sum():.3f}/"
            f"{com['perdas_MW'].sum():.3f} MWh | "
            f"Vmax sem/com={sem['vm_pu'].max():.5f}/"
            f"{com['vm_pu'].max():.5f} pu | "
            f"linha sem/com={sem['ocupacao_pct'].max():.2f}/"
            f"{com['ocupacao_pct'].max():.2f}%"
        )

    sem = _consolidar(resultados_sem, config.PENETRACOES, com_bess=False)
    com = _consolidar(resultados_com, config.PENETRACOES, com_bess=True)
    consolidado = raiz / "consolidado"
    _salvar_npz(
        consolidado / f"{config.SISTEMA}_sem_BESS_penetracoes.npz", sem
    )
    _salvar_npz(
        consolidado / f"{config.SISTEMA}_com_BESS_penetracoes.npz", com
    )

    csv, texto = _tabela_comparativa(sem, com, config)
    consolidado.mkdir(parents=True, exist_ok=True)
    (consolidado / f"{config.SISTEMA}_comparacao_penetracoes.csv").write_text(
        csv, encoding="utf-8"
    )
    (consolidado / f"{config.SISTEMA}_resumo_penetracoes.txt").write_text(
        texto, encoding="utf-8"
    )
    print("\n" + texto)
    return sem, com

