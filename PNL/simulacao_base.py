import time

import numpy as np

from dados_ca import carregar_sistema, montar_cenarios_24h
from fluxo_ca import preparar_controles_tap, resolver_fluxo
from fluxo_ca_inicial import resolver_fluxo_proporcional
from fpo_ca_integrado import resolver_fpo_ca_horario


def _estado_reativo(solucao, dbar, dger):
    estado = np.zeros(len(dbar), dtype=int)
    for g, gerador in dger.iterrows():
        barra = int(gerador["Barra"])
        if solucao["Qg"][g] <= gerador["Qmin"] + 1.0e-2:
            estado[barra] = -1
        elif solucao["Qg"][g] >= gerador["Qmax"] - 1.0e-2:
            estado[barra] = 1
    return estado


def executar_caso_base(config, penetracao, tee=False, referencia_vm=None):
    dbar, dlin, dger, carga, solar, mapa = carregar_sistema(config.ARQUIVO, config)
    cenarios = montar_cenarios_24h(
        dbar, carga, solar, mapa, config, penetracao
    )
    if referencia_vm is None:
        referencia_vm = np.tile(
            [
                config.TENSAO_GERADORES[int(barra)]
                for barra in dger["Barra_original"].to_numpy(int)
            ],
            (24, 1),
        )
    referencia_vm = np.asarray(referencia_vm, dtype=float)
    if referencia_vm.shape != (24, len(dger)):
        raise ValueError("A referencia de tensao dos geradores possui dimensao incorreta.")
    for hora, cenario in enumerate(cenarios):
        cenario["vm_referencia_geradores"] = referencia_vm[hora]
    resultados = []
    controles_tap = preparar_controles_tap(dlin)
    partida_anterior = None
    inicio = time.time()

    for hora, cenario in enumerate(cenarios):
        p_mmgd = np.zeros(len(dbar))
        np.add.at(
            p_mmgd,
            cenario["barras_mmgd"],
            cenario["disponibilidade_mmgd"],
        )
        def nova_partida():
            return resolver_fluxo_proporcional(
                dbar,
                dlin,
                dger,
                cenario["Pd"],
                cenario["Qd"],
                p_mmgd=p_mmgd,
                tolerancia=config.TOLERANCIA_NR,
                max_iteracoes=config.MAX_ITERACOES_NR,
            )
        if partida_anterior is None:
            partida = nova_partida()
            if not partida["convergiu"]:
                raise RuntimeError(f"O fluxo inicial falhou na hora {hora}.")
        else:
            partida = partida_anterior

        try:
            solucao = resolver_fpo_ca_horario(
                dbar, dlin, dger, cenario, config, partida, tee=tee
            )
        except RuntimeError:
            partida = nova_partida()
            solucao = resolver_fpo_ca_horario(
                dbar, dlin, dger, cenario, config, partida, tee=tee
            )
        dlin_hora = dlin.copy()
        dlin_hora["Tap"] = solucao["taps"]
        estado_q = solucao["estado_q"]
        vm_especificado = {
            int(dger.loc[g, "Barra"]): float(referencia_vm[hora, g])
            for g in range(len(dger))
        }
        verificacao = resolver_fluxo(
            dbar,
            dlin_hora,
            dger,
            cenario["Pd"],
            cenario["Qd"],
            solucao["Pg"],
            vm_especificado=vm_especificado,
            p_mmgd=p_mmgd,
            p_deficit=solucao["Pdef"],
            vm_inicial=solucao["Vm"],
            va_inicial=solucao["Va"],
            estado_q_inicial=solucao["estado_q"],
            atualizar_controles_q=False,
            tolerancia_q=config.TOLERANCIA_REATIVO_MVAR,
            max_iteracoes_controle_q=config.MAX_ITERACOES_PV_PQ,
            tolerancia=config.TOLERANCIA_NR,
            max_iteracoes=config.MAX_ITERACOES_NR,
        )
        if not verificacao["convergiu"] or not verificacao["controle_q_convergiu"]:
            raise RuntimeError(
                f"A verificacao Newton-Raphson com controle PV/PQ falhou na hora {hora}."
            )

        erro_vm = float(np.max(np.abs(solucao["Vm"] - verificacao["Vm"])))
        erro_pg = float(np.max(np.abs(solucao["Pg"] - verificacao["Pg"])))
        solucao["erro_validacao_vm_pu"] = erro_vm
        solucao["erro_validacao_pg_MW"] = erro_pg
        solucao["Vm_nr"] = verificacao["Vm"]
        solucao["Va_nr"] = verificacao["Va"]
        solucao["Pg_nr"] = verificacao["Pg"]
        solucao["Qg_nr"] = verificacao["Qg"]
        solucao["perdas_nr_MW"] = verificacao["perdas_ativas"]
        solucao["ocupacao_nr"] = verificacao["fluxos"]["ocupacao"]
        solucao["estado_q_nr"] = verificacao["estado_q"]
        solucao["estado_q"] = estado_q
        estado_tap = np.zeros(len(controles_tap), dtype=int)
        for k, controle in enumerate(controles_tap):
            tap = solucao["taps"][controle["linhas"][0]]
            if tap <= controle["minimo"] + config.TOLERANCIA_TAP:
                estado_tap[k] = -1
            elif tap >= controle["maximo"] - config.TOLERANCIA_TAP:
                estado_tap[k] = 1
        solucao["estado_tap"] = estado_tap
        partida_anterior = {
            "Vm": solucao["Vm"],
            "Va": solucao["Va"],
            "Pg": solucao["Pg"],
            "Qg": solucao["Qg"],
            "taps": solucao["taps"],
            "estado_q": solucao["estado_q"],
            "estado_tap": estado_tap,
        }
        resultados.append(solucao)
        print(
            f"Hora {hora:02d}/23 | Pg={solucao['Pg'].sum():8.3f} MW | "
            f"perdas={solucao['perdas_MW']:7.4f} MW | "
            f"V={solucao['Vm'].min():.4f}-{solucao['Vm'].max():.4f} pu | "
            f"linha={solucao['ocupacao'].max():6.2f}% | "
            f"NR dV={erro_vm:.2e} pu"
        )

    dados = {
        "horas": np.arange(24),
        "penetracao": np.array(float(penetracao)),
        "barras": dbar["Barra_original"].to_numpy(int),
        "barras_geradores": dger["Barra_original"].to_numpy(int),
        "barras_mmgd": cenarios[0]["barras_mmgd_originais"],
        "vm_referencia_geradores": referencia_vm,
        "linhas_de": dlin["De_original"].to_numpy(int),
        "linhas_para": dlin["Para_original"].to_numpy(int),
        "limites_linhas_MVA": dlin["Flim"].to_numpy(float),
        "curva_carga_pu": carga,
        "curva_solar_pu": solar,
        "carga_MW": np.stack([c["Pd"] for c in cenarios]),
        "carga_MVAr": np.stack([c["Qd"] for c in cenarios]),
        "mmgd_injetada_MW": np.stack([c["disponibilidade_mmgd"] for c in cenarios]),
        "capacidade_mmgd_MW": cenarios[0]["capacidade_mmgd"],
        "pg_MW": np.stack([r["Pg"] for r in resultados]),
        "qg_MVAr": np.stack([r["Qg"] for r in resultados]),
        "deficit_MW": np.stack([r["Pdef"] for r in resultados]),
        "vm_pu": np.stack([r["Vm"] for r in resultados]),
        "va_rad": np.stack([r["Va"] for r in resultados]),
        "taps": np.stack([r["taps"] for r in resultados]),
        "estado_q": np.stack([r["estado_q"] for r in resultados]),
        "estado_q_nr": np.stack([r["estado_q_nr"] for r in resultados]),
        "estado_tap": np.stack([r["estado_tap"] for r in resultados]),
        "perdas_MW": np.array([r["perdas_MW"] for r in resultados]),
        "fluxo_de_MW": np.stack([r["Pde"] for r in resultados]),
        "fluxo_para_MW": np.stack([r["Ppara"] for r in resultados]),
        "ocupacao_pct": np.stack([r["ocupacao"] for r in resultados]),
        "vm_nr_pu": np.stack([r["Vm_nr"] for r in resultados]),
        "va_nr_rad": np.stack([r["Va_nr"] for r in resultados]),
        "pg_nr_MW": np.stack([r["Pg_nr"] for r in resultados]),
        "qg_nr_MVAr": np.stack([r["Qg_nr"] for r in resultados]),
        "perdas_nr_MW": np.array([r["perdas_nr_MW"] for r in resultados]),
        "ocupacao_nr_pct": np.stack([r["ocupacao_nr"] for r in resultados]),
        "fob": np.array([r["fob"] for r in resultados]),
        "fob_desvio_pg": np.array([r["fob_desvio_pg"] for r in resultados]),
        "iteracoes_controle_reativo": np.array(
            [r["iteracoes_controle_q"] for r in resultados]
        ),
        "tempo_hora_s": np.array([r["tempo_s"] for r in resultados]),
        "numero_variaveis_fpo_horario": np.array(
            resultados[0]["numero_variaveis"]
        ),
        "numero_restricoes_fpo_horario": np.array(
            resultados[0]["numero_restricoes"]
        ),
        "numero_modelos_fpo_horario": np.array(len(resultados)),
        "erro_balanco_p_MW": np.array([r["erro_balanco_p_MW"] for r in resultados]),
        "erro_balanco_q_MVAr": np.array([r["erro_balanco_q_MVAr"] for r in resultados]),
        "erro_validacao_vm_pu": np.array([r["erro_validacao_vm_pu"] for r in resultados]),
        "erro_validacao_pg_MW": np.array([r["erro_validacao_pg_MW"] for r in resultados]),
        "tempo_total_s": np.array(time.time() - inicio),
    }
    vmax_operacional = np.array(
        [
            config.VMAX_BARRAS.get(int(barra), config.VMAX)
            for barra in dados["barras"]
        ]
    )
    dados["violacao_tensao"] = (
        (dados["vm_pu"] < config.VMIN)
        | (dados["vm_pu"] > vmax_operacional[None, :])
    )
    return dados


def salvar_resultados(dados, config):
    pasta = config.RESULTADOS / "etapa_01_fpo_ca_horario"
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / (
        f"{config.SISTEMA}_FPO_CA_"
        f"{int(dados['penetracao']):02d}pct.npz"
    )
    np.savez_compressed(arquivo, **dados)

    texto = "\n".join(
        [
            "FPO-CA PADRAO COM LIMITES OPERACIONAIS",
            "=" * 72,
            f"Penetracao de MMGD: {float(dados['penetracao']):.0f}%",
            f"Energia da carga: {dados['carga_MW'].sum():.6f} MWh",
            f"Energia da MMGD: {dados['mmgd_injetada_MW'].sum():.6f} MWh",
            f"Energia convencional: {dados['pg_MW'].sum():.6f} MWh",
            f"Perdas ativas: {dados['perdas_MW'].sum():.6f} MWh",
            f"Deficit: {dados['deficit_MW'].sum():.6f} MWh",
            f"Faixa de tensao: {dados['vm_pu'].min():.6f} a {dados['vm_pu'].max():.6f} pu",
            f"Violacoes de tensao: {np.count_nonzero(dados['violacao_tensao'])}",
            f"Horas com violacao de tensao: {np.count_nonzero(np.any(dados['violacao_tensao'], axis=1))}",
            f"Maior ocupacao: {dados['ocupacao_pct'].max():.6f}%",
            f"Erro maximo balanco P: {dados['erro_balanco_p_MW'].max():.3e} MW",
            f"Erro maximo balanco Q: {dados['erro_balanco_q_MVAr'].max():.3e} MVAr",
            f"Diferenca maxima FPO/NR em V: {dados['erro_validacao_vm_pu'].max():.3e} pu",
            f"Diferenca maxima FPO/NR em Pg: {dados['erro_validacao_pg_MW'].max():.3e} MW",
            f"Variaveis por FPO horario: {int(dados['numero_variaveis_fpo_horario'])}",
            f"Restricoes por FPO horario: {int(dados['numero_restricoes_fpo_horario'])}",
            f"FPOs horarios resolvidos: {int(dados['numero_modelos_fpo_horario'])}",
            f"Tempo total: {float(dados['tempo_total_s']):.3f} s",
            "Curtailment de MMGD: 0.000000 MWh",
        ]
    )
    relatorio = pasta / "relatorio.txt"
    relatorio.write_text(texto, encoding="utf-8")
    return arquivo, relatorio

