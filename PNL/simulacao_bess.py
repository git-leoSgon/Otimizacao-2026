import numpy as np

from dados_ca import carregar_sistema, montar_cenarios_24h
from fluxo_ca import preparar_controles_tap, resolver_fluxo
from fpo_ca_minlp import (
    resolver_fpo_ca_multistart,
)


def _estado_reativo(qg, dbar, dger):
    estado = np.zeros(len(dbar), dtype=int)
    for g, gerador in dger.iterrows():
        barra = int(gerador["Barra"])
        if qg[g] <= gerador["Qmin"] + 1.0e-2:
            estado[barra] = -1
        elif qg[g] >= gerador["Qmax"] - 1.0e-2:
            estado[barra] = 1
    return estado


def executar_caso_bess(config, penetracao, partida, tee=True):
    dbar, dlin, dger, carga, solar, mapa = carregar_sistema(config.ARQUIVO, config)
    cenarios = montar_cenarios_24h(
        dbar, carga, solar, mapa, config, penetracao
    )
    referencia_vm = np.asarray(
        partida.get(
            "vm_referencia_geradores",
            np.tile(
                [
                    config.TENSAO_GERADORES[int(barra)]
                    for barra in dger["Barra_original"].to_numpy(int)
                ],
                (24, 1),
            ),
        ),
        dtype=float,
    )
    for hora, cenario in enumerate(cenarios):
        cenario["vm_referencia_geradores"] = referencia_vm[hora]
    barras_bess = cenarios[0]["barras_mmgd"]
    controles_tap = preparar_controles_tap(dlin)
    partida_atual = dict(partida)

    solucao_inteira = resolver_fpo_ca_multistart(
        dbar, dlin, dger, cenarios, config, partida_atual, tee=tee
    )
    if not solucao_inteira["sucesso"]:
        raise RuntimeError(
            "O FPO com BESS terminou sem uma solucao aceita: "
            f"{solucao_inteira['condicao']}"
        )
    solucao = solucao_inteira
    if (
        not solucao["sucesso"]
        or solucao["erro_restricao_rel_max"] > 1.0e-3
        or solucao["erro_balanco_p_MW"] > 1.0e-3
        or solucao["erro_balanco_q_MVAr"] > 1.0e-3
    ):
        raise RuntimeError(
            "O FPO com BESS terminou sem uma solucao viavel: "
            f"{solucao['condicao']}; "
            f"restricao={solucao['restricao_critica_rel']} "
            f"({solucao['erro_restricao_rel_max']:.3e})."
        )
    solucao["tempo_minlp_inicial_s"] = solucao["tempo_minlp_binario_s"]
    solucao["tempo_minlp_final_s"] = 0.0

    estado_q = solucao["estado_q"]
    estado_tap = np.zeros((24, len(controles_tap)), dtype=int)
    for h in range(24):
        for k, controle in enumerate(controles_tap):
            tap = solucao["taps"][h, controle["linhas"][0]]
            if tap <= controle["minimo"] + config.TOLERANCIA_TAP:
                estado_tap[h, k] = -1
            elif tap >= controle["maximo"] - config.TOLERANCIA_TAP:
                estado_tap[h, k] = 1

    verificacoes = []
    for hora, cenario in enumerate(cenarios):
        injecao = solucao["p_mmgd_barra_MW"][hora].copy()
        np.add.at(injecao, barras_bess, solucao["potencia_bess_MW"][hora])
        dlin_hora = dlin.copy()
        dlin_hora["Tap"] = solucao["taps"][hora]
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
            solucao["pg_MW"][hora],
            vm_especificado=vm_especificado,
            p_mmgd=injecao,
            p_deficit=solucao["deficit_MW"][hora],
            vm_inicial=solucao["vm_pu"][hora],
            va_inicial=solucao["va_rad"][hora],
            estado_q_inicial=solucao["estado_q"][hora],
            atualizar_controles_q=False,
            tolerancia_q=config.TOLERANCIA_REATIVO_MVAR,
            max_iteracoes_controle_q=config.MAX_ITERACOES_PV_PQ,
            tolerancia=config.TOLERANCIA_NR,
            max_iteracoes=config.MAX_ITERACOES_NR,
        )
        if not verificacao["convergiu"] or not verificacao["controle_q_convergiu"]:
            raise RuntimeError(
                f"A verificacao CA com controle PV/PQ falhou na hora {hora}."
            )
        verificacoes.append(verificacao)

    vm_nr = np.stack([v["Vm"] for v in verificacoes])
    va_nr = np.stack([v["Va"] for v in verificacoes])
    pg_nr = np.stack([v["Pg"] for v in verificacoes])
    qg_nr = np.stack([v["Qg"] for v in verificacoes])
    erro_vm = float(np.max(np.abs(solucao["vm_pu"] - vm_nr)))
    erro_pg = float(np.max(np.abs(solucao["pg_MW"] - pg_nr)))
    solucao["erro_validacao_vm_pu"] = erro_vm
    solucao["erro_validacao_pg_MW"] = erro_pg
    solucao["tempo_controle_reativo_s"] = solucao["tempo_ajustes_pv_pq_s"]
    solucao["vm_nr_pu"] = vm_nr
    solucao["va_nr_rad"] = va_nr
    solucao["pg_nr_MW"] = pg_nr
    solucao["qg_nr_MVAr"] = qg_nr
    solucao["estado_q_nr"] = np.stack([v["estado_q"] for v in verificacoes])
    solucao["estado_q"] = estado_q
    solucao["estado_tap"] = estado_tap
    solucao["versao_solucao_bess"] = np.array(
        config.VERSAO_SOLUCAO_BESS
    )
    solucao["perdas_nr_MW"] = np.array(
        [v["perdas_ativas"] for v in verificacoes]
    )
    solucao["ocupacao_nr_pct"] = np.stack(
        [v["fluxos"]["ocupacao"] for v in verificacoes]
    )
    solucao.update(
        {
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
            "mmgd_injetada_MW": np.stack(
                [c["disponibilidade_mmgd"] for c in cenarios]
            ),
            "capacidade_mmgd_MW": cenarios[0]["capacidade_mmgd"],
            "numero_variaveis_fpo_horario": np.asarray(
                partida["numero_variaveis_fpo_horario"]
            ),
            "numero_restricoes_fpo_horario": np.asarray(
                partida["numero_restricoes_fpo_horario"]
            ),
            "numero_modelos_fpo_horario": np.asarray(
                partida["numero_modelos_fpo_horario"]
            ),
        }
    )
    vmax_operacional = np.array(
        [
            config.VMAX_BARRAS.get(int(barra), config.VMAX)
            for barra in solucao["barras"]
        ]
    )
    solucao["violacao_tensao"] = (
        (solucao["vm_pu"] < config.VMIN)
        | (solucao["vm_pu"] > vmax_operacional[None, :])
    )
    solucao.pop("p_mmgd_barra_MW")
    return solucao


def salvar_resultados(dados, config):
    pasta = config.RESULTADOS / "etapa_02_minlp_bess"
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / (
        f"{config.SISTEMA}_FPO_CA_MINLP_BESS_"
        f"{int(dados['penetracao']):02d}pct.npz"
    )
    componentes = dados.pop("componentes_fob")
    salvar = {
        chave: np.asarray(valor)
        for chave, valor in dados.items()
        if chave not in {"sucesso", "condicao", "viavel_multistart"}
    }
    for nome, valor in componentes.items():
        salvar[f"fob_{nome}"] = np.array(valor)
    salvar["condicao"] = np.array(dados["condicao"])
    np.savez_compressed(arquivo, **salvar)
    dados["componentes_fob"] = componentes

    simultaneidade = np.max(
        np.minimum(dados["carga_bess_MW"], dados["descarga_bess_MW"])
    )
    texto = "\n".join(
        [
            "FPO-CA MULTIPERIODO COM BESS",
            "=" * 72,
            f"Condicao do solver: {dados['condicao']}",
            f"Partida selecionada: {dados['partida']}",
            "Modos de carga/descarga otimizados: "
            f"{'sim' if dados['modos_bess_otimizados'] else 'nao'}",
            f"Gap informado pelo MindtPy: {dados['gap_mindtpy_pct']:.4f}%",
            f"Iteracoes MINLP: {dados['iteracoes_minlp']}",
            f"Variaveis do modelo multiperiodo: {dados['numero_variaveis']}",
            f"Variaveis binarias: {dados['numero_variaveis_binarias']}",
            f"Restricoes do modelo multiperiodo: {dados['numero_restricoes']}",
            f"Penetracao de MMGD: {float(dados['penetracao']):.0f}%",
            f"Energia da carga: {dados['carga_MW'].sum():.6f} MWh",
            f"Energia da MMGD: {dados['mmgd_injetada_MW'].sum():.6f} MWh",
            f"Energia convencional: {dados['pg_MW'].sum():.6f} MWh",
            f"Perdas ativas: {dados['perdas_MW'].sum():.6f} MWh",
            f"FOB fisica: {dados['fob_fisica']:.8f}",
            f"Parcela de regularizacao de Pg: {dados['componentes_fob']['desvio_pg']:.8f}",
            f"Maior desvio de Pg: {dados['desvio_pg_max_MW']:.6f} MW",
            f"Desvio medio absoluto de Pg: {dados['desvio_pg_medio_MW']:.6f} MW",
            f"Deficit: {dados['deficit_MW'].sum():.6f} MWh",
            f"Faixa de tensao: {dados['vm_pu'].min():.6f} a {dados['vm_pu'].max():.6f} pu",
            f"Violacoes de tensao: {np.count_nonzero(dados['violacao_tensao'])}",
            f"Horas com violacao de tensao: {np.count_nonzero(np.any(dados['violacao_tensao'], axis=1))}",
            f"Maior ocupacao: {dados['ocupacao_pct'].max():.6f}%",
            f"SOC: {dados['soc_bess'].min():.6f} a {dados['soc_bess'].max():.6f} pu",
            f"Carga e descarga simultaneas: {simultaneidade:.3e} MW",
            f"Erro maximo balanco P: {dados['erro_balanco_p_MW']:.3e} MW",
            f"Erro maximo balanco Q: {dados['erro_balanco_q_MVAr']:.3e} MVAr",
            f"Maior violacao relativa de restricao: {dados['erro_restricao_rel_max']:.3e}",
            f"Diferenca maxima FPO/NR em V: {dados['erro_validacao_vm_pu']:.3e} pu",
            f"Diferenca maxima FPO/NR em Pg: {dados['erro_validacao_pg_MW']:.3e} MW",
            f"FOB normalizada: {dados['fob']:.8f}",
            f"Tempo da etapa binaria: {dados['tempo_minlp_inicial_s']:.3f} s",
            f"Tempo da pre-solucao continua: {dados['tempo_pre_solucao_s']:.3f} s",
            f"Tempo dos ajustes PV/PQ com IPOPT: {dados['tempo_ajustes_pv_pq_s']:.3f} s",
            f"Tempo total da otimizacao: {dados['tempo_multistart_s']:.3f} s",
            "Curtailment de MMGD: 0.000000 MWh",
        ]
    )
    relatorio = pasta / "relatorio.txt"
    relatorio.write_text(texto, encoding="utf-8")
    return arquivo, relatorio

