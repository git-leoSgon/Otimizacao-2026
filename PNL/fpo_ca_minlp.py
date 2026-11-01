import os
import time

import numpy as np
import pyomo.environ as pyo

from fpo_ca_integrado import _dados_rede, estado_q_por_barra


def _normalizar_estado_q(estado_q, dger, nh, nb):
    if estado_q is None:
        return np.zeros((nh, len(dger)), dtype=int)
    estado_q = np.asarray(estado_q, dtype=int)
    if estado_q.shape == (nh, len(dger)):
        return estado_q.copy()
    if estado_q.shape == (nh, nb):
        barras = dger["Barra"].to_numpy(int)
        return estado_q[:, barras].copy()
    raise ValueError("O estado multiperiodo de controle reativo possui dimensao incorreta.")


def montar_modelo_minlp(
    dbar,
    dlin,
    dger,
    cenarios,
    config,
    partida,
    modo_inicial=None,
    estado_q=None,
):
    nh = len(cenarios)
    nb = len(dbar)
    ng = len(dger)
    barras_bess = cenarios[0]["barras_mmgd"]
    nbat = len(barras_bess)
    linhas, controles = _dados_rede(dlin)
    referencia = int(np.flatnonzero(dbar["Tipo"].to_numpy(int) == 2)[0])
    pd = np.stack([cenario["Pd"] for cenario in cenarios])
    qd = np.stack([cenario["Qd"] for cenario in cenarios])
    shunt = dbar["Sh_MVAr"].to_numpy(float)
    disponibilidade = np.stack(
        [cenario["disponibilidade_mmgd"] for cenario in cenarios]
    )
    p_mmgd = np.zeros((nh, nb))
    for h in range(nh):
        np.add.at(p_mmgd[h], barras_bess, disponibilidade[h])
    fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
    capacidade_mmgd = cenarios[0]["capacidade_mmgd"]
    energia_maxima = config.FRACAO_BESS * capacidade_mmgd * config.DURACAO_BESS
    potencia_maxima = energia_maxima / config.DURACAO_BESS
    energia_inicial = config.SOC_INICIAL * energia_maxima
    rampa_bess = config.RAMPA_BESS * potencia_maxima
    estado_q = _normalizar_estado_q(estado_q, dger, nh, nb)
    vm_referencia = np.stack(
        [
            np.asarray(
                cenario.get(
                    "vm_referencia_geradores",
                    [
                        dbar.loc[int(dger.loc[g, "Barra"]), "Vm"]
                        for g in range(ng)
                    ],
                ),
                dtype=float,
            )
            for cenario in cenarios
        ]
    )
    geradores_barra = {
        barra: [g for g in range(ng) if int(dger.loc[g, "Barra"]) == barra]
        for barra in range(nb)
    }
    bateria_na_barra = {
        barra: [k for k in range(nbat) if int(barras_bess[k]) == barra]
        for barra in range(nb)
    }
    linhas_de = {
        barra: [l for l, linha in enumerate(linhas) if linha["de"] == barra]
        for barra in range(nb)
    }
    linhas_para = {
        barra: [l for l, linha in enumerate(linhas) if linha["para"] == barra]
        for barra in range(nb)
    }

    if modo_inicial is None:
        modo_inicial = np.zeros((nh, nbat))
        pliq_base = pd[:, barras_bess] - disponibilidade
        for k in range(nbat):
            modo_inicial[:, k] = (
                (disponibilidade[:, k] > 1.0e-8)
                & (pliq_base[:, k] <= np.mean(pliq_base[:, k]))
            )
    modo_inicial = np.asarray(modo_inicial, dtype=int)
    if modo_inicial.shape != (nh, nbat):
        raise ValueError("A partida binaria possui dimensao incorreta.")

    taps_partida = np.asarray(
        partida.get(
            "taps",
            np.tile(dlin["Tap"].to_numpy(float), (nh, 1)),
        ),
        dtype=float,
    )
    if taps_partida.ndim == 1:
        taps_partida = np.tile(taps_partida, (nh, 1))
    pg_referencia = np.asarray(
        partida.get("pg_referencia_MW", partida["pg_MW"]), dtype=float
    )
    if pg_referencia.shape != (nh, ng):
        raise ValueError("A referencia de Pg possui dimensao incorreta.")
    modelo = pyo.ConcreteModel()
    modelo.H = pyo.RangeSet(0, nh - 1)
    modelo.B = pyo.RangeSet(0, nb - 1)
    modelo.G = pyo.RangeSet(0, ng - 1)
    modelo.L = pyo.RangeSet(0, len(dlin) - 1)
    modelo.K = pyo.RangeSet(0, nbat - 1)
    modelo.C = pyo.RangeSet(0, len(controles) - 1)
    modelo.T = pyo.RangeSet(0, nh)

    modelo.vm = pyo.Var(
        modelo.H,
        modelo.B,
        bounds=(0.0, None),
        initialize=lambda m, h, i: float(max(partida["vm_pu"][h, i], 1.0e-3)),
    )
    modelo.va = pyo.Var(
        modelo.H,
        modelo.B,
        initialize=lambda m, h, i: float(partida["va_rad"][h, i]),
    )
    modelo.pg = pyo.Var(
        modelo.H,
        modelo.G,
        bounds=lambda m, h, g: (
            float(dger.loc[g, "Pmin"]),
            float(dger.loc[g, "Pmax"]),
        ),
        initialize=lambda m, h, g: float(np.clip(
            partida["pg_MW"][h, g],
            dger.loc[g, "Pmin"],
            dger.loc[g, "Pmax"],
        )),
    )
    modelo.qg = pyo.Var(
        modelo.H,
        modelo.G,
        bounds=(None, None),
        initialize=lambda m, h, g: float(partida["qg_MVAr"][h, g]),
    )
    modelo.pdef = pyo.Var(
        modelo.H,
        modelo.B,
        bounds=lambda m, h, i: (0.0, float(pd[h, i])),
        initialize=lambda m, h, i: float(
            partida.get("deficit_MW", np.zeros((nh, nb)))[h, i]
        ),
    )
    modelo.pch = pyo.Var(
        modelo.H,
        modelo.K,
        bounds=lambda m, h, k: (0.0, float(potencia_maxima[k])),
        initialize=lambda m, h, k: float(
            partida.get("carga_bess_MW", np.zeros((nh, nbat)))[h, k]
        ),
    )
    modelo.pdis = pyo.Var(
        modelo.H,
        modelo.K,
        bounds=lambda m, h, k: (0.0, float(potencia_maxima[k])),
        initialize=lambda m, h, k: float(
            partida.get("descarga_bess_MW", np.zeros((nh, nbat)))[h, k]
        ),
    )
    modelo.modo_carga = pyo.Var(
        modelo.H,
        modelo.K,
        domain=pyo.Binary,
        initialize=lambda m, h, k: int(modo_inicial[h, k]),
    )
    modelo.energia = pyo.Var(
        modelo.T,
        modelo.K,
        bounds=lambda m, t, k: (
            float(config.SOC_MIN * energia_maxima[k]),
            float(config.SOC_MAX * energia_maxima[k]),
        ),
        initialize=lambda m, t, k: float(
            np.clip(
                partida.get(
                    "soc_bess",
                    np.full((nh + 1, nbat), config.SOC_INICIAL),
                )[t, k]
                * energia_maxima[k],
                config.SOC_MIN * energia_maxima[k],
                config.SOC_MAX * energia_maxima[k],
            )
        ),
    )
    modelo.tap = pyo.Var(
        modelo.H,
        modelo.C,
        bounds=lambda m, h, k: (
            controles[k]["minimo"],
            controles[k]["maximo"],
        ),
        initialize=lambda m, h, k: float(
            taps_partida[h, controles[k]["linhas"][0]]
        ),
    )
    modelo.desvio_tap_positivo = pyo.Var(
        modelo.H, modelo.C, bounds=(0.0, None), initialize=0.0
    )
    modelo.desvio_tap_negativo = pyo.Var(
        modelo.H, modelo.C, bounds=(0.0, None), initialize=0.0
    )
    for h in range(nh):
        modelo.va[h, referencia].fix(0.0)
        for g in range(ng):
            barra = int(dger.loc[g, "Barra"])
            modelo.vm[h, barra].fix(float(vm_referencia[h, g]))
    for k in range(nbat):
        modelo.energia[0, k].fix(float(energia_inicial[k]))
        modelo.energia[nh, k].fix(float(energia_inicial[k]))

    def inverso_tap(m, h, l):
        grupo = linhas[l]["grupo"]
        if grupo >= 0:
            return 1.0 / m.tap[h, grupo]
        return 1.0 / linhas[l]["tap"]

    def fluxo_de(m, h, l):
        linha = linhas[l]
        de = linha["de"]
        para = linha["para"]
        a = inverso_tap(m, h, l)
        angulo = m.va[h, de] - m.va[h, para]
        return config.PB * (
            m.vm[h, de] ** 2 * linha["yff"].real * a ** 2
            + m.vm[h, de] * m.vm[h, para] * a
            * (
                linha["yft"].real * pyo.cos(angulo)
                + linha["yft"].imag * pyo.sin(angulo)
            )
        )

    def reativo_de(m, h, l):
        linha = linhas[l]
        de = linha["de"]
        para = linha["para"]
        a = inverso_tap(m, h, l)
        angulo = m.va[h, de] - m.va[h, para]
        return config.PB * (
            -m.vm[h, de] ** 2 * linha["yff"].imag * a ** 2
            + m.vm[h, de] * m.vm[h, para] * a
            * (
                linha["yft"].real * pyo.sin(angulo)
                - linha["yft"].imag * pyo.cos(angulo)
            )
        )

    def fluxo_para(m, h, l):
        linha = linhas[l]
        de = linha["de"]
        para = linha["para"]
        a = inverso_tap(m, h, l)
        angulo = m.va[h, para] - m.va[h, de]
        return config.PB * (
            m.vm[h, para] ** 2 * linha["ytt"].real
            + m.vm[h, para] * m.vm[h, de] * a
            * (
                linha["ytf"].real * pyo.cos(angulo)
                + linha["ytf"].imag * pyo.sin(angulo)
            )
        )

    def reativo_para(m, h, l):
        linha = linhas[l]
        de = linha["de"]
        para = linha["para"]
        a = inverso_tap(m, h, l)
        angulo = m.va[h, para] - m.va[h, de]
        return config.PB * (
            -m.vm[h, para] ** 2 * linha["ytt"].imag
            + m.vm[h, para] * m.vm[h, de] * a
            * (
                linha["ytf"].real * pyo.sin(angulo)
                - linha["ytf"].imag * pyo.cos(angulo)
            )
        )

    modelo.p_de = pyo.Expression(modelo.H, modelo.L, rule=fluxo_de)
    modelo.q_de = pyo.Expression(modelo.H, modelo.L, rule=reativo_de)
    modelo.p_para = pyo.Expression(modelo.H, modelo.L, rule=fluxo_para)
    modelo.q_para = pyo.Expression(modelo.H, modelo.L, rule=reativo_para)

    modelo.balanco_p = pyo.Constraint(
        modelo.H,
        modelo.B,
        rule=lambda m, h, i: sum(m.pg[h, g] for g in geradores_barra[i])
        + p_mmgd[h, i]
        + sum(m.pdis[h, k] - m.pch[h, k] for k in bateria_na_barra[i])
        - pd[h, i]
        + m.pdef[h, i]
        == sum(m.p_de[h, l] for l in linhas_de[i])
        + sum(m.p_para[h, l] for l in linhas_para[i]),
    )
    modelo.balanco_q = pyo.Constraint(
        modelo.H,
        modelo.B,
        rule=lambda m, h, i: sum(m.qg[h, g] for g in geradores_barra[i])
        - qd[h, i]
        + fator_q[h, i] * m.pdef[h, i]
        + shunt[i] * m.vm[h, i] ** 2
        == sum(m.q_de[h, l] for l in linhas_de[i])
        + sum(m.q_para[h, l] for l in linhas_para[i]),
    )
    modelo.limite_de = pyo.Constraint(
        modelo.H,
        modelo.L,
        rule=lambda m, h, l: m.p_de[h, l] ** 2 + m.q_de[h, l] ** 2
        <= linhas[l]["limite"] ** 2,
    )
    modelo.limite_para = pyo.Constraint(
        modelo.H,
        modelo.L,
        rule=lambda m, h, l: m.p_para[h, l] ** 2 + m.q_para[h, l] ** 2
        <= linhas[l]["limite"] ** 2,
    )
    modelo.alvo_tap = pyo.Constraint(
        modelo.H,
        modelo.C,
        rule=lambda m, h, k: m.vm[h, controles[k]["barra"]]
        - controles[k]["alvo"]
        == m.desvio_tap_positivo[h, k] - m.desvio_tap_negativo[h, k],
    )

    modelo.limite_carga = pyo.Constraint(
        modelo.H,
        modelo.K,
        rule=lambda m, h, k: m.pch[h, k]
        <= min(float(potencia_maxima[k]), float(disponibilidade[h, k]))
        * m.modo_carga[h, k],
    )
    modelo.limite_descarga = pyo.Constraint(
        modelo.H,
        modelo.K,
        rule=lambda m, h, k: m.pdis[h, k]
        <= float(potencia_maxima[k]) * (1.0 - m.modo_carga[h, k]),
    )
    modelo.balanco_energia = pyo.Constraint(
        modelo.H,
        modelo.K,
        rule=lambda m, h, k: m.energia[h + 1, k]
        == m.energia[h, k]
        + config.ETA_CARGA * m.pch[h, k]
        - m.pdis[h, k] / config.ETA_DESCARGA,
    )

    def potencia_bess(m, h, k):
        return m.pdis[h, k] - m.pch[h, k]

    modelo.rampa_bess_superior = pyo.Constraint(
        modelo.H,
        modelo.K,
        rule=lambda m, h, k: potencia_bess(m, h, k)
        - potencia_bess(m, (h - 1) % nh, k)
        <= float(rampa_bess[k]),
    )
    modelo.rampa_bess_inferior = pyo.Constraint(
        modelo.H,
        modelo.K,
        rule=lambda m, h, k: potencia_bess(m, h, k)
        - potencia_bess(m, (h - 1) % nh, k)
        >= -float(rampa_bess[k]),
    )
    modelo.rampa_gerador_superior = pyo.Constraint(
        modelo.H,
        modelo.G,
        rule=lambda m, h, g: m.pg[h, g] - m.pg[(h - 1) % nh, g]
        <= config.RAMPA_GERADORES * float(dger.loc[g, "Pmax"]),
    )
    modelo.rampa_gerador_inferior = pyo.Constraint(
        modelo.H,
        modelo.G,
        rule=lambda m, h, g: m.pg[h, g] - m.pg[(h - 1) % nh, g]
        >= -config.RAMPA_GERADORES * float(dger.loc[g, "Pmax"]),
    )

    modelo.pliq = pyo.Expression(
        modelo.H,
        modelo.K,
        rule=lambda m, h, k: pd[h, int(barras_bess[k])]
        - m.pdef[h, int(barras_bess[k])]
        - disponibilidade[h, k]
        + m.pch[h, k]
        - m.pdis[h, k],
    )
    modelo.pliq_media = pyo.Expression(
        modelo.K,
        rule=lambda m, k: sum(m.pliq[h, k] for h in m.H) / nh,
    )
    modelo.perdas = pyo.Expression(
        expr=sum(
            modelo.p_de[h, l] + modelo.p_para[h, l]
            for h in modelo.H
            for l in modelo.L
        )
    )
    escala_carga = max(float(np.sum(pd)), 1.0)
    escala_custo_geracao = max(
        nh
        * sum(
            float(dger.loc[g, "Custo"]) * float(dger.loc[g, "Pmax"])
            for g in range(ng)
        ),
        1.0,
    )
    escala_pliq = np.maximum(np.max(pd[:, barras_bess], axis=0), 1.0)
    escala_bess = np.maximum(potencia_maxima, 1.0)
    escala_pg = np.maximum(dger["Pmax"].to_numpy(float), 1.0)
    numero_amostras_bess = max(nh * nbat, 1)

    modelo.indice_deficit = pyo.Expression(
        expr=sum(modelo.pdef[h, i] for h in modelo.H for i in modelo.B)
        / escala_carga
    )
    modelo.indice_perdas = pyo.Expression(expr=modelo.perdas / escala_carga)
    modelo.indice_geracao = pyo.Expression(
        expr=sum(
            float(dger.loc[g, "Custo"]) * modelo.pg[h, g]
            for h in modelo.H
            for g in modelo.G
        )
        / escala_custo_geracao
    )
    modelo.indice_desvio_pg = pyo.Expression(
        expr=sum(
            (
                (modelo.pg[h, g] - pg_referencia[h, g])
                / escala_pg[g]
            ) ** 2
            for h in modelo.H
            for g in modelo.G
        )
        / (nh * ng)
    )
    modelo.indice_pliq = pyo.Expression(
        expr=sum(
            ((modelo.pliq[h, k] - modelo.pliq_media[k]) / escala_pliq[k]) ** 2
            for h in modelo.H
            for k in modelo.K
        )
        / numero_amostras_bess
    )
    modelo.indice_tensao = pyo.Expression(
        expr=sum((modelo.vm[h, i] - 1.0) ** 2 for h in modelo.H for i in modelo.B)
        / (nh * nb)
    )
    modelo.indice_controle_tap = pyo.Expression(
        expr=sum(
            modelo.desvio_tap_positivo[h, k] + modelo.desvio_tap_negativo[h, k]
            for h in modelo.H
            for k in modelo.C
        )
        / (nh * max(len(controles), 1))
    )
    modelo.indice_movimentacao = pyo.Expression(
        expr=sum(
            (modelo.pch[h, k] + modelo.pdis[h, k]) / escala_bess[k]
            for h in modelo.H
            for k in modelo.K
        )
        / numero_amostras_bess
    )
    modelo.indice_variacao = pyo.Expression(
        expr=sum(
            (
                (
                    potencia_bess(modelo, h, k)
                    - potencia_bess(modelo, (h - 1) % nh, k)
                )
                / escala_bess[k]
            ) ** 2
            for h in modelo.H
            for k in modelo.K
        )
        / numero_amostras_bess
    )
    modelo.custo_deficit = pyo.Expression(
        expr=config.PESO_DEFICIT * modelo.indice_deficit
    )
    modelo.custo_perdas = pyo.Expression(
        expr=config.PESO_PERDAS * modelo.indice_perdas
    )
    modelo.custo_geracao = pyo.Expression(
        expr=config.PESO_CUSTO_GERACAO * modelo.indice_geracao
    )
    modelo.custo_desvio_pg = pyo.Expression(
        expr=getattr(config, "PESO_DESVIO_PG", 0.0)
        * modelo.indice_desvio_pg
    )
    modelo.custo_pliq = pyo.Expression(
        expr=config.PESO_PLIQ * modelo.indice_pliq
    )
    modelo.custo_tensao = pyo.Expression(
        expr=config.PESO_TENSAO * modelo.indice_tensao
    )
    modelo.custo_controle_tap = pyo.Expression(
        expr=getattr(config, "PESO_CONTROLE_TAP", 0.0)
        * modelo.indice_controle_tap
    )
    modelo.custo_movimentacao = pyo.Expression(
        expr=config.PESO_MOVIMENTACAO_BESS * modelo.indice_movimentacao
    )
    modelo.custo_variacao = pyo.Expression(
        expr=config.PESO_VARIACAO_BESS * modelo.indice_variacao
    )
    modelo.objetivo = pyo.Objective(
        expr=modelo.custo_deficit
        + modelo.custo_perdas
        + modelo.custo_geracao
        + modelo.custo_desvio_pg
        + modelo.custo_pliq
        + modelo.custo_tensao
        + modelo.custo_controle_tap
        + modelo.custo_movimentacao
        + modelo.custo_variacao,
        sense=pyo.minimize,
    )
    return modelo, {
        "pd": pd,
        "qd": qd,
        "p_mmgd": p_mmgd,
        "disponibilidade": disponibilidade,
        "barras_bess": barras_bess,
        "energia_maxima": energia_maxima,
        "potencia_maxima": potencia_maxima,
        "pg_referencia": pg_referencia,
        "escala_pg": escala_pg,
        "estado_q": estado_q,
        "vm_referencia": vm_referencia,
        "linhas": linhas,
        "controles": controles,
    }


def resolver_fpo_ca_minlp(
    dbar,
    dlin,
    dger,
    cenarios,
    config,
    partida,
    modo_inicial=None,
    estado_q=None,
    estrategia_inicial="initial_binary",
    nome_partida="demanda_liquida",
    fixar_modo=False,
    fixar_bess_zero=False,
    tee=True,
):
    modelo, meta = montar_modelo_minlp(
        dbar,
        dlin,
        dger,
        cenarios,
        config,
        partida,
        modo_inicial=modo_inicial,
        estado_q=estado_q,
    )
    numero_variaveis = sum(
        1 for _ in modelo.component_data_objects(pyo.Var, active=True)
    )
    numero_variaveis_binarias = sum(
        1
        for variavel in modelo.component_data_objects(pyo.Var, active=True)
        if variavel.is_binary()
    )
    numero_restricoes = sum(
        1 for _ in modelo.component_data_objects(pyo.Constraint, active=True)
    )
    inicio = time.time()
    if fixar_modo:
        for h in modelo.H:
            for k in modelo.K:
                modelo.modo_carga[h, k].fix(int(modo_inicial[h, k]))
                if fixar_bess_zero:
                    modelo.pch[h, k].fix(0.0)
                    modelo.pdis[h, k].fix(0.0)
        solver = pyo.SolverFactory("ipopt")
        resultado = solver.solve(
            modelo,
            tee=tee,
            options={
                "tol": config.TOLERANCIA_IPOPT,
                "acceptable_tol": config.TOLERANCIA_ACEITAVEL_IPOPT,
                "acceptable_constr_viol_tol": config.TOLERANCIA_ACEITAVEL_IPOPT,
                "max_iter": config.MAX_ITERACOES_IPOPT,
                "print_level": 0,
            },
        )
    else:
        solver = pyo.SolverFactory("mindtpy")
        argumentos = {
            "strategy": "OA",
            "init_strategy": estrategia_inicial,
            "mip_solver": os.environ.get("FPO_MIP_SOLVER", "glpk"),
            "nlp_solver": "ipopt",
            "iteration_limit": config.MAX_ITERACOES_MINLP,
            "tee": tee,
            "nlp_solver_args": {
                "options": {
                    "tol": config.TOLERANCIA_IPOPT,
                    "acceptable_tol": config.TOLERANCIA_ACEITAVEL_IPOPT,
                    "acceptable_constr_viol_tol": config.TOLERANCIA_ACEITAVEL_IPOPT,
                    "max_iter": config.MAX_ITERACOES_IPOPT,
                    "print_level": 0,
                }
            },
        }
        argumentos["time_limit"] = (
            config.TEMPO_LIMITE_MINLP
            if config.TEMPO_LIMITE_MINLP is not None
            else 2147483647
        )
        resultado = solver.solve(modelo, **argumentos)
    tempo = time.time() - inicio
    condicao = resultado.solver.termination_condition
    if fixar_modo:
        limite_mestre = float(pyo.value(modelo.objetivo))
        melhor_fob = limite_mestre
        gap = 0.0
    else:
        limite_mestre = float(resultado.problem.lower_bound)
        melhor_fob = float(resultado.problem.upper_bound)
        gap = 100.0 * abs(melhor_fob - limite_mestre) / max(abs(melhor_fob), 1.0e-9)

    def matriz(var, primeiro, segundo):
        return np.array([[pyo.value(var[i, j]) for j in segundo] for i in primeiro])

    h = range(len(cenarios))
    b = range(len(dbar))
    g = range(len(dger))
    k = range(len(meta["barras_bess"]))
    l = range(len(dlin))
    c = range(len(meta["controles"]))
    t = range(len(cenarios) + 1)
    vm = matriz(modelo.vm, h, b)
    va = matriz(modelo.va, h, b)
    pg = matriz(modelo.pg, h, g)
    qg = matriz(modelo.qg, h, g)
    pdef = matriz(modelo.pdef, h, b)
    pch = matriz(modelo.pch, h, k)
    pdis = matriz(modelo.pdis, h, k)
    modo = matriz(modelo.modo_carga, h, k)
    energia = matriz(modelo.energia, t, k)
    pde = matriz(modelo.p_de, h, l)
    qde = matriz(modelo.q_de, h, l)
    ppara = matriz(modelo.p_para, h, l)
    qpara = matriz(modelo.q_para, h, l)
    tap_grupo = matriz(modelo.tap, h, c)
    taps = np.tile(dlin["Tap"].to_numpy(float), (len(cenarios), 1))
    for grupo, controle in enumerate(meta["controles"]):
        taps[:, controle["linhas"]] = tap_grupo[:, grupo, None]
    estado_q_barras = np.stack(
        [
            estado_q_por_barra(meta["estado_q"][hora], dger, len(dbar))
            for hora in h
        ]
    )
    limites = dlin["Flim"].to_numpy(float)
    ocupacao = 100.0 * np.maximum(
        np.hypot(pde, qde),
        np.hypot(ppara, qpara),
    ) / limites[None, :]
    erro_p = max(
        abs(pyo.value(modelo.balanco_p[hora, barra].body))
        for hora in modelo.H for barra in modelo.B
    )
    erro_q = max(
        abs(pyo.value(modelo.balanco_q[hora, barra].body))
        for hora in modelo.H for barra in modelo.B
    )
    componentes = {
        "deficit": pyo.value(modelo.custo_deficit),
        "perdas": pyo.value(modelo.custo_perdas),
        "geracao": pyo.value(modelo.custo_geracao),
        "desvio_pg": pyo.value(modelo.custo_desvio_pg),
        "pliq": pyo.value(modelo.custo_pliq),
        "tensao": pyo.value(modelo.custo_tensao),
        "controle_tap": pyo.value(modelo.custo_controle_tap),
        "movimentacao": pyo.value(modelo.custo_movimentacao),
        "variacao": pyo.value(modelo.custo_variacao),
    }
    desvio_pg = pg - meta["pg_referencia"]
    indice_desvio_pg = float(
        np.mean((desvio_pg / meta["escala_pg"][None, :]) ** 2)
    )
    fob_fisica = float(
        pyo.value(modelo.objetivo) - componentes["desvio_pg"]
    )
    maior_violacao = 0.0
    restricao_critica = ""
    maior_violacao_relativa = 0.0
    restricao_critica_relativa = ""
    for restricao in modelo.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        corpo = pyo.value(restricao.body, exception=False)
        if corpo is None:
            continue
        violacao = 0.0
        if restricao.has_lb():
            violacao = max(
                violacao,
                pyo.value(restricao.lower) - corpo,
            )
        if restricao.has_ub():
            violacao = max(
                violacao,
                corpo - pyo.value(restricao.upper),
            )
        if violacao > maior_violacao:
            maior_violacao = float(violacao)
            restricao_critica = restricao.name
        escala = max(
            abs(pyo.value(restricao.lower)) if restricao.has_lb() else 0.0,
            abs(pyo.value(restricao.upper)) if restricao.has_ub() else 0.0,
            1.0,
        )
        violacao_relativa = violacao / escala
        if violacao_relativa > maior_violacao_relativa:
            maior_violacao_relativa = float(violacao_relativa)
            restricao_critica_relativa = restricao.name
    solucao = {
        "sucesso": condicao in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
        },
        "condicao": str(condicao),
        "partida": nome_partida,
        "estrategia_inicial": estrategia_inicial,
        "limite_mestre": limite_mestre,
        "gap_mindtpy_pct": gap,
        "iteracoes_minlp": int(getattr(resultado.solver, "iterations", 0)),
        "tempo_s": tempo,
        "numero_variaveis": numero_variaveis,
        "numero_variaveis_binarias": numero_variaveis_binarias,
        "numero_restricoes": numero_restricoes,
        "fob": pyo.value(modelo.objetivo),
        "fob_fisica": fob_fisica,
        "componentes_fob": componentes,
        "indice_desvio_pg": indice_desvio_pg,
        "desvio_pg_max_MW": float(np.max(np.abs(desvio_pg))),
        "desvio_pg_medio_MW": float(np.mean(np.abs(desvio_pg))),
        "vm_pu": vm,
        "va_rad": va,
        "pg_MW": pg,
        "qg_MVAr": qg,
        "deficit_MW": pdef,
        "carga_bess_MW": pch,
        "descarga_bess_MW": pdis,
        "potencia_bess_MW": pdis - pch,
        "modo_carga": np.rint(modo).astype(int),
        "energia_bess_MWh": energia,
        "soc_bess": energia / meta["energia_maxima"][None, :],
        "potencia_bess_max_MW": meta["potencia_maxima"],
        "energia_bess_max_MWh": meta["energia_maxima"],
        "pliq_barras_MW": np.array(
            [[pyo.value(modelo.pliq[hora, bat]) for bat in k] for hora in h]
        ),
        "perdas_MW": np.sum(pde + ppara, axis=1),
        "fluxo_de_MW": pde,
        "fluxo_para_MW": ppara,
        "ocupacao_pct": ocupacao,
        "taps": taps,
        "estado_q": estado_q_barras,
        "erro_balanco_p_MW": erro_p,
        "erro_balanco_q_MVAr": erro_q,
        "erro_restricao_max": maior_violacao,
        "restricao_critica": restricao_critica,
        "erro_restricao_rel_max": maior_violacao_relativa,
        "restricao_critica_rel": restricao_critica_relativa,
        "p_mmgd_barra_MW": meta["p_mmgd"],
    }
    return solucao


def gerar_partidas(cenarios, nomes):
    disponibilidade = np.stack(
        [cenario["disponibilidade_mmgd"] for cenario in cenarios]
    )
    barras_bess = cenarios[0]["barras_mmgd"]
    carga_local = np.stack([cenario["Pd"][barras_bess] for cenario in cenarios])
    pliq = carga_local - disponibilidade
    maximo_solar = disponibilidade.max(axis=0)
    todas = {
        "demanda_liquida": (
            ((disponibilidade > 1.0e-8) & (pliq <= pliq.mean(axis=0)[None, :])).astype(int),
            "initial_binary",
        ),
        "solar_30": (
            ((disponibilidade > 1.0e-8) & (disponibilidade >= 0.30 * maximo_solar[None, :])).astype(int),
            "initial_binary",
        ),
        "solar_50": (
            ((disponibilidade > 1.0e-8) & (disponibilidade >= 0.50 * maximo_solar[None, :])).astype(int),
            "initial_binary",
        ),
    }
    return [(nome, *todas[nome]) for nome in nomes]


def resolver_fpo_ca_multistart(
    dbar,
    dlin,
    dger,
    cenarios,
    config,
    partida,
    tee=False,
):
    partidas = gerar_partidas(cenarios, config.PARTIDAS_MINLP)
    nh = len(cenarios)
    ng = len(dger)
    estado_q = np.zeros((nh, ng), dtype=int)
    if getattr(config, "REUTILIZAR_ESTADO_Q_PARTIDA", False):
        estado_q = _normalizar_estado_q(
            partida.get("estado_q"),
            dger,
            nh,
            len(dbar),
        )
    partida_controle = dict(partida)
    partida_controle["pg_referencia_MW"] = np.asarray(
        partida.get("pg_referencia_MW", partida["pg_MW"]), dtype=float
    ).copy()
    tempo_pre_solucao = 0.0
    estados_pq_pre_solucao = 0

    penetracao = float(cenarios[0].get("penetracao", 0.0))
    usar_pre_solucao = (
        getattr(config, "USAR_PRE_SOLUCAO_CONTINUA", False)
        and penetracao
        >= getattr(config, "PRE_SOLUCAO_PENETRACAO_MIN", 0.0)
    )
    if usar_pre_solucao:
        nome_pre, modo_pre, _ = partidas[0]
        estado_pre = np.zeros((nh, ng), dtype=int)
        partida_pre = dict(partida_controle)
        print("  Pre-solucao continua para aquecimento do FPO com BESS")
        for iteracao_pre in range(
            1, getattr(config, "MAX_ITERACOES_PRE_SOLUCAO", 10) + 1
        ):
            print(f"    Ajuste PV/PQ continuo {iteracao_pre}")
            solucao_pre = resolver_fpo_ca_minlp(
                dbar,
                dlin,
                dger,
                cenarios,
                config,
                partida_pre,
                modo_inicial=modo_pre,
                estado_q=estado_pre,
                estrategia_inicial="pre_solucao_continua",
                nome_partida=nome_pre,
                fixar_modo=True,
                tee=False,
            )
            tempo_pre_solucao += solucao_pre["tempo_s"]
            if (
                not solucao_pre["sucesso"]
                or solucao_pre["erro_balanco_p_MW"] > 1.0e-3
                or solucao_pre["erro_balanco_q_MVAr"] > 1.0e-3
            ):
                raise RuntimeError("A pre-solucao continua nao produziu solucao viavel.")

            mudancas_pre = []
            for h in range(nh):
                for g in range(ng):
                    if estado_pre[h, g] != 0:
                        continue
                    qg = solucao_pre["qg_MVAr"][h, g]
                    qmin = float(dger.loc[g, "Qmin"])
                    qmax = float(dger.loc[g, "Qmax"])
                    if qg < qmin - config.TOLERANCIA_REATIVO_MVAR:
                        mudancas_pre.append((h, g, -1))
                    elif qg > qmax + config.TOLERANCIA_REATIVO_MVAR:
                        mudancas_pre.append((h, g, 1))

            partida_pre.update(
                {
                    "vm_pu": solucao_pre["vm_pu"],
                    "va_rad": solucao_pre["va_rad"],
                    "pg_MW": solucao_pre["pg_MW"],
                    "qg_MVAr": solucao_pre["qg_MVAr"],
                    "deficit_MW": solucao_pre["deficit_MW"],
                    "carga_bess_MW": solucao_pre["carga_bess_MW"],
                    "descarga_bess_MW": solucao_pre["descarga_bess_MW"],
                    "energia_bess_MWh": solucao_pre["energia_bess_MWh"],
                    "taps": solucao_pre["taps"],
                }
            )
            if not mudancas_pre:
                break
            for h, g, novo_estado in mudancas_pre:
                estado_pre[h, g] = novo_estado
        else:
            raise RuntimeError("A pre-solucao PV/PQ nao convergiu.")

        estados_pq_pre_solucao = int(np.count_nonzero(estado_pre))
        partida_controle = partida_pre
        print(
            f"  Pre-solucao concluida em {tempo_pre_solucao:.1f}s | "
            f"estados PQ candidatos: {estados_pq_pre_solucao}"
        )

    tempo_total = tempo_pre_solucao
    tempo_minlp_binario = 0.0
    tempo_ajustes_pv_pq = 0.0
    otimizar_modos = getattr(config, "OTIMIZAR_MODOS_BESS", True)
    if otimizar_modos:
        modo_fixo = None
        diagnostico_minlp = None
        ranking_inicial = None
    else:
        nome_fixo, modo_fixo, _ = partidas[0]
        if usar_pre_solucao:
            estado_q = estado_pre.copy()
        diagnostico_minlp = {
            "partida": nome_fixo,
            "estrategia_inicial": "modo_bess_predefinido",
            "condicao": "modo_bess_predefinido",
            "gap_mindtpy_pct": 0.0,
            "iteracoes_minlp": 0,
        }
        ranking_inicial = {
            "ranking_partida": np.array([nome_fixo]),
            "ranking_condicao": np.array(["modo_bess_predefinido"]),
            "ranking_fob": np.array([np.nan]),
            "ranking_gap_pct": np.array([0.0]),
            "ranking_tempo_s": np.array([0.0]),
            "ranking_erro_p_MW": np.array([np.nan]),
            "ranking_erro_q_MVAr": np.array([np.nan]),
            "ranking_viavel": np.array([True]),
        }
        print(
            f"  Modos da BESS predefinidos pela partida {nome_fixo}; "
            "operacao continua resolvida pelo IPOPT."
        )

    for iteracao_controle in range(1, config.MAX_ITERACOES_PV_PQ + 1):
        print(
            f"  Controle PV/PQ {iteracao_controle}/"
            f"{config.MAX_ITERACOES_PV_PQ}"
        )
        resolver_binarios = modo_fixo is None
        if resolver_binarios:
            configuracoes = partidas
        else:
            configuracoes = [
                (
                    diagnostico_minlp["partida"],
                    modo_fixo,
                    "modo_bess_fixo",
                )
            ]
            print("  Modos da BESS fixados; ajuste continuo resolvido pelo IPOPT.")

        candidatos = []
        for numero, (nome, modo, estrategia) in enumerate(configuracoes, start=1):
            print(
                f"  Partida {numero}/{len(configuracoes)}: "
                f"{nome} ({estrategia})"
            )
            try:
                solucao = resolver_fpo_ca_minlp(
                    dbar,
                    dlin,
                    dger,
                    cenarios,
                    config,
                    partida_controle,
                    modo_inicial=modo,
                    estado_q=estado_q,
                    estrategia_inicial=estrategia,
                    nome_partida=nome,
                    fixar_modo=not resolver_binarios,
                    tee=tee,
                )
                viavel = (
                    solucao["sucesso"]
                    and solucao["erro_balanco_p_MW"] <= 1.0e-3
                    and solucao["erro_balanco_q_MVAr"] <= 1.0e-3
                )
                solucao["viavel_multistart"] = viavel
                candidatos.append(solucao)
                print(
                    f"    FOB={solucao['fob']:.6f} | "
                    f"gap={solucao['gap_mindtpy_pct']:.2f}% | "
                    f"condicao={solucao['condicao']} | "
                    f"erro P/Q={solucao['erro_balanco_p_MW']:.2e}/"
                    f"{solucao['erro_balanco_q_MVAr']:.2e} | "
                    f"tempo={solucao['tempo_s']:.1f}s | "
                    f"viavel={'sim' if viavel else 'nao'}"
                )
            except Exception as erro:
                candidatos.append(
                    {
                        "partida": nome,
                        "condicao": type(erro).__name__,
                        "fob": np.inf,
                        "gap_mindtpy_pct": np.inf,
                        "tempo_s": 0.0,
                        "erro_balanco_p_MW": np.inf,
                        "erro_balanco_q_MVAr": np.inf,
                        "viavel_multistart": False,
                    }
                )
                print(f"    falhou: {erro}")

        tempo_etapa = sum(c["tempo_s"] for c in candidatos)
        tempo_total += tempo_etapa
        if resolver_binarios:
            tempo_minlp_binario += tempo_etapa
        else:
            tempo_ajustes_pv_pq += tempo_etapa
        viaveis = [
            candidato
            for candidato in candidatos
            if candidato["viavel_multistart"]
        ]
        if not viaveis:
            metodo = "MINLP" if resolver_binarios else "NLP continuo"
            raise RuntimeError(
                f"Nenhuma partida do {metodo} produziu solucao viavel."
            )

        melhor = min(viaveis, key=lambda candidato: candidato["fob"])
        if resolver_binarios:
            modo_fixo = melhor["modo_carga"].copy()
            diagnostico_minlp = {
                "partida": melhor["partida"],
                "estrategia_inicial": melhor["estrategia_inicial"],
                "condicao": melhor["condicao"],
                "gap_mindtpy_pct": melhor["gap_mindtpy_pct"],
                "iteracoes_minlp": melhor["iteracoes_minlp"],
            }
            ordem = sorted(candidatos, key=lambda candidato: candidato["fob"])
            ranking_inicial = {
                "ranking_partida": np.array([c["partida"] for c in ordem]),
                "ranking_condicao": np.array([c["condicao"] for c in ordem]),
                "ranking_fob": np.array([c["fob"] for c in ordem], dtype=float),
                "ranking_gap_pct": np.array(
                    [c["gap_mindtpy_pct"] for c in ordem], dtype=float
                ),
                "ranking_tempo_s": np.array(
                    [c["tempo_s"] for c in ordem], dtype=float
                ),
                "ranking_erro_p_MW": np.array(
                    [c["erro_balanco_p_MW"] for c in ordem], dtype=float
                ),
                "ranking_erro_q_MVAr": np.array(
                    [c["erro_balanco_q_MVAr"] for c in ordem], dtype=float
                ),
                "ranking_viavel": np.array(
                    [c["viavel_multistart"] for c in ordem], dtype=bool
                ),
            }

        mudancas = []
        for h in range(nh):
            for g in range(ng):
                if estado_q[h, g] != 0:
                    continue
                qg = melhor["qg_MVAr"][h, g]
                qmin = float(dger.loc[g, "Qmin"])
                qmax = float(dger.loc[g, "Qmax"])
                if qg < qmin - config.TOLERANCIA_REATIVO_MVAR:
                    mudancas.append((h, g, -1))
                elif qg > qmax + config.TOLERANCIA_REATIVO_MVAR:
                    mudancas.append((h, g, 1))

        if not mudancas:
            melhor.update(ranking_inicial)
            melhor["partida"] = diagnostico_minlp["partida"]
            melhor["estrategia_inicial"] = diagnostico_minlp[
                "estrategia_inicial"
            ]
            melhor["condicao_minlp_inicial"] = diagnostico_minlp["condicao"]
            melhor["condicao_ajuste_final"] = melhor["condicao"]
            melhor["gap_mindtpy_pct"] = diagnostico_minlp[
                "gap_mindtpy_pct"
            ]
            melhor["iteracoes_minlp"] = diagnostico_minlp[
                "iteracoes_minlp"
            ]
            melhor["tempo_multistart_s"] = float(tempo_total)
            melhor["tempo_pre_solucao_s"] = float(tempo_pre_solucao)
            melhor["tempo_minlp_binario_s"] = float(tempo_minlp_binario)
            melhor["tempo_ajustes_pv_pq_s"] = float(tempo_ajustes_pv_pq)
            melhor["estados_pq_pre_solucao"] = estados_pq_pre_solucao
            melhor["modo_bess_fixado_apos_minlp"] = bool(otimizar_modos)
            melhor["modos_bess_otimizados"] = bool(otimizar_modos)
            melhor["iteracoes_controle_reativo"] = iteracao_controle
            melhor["controle_q_convergiu"] = True
            return melhor

        for h, g, novo_estado in mudancas:
            estado_q[h, g] = novo_estado
        print(f"    Novos limites reativos ativos: {len(mudancas)}")
        partida_controle.update(
            {
                "vm_pu": melhor["vm_pu"],
                "va_rad": melhor["va_rad"],
                "pg_MW": melhor["pg_MW"],
                "qg_MVAr": melhor["qg_MVAr"],
                "deficit_MW": melhor["deficit_MW"],
                "carga_bess_MW": melhor["carga_bess_MW"],
                "descarga_bess_MW": melhor["descarga_bess_MW"],
                "energia_bess_MWh": melhor["energia_bess_MWh"],
                "taps": melhor["taps"],
            }
        )

    raise RuntimeError("O controle PV/PQ do MINLP nao convergiu.")

