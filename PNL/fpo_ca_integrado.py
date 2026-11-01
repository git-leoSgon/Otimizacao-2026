import time

import numpy as np
import pyomo.environ as pyo

from fluxo_ca import preparar_controles_tap


def normalizar_estado_q(estado_q, dger, numero_barras):
    if estado_q is None:
        return np.zeros(len(dger), dtype=int)
    estado_q = np.asarray(estado_q, dtype=int)
    if estado_q.shape == (len(dger),):
        return estado_q.copy()
    if estado_q.shape == (numero_barras,):
        return np.array(
            [estado_q[int(dger.loc[g, "Barra"])] for g in range(len(dger))],
            dtype=int,
        )
    raise ValueError("O estado de controle reativo possui dimensao incorreta.")


def estado_q_por_barra(estado_q, dger, numero_barras):
    resultado = np.zeros(numero_barras, dtype=int)
    for g, estado in enumerate(np.asarray(estado_q, dtype=int)):
        resultado[int(dger.loc[g, "Barra"])] = int(estado)
    return resultado


def atualizar_estado_q(
    estado_q,
    qg,
    vm,
    dbar,
    dger,
    config,
    vm_referencia=None,
):
    estado_q = np.asarray(estado_q, dtype=int).copy()
    if vm_referencia is None:
        vm_referencia = np.array(
            [dbar.loc[int(dger.loc[g, "Barra"]), "Vm"] for g in range(len(dger))],
            dtype=float,
        )
    vm_referencia = np.asarray(vm_referencia, dtype=float)
    mudancas = []
    for g, estado in enumerate(estado_q):
        barra = int(dger.loc[g, "Barra"])
        vref = float(vm_referencia[g])
        qmin = float(dger.loc[g, "Qmin"])
        qmax = float(dger.loc[g, "Qmax"])
        if estado == 0 and qg[g] < qmin - config.TOLERANCIA_REATIVO_MVAR:
            mudancas.append(((qmin - qg[g]) / max(abs(qmin), 1.0), g, -1))
        elif estado == 0 and qg[g] > qmax + config.TOLERANCIA_REATIVO_MVAR:
            mudancas.append(((qg[g] - qmax) / max(abs(qmax), 1.0), g, 1))
    if not mudancas:
        return estado_q, False
    for _, gerador, novo_estado in mudancas:
        estado_q[gerador] = novo_estado
    return estado_q, True


def _dados_rede(dlin):
    controles = preparar_controles_tap(dlin)
    grupo_linha = {}
    for k, controle in enumerate(controles):
        for linha in controle["linhas"]:
            grupo_linha[int(linha)] = k

    linhas = []
    for l, linha in dlin.iterrows():
        y = 1.0 / complex(linha["R"], linha["X"])
        ysh = 1j * linha["Bsh"] / 2.0
        fase = float(linha["Phs"])
        linhas.append(
            {
                "de": int(linha["De"]),
                "para": int(linha["Para"]),
                "yff": y + ysh,
                "yft": -y * np.exp(1j * fase),
                "ytf": -y * np.exp(-1j * fase),
                "ytt": y + ysh,
                "tap": float(linha["Tap"]),
                "grupo": grupo_linha.get(int(l), -1),
                "limite": float(linha["Flim"]),
            }
        )
    return linhas, controles


def montar_modelo_horario(
    dbar,
    dlin,
    dger,
    cenario,
    config,
    partida,
    estado_q=None,
):
    nb = len(dbar)
    ng = len(dger)
    linhas, controles = _dados_rede(dlin)
    referencia = int(np.flatnonzero(dbar["Tipo"].to_numpy(int) == 2)[0])
    pd = cenario["Pd"]
    qd = cenario["Qd"]
    shunt = dbar["Sh_MVAr"].to_numpy(float)
    p_mmgd = np.zeros(nb)
    np.add.at(p_mmgd, cenario["barras_mmgd"], cenario["disponibilidade_mmgd"])
    fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
    estado_q = normalizar_estado_q(estado_q, dger, nb)
    vm_referencia = np.asarray(
        cenario.get(
            "vm_referencia_geradores",
            [dbar.loc[int(dger.loc[g, "Barra"]), "Vm"] for g in range(ng)],
        ),
        dtype=float,
    )

    geradores_barra = {
        barra: [g for g in range(ng) if int(dger.loc[g, "Barra"]) == barra]
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
    taps_partida = np.asarray(
        partida.get("taps", dlin["Tap"].to_numpy(float)),
        dtype=float,
    )
    modelo = pyo.ConcreteModel()
    modelo.B = pyo.RangeSet(0, nb - 1)
    modelo.G = pyo.RangeSet(0, ng - 1)
    modelo.L = pyo.RangeSet(0, len(dlin) - 1)
    modelo.C = pyo.RangeSet(0, len(controles) - 1)

    modelo.vm = pyo.Var(
        modelo.B,
        bounds=(0.0, None),
        initialize=lambda m, i: float(max(partida["Vm"][i], 1.0e-3)),
    )
    modelo.va = pyo.Var(
        modelo.B,
        initialize=lambda m, i: float(partida["Va"][i]),
    )
    modelo.pg = pyo.Var(
        modelo.G,
        bounds=lambda m, g: (
            float(dger.loc[g, "Pmin"]),
            float(dger.loc[g, "Pmax"]),
        ),
        initialize=lambda m, g: float(
            np.clip(partida["Pg"][g], dger.loc[g, "Pmin"], dger.loc[g, "Pmax"])
        ),
    )
    modelo.qg = pyo.Var(
        modelo.G,
        bounds=(None, None),
        initialize=lambda m, g: float(partida["Qg"][g]),
    )
    modelo.pdef = pyo.Var(
        modelo.B,
        bounds=lambda m, i: (0.0, float(pd[i])),
        initialize=0.0,
    )
    modelo.tap = pyo.Var(
        modelo.C,
        bounds=lambda m, k: (
            controles[k]["minimo"],
            controles[k]["maximo"],
        ),
        initialize=lambda m, k: float(
            taps_partida[controles[k]["linhas"][0]]
        ),
    )
    modelo.desvio_tap_positivo = pyo.Var(modelo.C, bounds=(0.0, None), initialize=0.0)
    modelo.desvio_tap_negativo = pyo.Var(modelo.C, bounds=(0.0, None), initialize=0.0)
    modelo.va[referencia].fix(0.0)
    for g in range(ng):
        barra = int(dger.loc[g, "Barra"])
        modelo.vm[barra].fix(float(vm_referencia[g]))

    def inverso_tap(m, l):
        grupo = linhas[l]["grupo"]
        if grupo >= 0:
            return 1.0 / m.tap[grupo]
        return 1.0 / linhas[l]["tap"]

    def fluxo_de(m, l):
        linha = linhas[l]
        de = linha["de"]
        para = linha["para"]
        a = inverso_tap(m, l)
        angulo = m.va[de] - m.va[para]
        return config.PB * (
            m.vm[de] ** 2 * linha["yff"].real * a ** 2
            + m.vm[de] * m.vm[para]
            * a
            * (
                linha["yft"].real * pyo.cos(angulo)
                + linha["yft"].imag * pyo.sin(angulo)
            )
        )

    def reativo_de(m, l):
        linha = linhas[l]
        de = linha["de"]
        para = linha["para"]
        a = inverso_tap(m, l)
        angulo = m.va[de] - m.va[para]
        return config.PB * (
            -m.vm[de] ** 2 * linha["yff"].imag * a ** 2
            + m.vm[de] * m.vm[para]
            * a
            * (
                linha["yft"].real * pyo.sin(angulo)
                - linha["yft"].imag * pyo.cos(angulo)
            )
        )

    def fluxo_para(m, l):
        linha = linhas[l]
        de = linha["de"]
        para = linha["para"]
        a = inverso_tap(m, l)
        ytt = linha["ytt"]
        angulo = m.va[para] - m.va[de]
        return config.PB * (
            m.vm[para] ** 2 * ytt.real
            + m.vm[para] * m.vm[de]
            * a
            * (
                linha["ytf"].real * pyo.cos(angulo)
                + linha["ytf"].imag * pyo.sin(angulo)
            )
        )

    def reativo_para(m, l):
        linha = linhas[l]
        de = linha["de"]
        para = linha["para"]
        a = inverso_tap(m, l)
        ytt = linha["ytt"]
        angulo = m.va[para] - m.va[de]
        return config.PB * (
            -m.vm[para] ** 2 * ytt.imag
            + m.vm[para] * m.vm[de]
            * a
            * (
                linha["ytf"].real * pyo.sin(angulo)
                - linha["ytf"].imag * pyo.cos(angulo)
            )
        )

    modelo.p_de = pyo.Expression(modelo.L, rule=fluxo_de)
    modelo.q_de = pyo.Expression(modelo.L, rule=reativo_de)
    modelo.p_para = pyo.Expression(modelo.L, rule=fluxo_para)
    modelo.q_para = pyo.Expression(modelo.L, rule=reativo_para)

    modelo.balanco_p = pyo.Constraint(
        modelo.B,
        rule=lambda m, i: sum(m.pg[g] for g in geradores_barra[i])
        + p_mmgd[i]
        - pd[i]
        + m.pdef[i]
        == sum(m.p_de[l] for l in linhas_de[i])
        + sum(m.p_para[l] for l in linhas_para[i]),
    )
    modelo.balanco_q = pyo.Constraint(
        modelo.B,
        rule=lambda m, i: sum(m.qg[g] for g in geradores_barra[i])
        - qd[i]
        + fator_q[i] * m.pdef[i]
        + shunt[i] * m.vm[i] ** 2
        == sum(m.q_de[l] for l in linhas_de[i])
        + sum(m.q_para[l] for l in linhas_para[i]),
    )
    modelo.limite_de = pyo.Constraint(
        modelo.L,
        rule=lambda m, l: m.p_de[l] ** 2 + m.q_de[l] ** 2
        <= linhas[l]["limite"] ** 2,
    )
    modelo.limite_para = pyo.Constraint(
        modelo.L,
        rule=lambda m, l: m.p_para[l] ** 2 + m.q_para[l] ** 2
        <= linhas[l]["limite"] ** 2,
    )
    modelo.alvo_tap = pyo.Constraint(
        modelo.C,
        rule=lambda m, k: m.vm[controles[k]["barra"]] - controles[k]["alvo"]
        == m.desvio_tap_positivo[k] - m.desvio_tap_negativo[k],
    )

    modelo.perdas = pyo.Expression(
        expr=sum(modelo.p_de[l] + modelo.p_para[l] for l in modelo.L)
    )
    modelo.indice_tensao = pyo.Expression(
        expr=sum((modelo.vm[i] - 1.0) ** 2 for i in modelo.B) / nb
    )
    modelo.indice_controle_tap = pyo.Expression(
        expr=sum(
            modelo.desvio_tap_positivo[k] + modelo.desvio_tap_negativo[k]
            for k in modelo.C
        )
        / max(len(controles), 1)
    )
    escala_carga = max(float(np.sum(pd)), 1.0)
    escala_custo_geracao = max(
        sum(
            float(dger.loc[g, "Custo"]) * float(dger.loc[g, "Pmax"])
            for g in range(ng)
        ),
        1.0,
    )
    escala_pg = np.maximum(dger["Pmax"].to_numpy(float), 1.0)
    pg_referencia = np.asarray(partida["Pg"], dtype=float)
    modelo.indice_desvio_pg = pyo.Expression(
        expr=sum(
            ((modelo.pg[g] - pg_referencia[g]) / escala_pg[g]) ** 2
            for g in modelo.G
        )
        / max(ng, 1)
    )
    modelo.objetivo = pyo.Objective(
        expr=config.PESO_DEFICIT
        * sum(modelo.pdef[i] for i in modelo.B)
        / escala_carga
        + config.PESO_PERDAS * modelo.perdas / escala_carga
        + config.PESO_CUSTO_GERACAO
        * sum(float(dger.loc[g, "Custo"]) * modelo.pg[g] for g in modelo.G)
        / escala_custo_geracao
        + getattr(config, "PESO_DESVIO_PG_BASE", 0.0)
        * modelo.indice_desvio_pg
        + config.PESO_TENSAO * modelo.indice_tensao
        + getattr(config, "PESO_CONTROLE_TAP", 0.0)
        * modelo.indice_controle_tap,
        sense=pyo.minimize,
    )
    return modelo, linhas, controles


def resolver_fpo_ca_horario(dbar, dlin, dger, cenario, config, partida, tee=False):
    solver = pyo.SolverFactory("ipopt")
    solver.options["tol"] = config.TOLERANCIA_IPOPT
    solver.options["acceptable_tol"] = config.TOLERANCIA_ACEITAVEL_IPOPT
    solver.options["acceptable_constr_viol_tol"] = config.TOLERANCIA_ACEITAVEL_IPOPT
    solver.options["max_iter"] = config.MAX_ITERACOES_IPOPT
    solver.options["print_level"] = 5 if tee else 0

    estado_q = np.zeros(len(dger), dtype=int)
    historico = {tuple(estado_q)}
    partida_iteracao = dict(partida)
    tempo = 0.0
    controle_convergiu = False
    reiniciou_em_pv = False
    for iteracao_controle in range(1, config.MAX_ITERACOES_PV_PQ + 1):
        modelo, linhas, controles = montar_modelo_horario(
            dbar,
            dlin,
            dger,
            cenario,
            config,
            partida_iteracao,
            estado_q=estado_q,
        )
        inicio = time.time()
        resultado = solver.solve(modelo, tee=tee)
        tempo += time.time() - inicio
        condicao = resultado.solver.termination_condition
        if condicao not in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
        }:
            if np.any(estado_q != 0) and not reiniciou_em_pv:
                estado_q = np.zeros(len(dger), dtype=int)
                historico = {tuple(estado_q)}
                partida_iteracao = dict(partida)
                reiniciou_em_pv = True
                continue
            raise RuntimeError(f"IPOPT terminou com a condicao: {condicao}")

        vm = np.array([pyo.value(modelo.vm[i]) for i in modelo.B])
        va = np.array([pyo.value(modelo.va[i]) for i in modelo.B])
        pg = np.array([pyo.value(modelo.pg[g]) for g in modelo.G])
        qg = np.array([pyo.value(modelo.qg[g]) for g in modelo.G])
        novo_estado, mudou = atualizar_estado_q(
            estado_q,
            qg,
            vm,
            dbar,
            dger,
            config,
            cenario.get("vm_referencia_geradores"),
        )
        if not mudou:
            controle_convergiu = True
            break
        assinatura = tuple(novo_estado)
        if assinatura in historico:
            raise RuntimeError("O controle PV/PQ entrou em ciclo.")
        historico.add(assinatura)
        estado_q = novo_estado
        partida_iteracao = {
            "Vm": vm,
            "Va": va,
            "Pg": pg,
            "Qg": qg,
            "estado_q": estado_q_por_barra(estado_q, dger, len(dbar)),
        }
        taps_iteracao = dlin["Tap"].to_numpy(float).copy()
        for k, controle in enumerate(controles):
            taps_iteracao[controle["linhas"]] = pyo.value(modelo.tap[k])
        partida_iteracao["taps"] = taps_iteracao
    if not controle_convergiu:
        raise RuntimeError("O controle PV/PQ nao convergiu.")

    pdef = np.array([pyo.value(modelo.pdef[i]) for i in modelo.B])
    pde = np.array([pyo.value(modelo.p_de[l]) for l in modelo.L])
    qde = np.array([pyo.value(modelo.q_de[l]) for l in modelo.L])
    ppara = np.array([pyo.value(modelo.p_para[l]) for l in modelo.L])
    qpara = np.array([pyo.value(modelo.q_para[l]) for l in modelo.L])
    taps = dlin["Tap"].to_numpy(float).copy()
    for k, controle in enumerate(controles):
        taps[controle["linhas"]] = pyo.value(modelo.tap[k])
    limites = dlin["Flim"].to_numpy(float)
    ocupacao = 100.0 * np.maximum(
        np.hypot(pde, qde),
        np.hypot(ppara, qpara),
    ) / limites
    numero_variaveis = sum(
        1 for _ in modelo.component_data_objects(pyo.Var, active=True)
    )
    numero_restricoes = sum(
        1 for _ in modelo.component_data_objects(pyo.Constraint, active=True)
    )
    erro_p = max(abs(pyo.value(modelo.balanco_p[i].body)) for i in modelo.B)
    erro_q = max(abs(pyo.value(modelo.balanco_q[i].body)) for i in modelo.B)
    return {
        "sucesso": True,
        "condicao": str(condicao),
        "tempo_s": tempo,
        "numero_variaveis": numero_variaveis,
        "numero_variaveis_binarias": 0,
        "numero_restricoes": numero_restricoes,
        "fob": pyo.value(modelo.objetivo),
        "fob_desvio_pg": getattr(config, "PESO_DESVIO_PG_BASE", 0.0)
        * pyo.value(modelo.indice_desvio_pg),
        "perdas_MW": pyo.value(modelo.perdas),
        "Vm": vm,
        "Va": va,
        "Pg": pg,
        "Qg": qg,
        "Pdef": pdef,
        "Pde": pde,
        "Qde": qde,
        "Ppara": ppara,
        "Qpara": qpara,
        "ocupacao": ocupacao,
        "taps": taps,
        "estado_q": estado_q_por_barra(estado_q, dger, len(dbar)),
        "iteracoes_controle_q": iteracao_controle,
        "controle_q_convergiu": controle_convergiu,
        "erro_balanco_p_MW": erro_p,
        "erro_balanco_q_MVAr": erro_q,
    }

