import numpy as np


def montar_ybarra(dbar, dlin):
    nb = len(dbar)
    ybarra = np.zeros((nb, nb), dtype=complex)

    for _, linha in dlin.iterrows():
        de = int(linha["De"])
        para = int(linha["Para"])
        y = 1.0 / complex(linha["R"], linha["X"])
        ysh = 1j * linha["Bsh"] / 2.0
        tap = linha["Tap"] * np.exp(1j * linha["Phs"])

        ybarra[de, de] += (y + ysh) / abs(tap) ** 2
        ybarra[de, para] += -y / np.conj(tap)
        ybarra[para, de] += -y / tap
        ybarra[para, para] += y + ysh

    if "Sh_MVAr" in dbar:
        ybarra[np.diag_indices(nb)] += (
            1j * dbar["Sh_MVAr"].to_numpy(float) / 100.0
        )

    return ybarra


def calcular_injecoes(vm, va, ybarra):
    tensao = vm * np.exp(1j * va)
    potencia = tensao * np.conj(ybarra @ tensao)
    return potencia.real, potencia.imag


def calcular_jacobiana(vm, va, ybarra, p, q, barras_p, barras_q):
    g = ybarra.real
    b = ybarra.imag
    npv_pq = len(barras_p)
    npq = len(barras_q)
    h = np.zeros((npv_pq, npv_pq))
    n = np.zeros((npv_pq, npq))
    m = np.zeros((npq, npv_pq))
    l = np.zeros((npq, npq))

    for a, i in enumerate(barras_p):
        for c, j in enumerate(barras_p):
            if i == j:
                h[a, c] = -q[i] - b[i, i] * vm[i] ** 2
            else:
                ang = va[i] - va[j]
                h[a, c] = vm[i] * vm[j] * (
                    g[i, j] * np.sin(ang) - b[i, j] * np.cos(ang)
                )

        for c, j in enumerate(barras_q):
            if i == j:
                n[a, c] = p[i] / vm[i] + g[i, i] * vm[i]
            else:
                ang = va[i] - va[j]
                n[a, c] = vm[i] * (
                    g[i, j] * np.cos(ang) + b[i, j] * np.sin(ang)
                )

    for a, i in enumerate(barras_q):
        for c, j in enumerate(barras_p):
            if i == j:
                m[a, c] = p[i] - g[i, i] * vm[i] ** 2
            else:
                ang = va[i] - va[j]
                m[a, c] = -vm[i] * vm[j] * (
                    g[i, j] * np.cos(ang) + b[i, j] * np.sin(ang)
                )

        for c, j in enumerate(barras_q):
            if i == j:
                l[a, c] = q[i] / vm[i] - b[i, i] * vm[i]
            else:
                ang = va[i] - va[j]
                l[a, c] = vm[i] * (
                    g[i, j] * np.sin(ang) - b[i, j] * np.cos(ang)
                )

    return np.block([[h, n], [m, l]])


def _dados_geracao(dbar, dger, pg_especificado):
    nb = len(dbar)
    pg_barra = np.zeros(nb)
    qg_barra = np.zeros(nb)
    qmin = np.full(nb, -np.inf)
    qmax = np.full(nb, np.inf)
    barras_geradoras = []

    for g, gerador in dger.iterrows():
        barra = int(gerador["Barra"])
        pg_barra[barra] += pg_especificado[g]
        qg_barra[barra] += gerador["Qg0"]
        if barra not in barras_geradoras:
            barras_geradoras.append(barra)
            qmin[barra] = 0.0
            qmax[barra] = 0.0
        qmin[barra] += gerador["Qmin"]
        qmax[barra] += gerador["Qmax"]

    return pg_barra, qg_barra, qmin, qmax, np.array(barras_geradoras, dtype=int)


def _resolver_estado_fixo(
    dbar,
    dlin,
    dger,
    pd,
    qd,
    pg_especificado,
    tipos,
    qg_barra,
    vm_especificado,
    p_mmgd,
    p_deficit,
    vm_inicial,
    va_inicial,
    tolerancia,
    max_iteracoes,
):
    pb = 100.0
    nb = len(dbar)
    ybarra = montar_ybarra(dbar, dlin)
    referencia = int(np.flatnonzero(dbar["Tipo"].to_numpy(int) == 2)[0])
    vm = (
        dbar["Vm"].to_numpy(float).copy()
        if vm_inicial is None
        else np.asarray(vm_inicial, dtype=float).copy()
    )
    va = (
        dbar["Va"].to_numpy(float).copy()
        if va_inicial is None
        else np.asarray(va_inicial, dtype=float).copy()
    )
    for barra, valor in vm_especificado.items():
        if tipos[int(barra)] != 0:
            vm[int(barra)] = float(valor)

    fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
    q_deficit = fator_q * p_deficit
    pg_barra = np.zeros(nb)
    for g, gerador in dger.iterrows():
        pg_barra[int(gerador["Barra"])] += pg_especificado[g]

    barras_p = np.delete(np.arange(nb), referencia)
    barras_q = np.flatnonzero(tipos == 0)
    p_esp = (pg_barra + p_mmgd - pd + p_deficit) / pb
    q_esp = (qg_barra - qd + q_deficit) / pb
    maior_erro = np.inf
    convergiu = False

    for iteracao in range(1, max_iteracoes + 1):
        p, q = calcular_injecoes(vm, va, ybarra)
        erro = np.r_[p_esp[barras_p] - p[barras_p], q_esp[barras_q] - q[barras_q]]
        maior_erro = float(np.max(np.abs(erro)))
        if maior_erro <= tolerancia:
            convergiu = True
            break

        jacobiana = calcular_jacobiana(vm, va, ybarra, p, q, barras_p, barras_q)
        try:
            passo = np.linalg.solve(jacobiana, erro)
        except np.linalg.LinAlgError:
            break
        va[barras_p] += passo[: len(barras_p)]
        vm[barras_q] += passo[len(barras_p) :]

    p, q = calcular_injecoes(vm, va, ybarra)
    return convergiu, iteracao, maior_erro, vm, va, p, q


def _resultados_geradores(dbar, dger, pd, p_deficit, p_mmgd, p, q, qd, q_deficit):
    pg_resultado = np.zeros(len(dger))
    qg_resultado = np.zeros(len(dger))
    por_barra = {}
    for g, gerador in dger.iterrows():
        por_barra.setdefault(int(gerador["Barra"]), []).append(g)

    for barra, geradores in por_barra.items():
        pg_total = p[barra] * 100.0 + pd[barra] - p_deficit[barra] - p_mmgd[barra]
        qg_total = q[barra] * 100.0 + qd[barra] - q_deficit[barra]
        pesos = np.array([max(float(dger.loc[g, "Pmax"]), 1.0) for g in geradores])
        pesos /= pesos.sum()
        for peso, g in zip(pesos, geradores):
            pg_resultado[g] = pg_total * peso
            qg_resultado[g] = qg_total * peso
    return pg_resultado, qg_resultado


def resolver_fluxo(
    dbar,
    dlin,
    dger,
    pd,
    qd,
    pg_especificado,
    vm_especificado=None,
    p_mmgd=None,
    p_deficit=None,
    vm_inicial=None,
    va_inicial=None,
    estado_q_inicial=None,
    atualizar_controles_q=True,
    tolerancia_q=1.0e-3,
    tolerancia_retorno=2.0e-3,
    max_iteracoes_controle_q=30,
    tolerancia=1e-8,
    max_iteracoes=50,
):
    nb = len(dbar)
    tipos_originais = dbar["Tipo"].astype(int).to_numpy()
    p_mmgd = np.zeros(nb) if p_mmgd is None else np.asarray(p_mmgd, float)
    p_deficit = np.zeros(nb) if p_deficit is None else np.asarray(p_deficit, float)
    vm_especificado = {} if vm_especificado is None else dict(vm_especificado)
    for barra in np.flatnonzero(tipos_originais != 0):
        vm_especificado.setdefault(int(barra), float(dbar.loc[barra, "Vm"]))

    _, qg_base, qmin, qmax, barras_geradoras = _dados_geracao(
        dbar, dger, np.asarray(pg_especificado, float)
    )
    barras_pv = np.array(
        [barra for barra in barras_geradoras if tipos_originais[barra] in (1, 2)],
        dtype=int,
    )
    estado_q = np.zeros(nb, dtype=int)
    if estado_q_inicial is not None:
        recebido = np.asarray(estado_q_inicial, dtype=int)
        estado_q[barras_pv] = recebido[barras_pv]

    vm_partida = vm_inicial
    va_partida = va_inicial
    historico_estados = set()
    controle_q_convergiu = not atualizar_controles_q
    resultado_fixo = None

    for iteracao_controle in range(1, max_iteracoes_controle_q + 1):
        tipos = tipos_originais.copy()
        qg_barra = qg_base.copy()
        for barra in barras_pv:
            if estado_q[barra] == -1:
                tipos[barra] = 0
                qg_barra[barra] = qmin[barra]
            elif estado_q[barra] == 1:
                tipos[barra] = 0
                qg_barra[barra] = qmax[barra]

        resultado_fixo = _resolver_estado_fixo(
            dbar,
            dlin,
            dger,
            pd,
            qd,
            pg_especificado,
            tipos,
            qg_barra,
            vm_especificado,
            p_mmgd,
            p_deficit,
            vm_partida,
            va_partida,
            tolerancia,
            max_iteracoes,
        )
        convergiu, iteracoes, maior_erro, vm, va, p, q = resultado_fixo
        if not convergiu or not atualizar_controles_q:
            break

        fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
        q_deficit = fator_q * p_deficit
        qg_calculado = q * 100.0 + qd - q_deficit
        mudancas = []
        for barra in barras_pv:
            if estado_q[barra] == 0:
                if qg_calculado[barra] > qmax[barra] + tolerancia_q:
                    excesso = (qg_calculado[barra] - qmax[barra]) / max(abs(qmax[barra]), 1.0)
                    mudancas.append((excesso, barra, 1))
                elif qg_calculado[barra] < qmin[barra] - tolerancia_q:
                    excesso = (qmin[barra] - qg_calculado[barra]) / max(abs(qmin[barra]), 1.0)
                    mudancas.append((excesso, barra, -1))

        if not mudancas:
            controle_q_convergiu = True
            break

        for _, barra, novo_estado in mudancas:
            estado_q[barra] = novo_estado
        assinatura = tuple(estado_q[barras_pv])
        if assinatura in historico_estados:
            break
        historico_estados.add(assinatura)
        vm_partida = vm
        va_partida = va

    convergiu, iteracoes, maior_erro, vm, va, p, q = resultado_fixo
    fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
    q_deficit = fator_q * p_deficit
    pg_resultado, qg_resultado = _resultados_geradores(
        dbar, dger, pd, p_deficit, p_mmgd, p, q, qd, q_deficit
    )
    fluxos = calcular_fluxos(vm, va, dlin)
    limites_q_ativos = [
        (int(barra), "Qmin" if estado_q[barra] == -1 else "Qmax")
        for barra in barras_pv
        if estado_q[barra] != 0
    ]

    return {
        "convergiu": convergiu,
        "controle_q_convergiu": controle_q_convergiu,
        "iteracoes": iteracoes,
        "iteracoes_controle_q": iteracao_controle,
        "erro": maior_erro,
        "Vm": vm,
        "Va": va,
        "Pg": pg_resultado,
        "Qg": qg_resultado,
        "tipos_finais": tipos,
        "estado_q": estado_q,
        "limites_q_ativos": limites_q_ativos,
        "fluxos": fluxos,
        "perdas_ativas": np.sum(fluxos["Pde"] + fluxos["Ppara"]),
        "perdas_reativas": np.sum(fluxos["Qde"] + fluxos["Qpara"]),
    }


def resolver_fluxo_injecoes(
    dbar,
    dlin,
    dger,
    pd,
    qd,
    pg_especificado,
    qg_especificado,
    p_mmgd=None,
    p_deficit=None,
    vm_inicial=None,
    va_inicial=None,
    tolerancia=1.0e-8,
    max_iteracoes=50,
):
    nb = len(dbar)
    referencia = int(np.flatnonzero(dbar["Tipo"].to_numpy(int) == 2)[0])
    p_mmgd = np.zeros(nb) if p_mmgd is None else np.asarray(p_mmgd, float)
    p_deficit = np.zeros(nb) if p_deficit is None else np.asarray(p_deficit, float)
    vm = (
        dbar["Vm"].to_numpy(float).copy()
        if vm_inicial is None
        else np.asarray(vm_inicial, float).copy()
    )
    va = (
        dbar["Va"].to_numpy(float).copy()
        if va_inicial is None
        else np.asarray(va_inicial, float).copy()
    )

    fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
    q_deficit = fator_q * p_deficit
    pg_barra = np.zeros(nb)
    qg_barra = np.zeros(nb)
    for g, gerador in dger.iterrows():
        barra = int(gerador["Barra"])
        pg_barra[barra] += pg_especificado[g]
        qg_barra[barra] += qg_especificado[g]

    p_esp = (pg_barra + p_mmgd - pd + p_deficit) / 100.0
    q_esp = (qg_barra - qd + q_deficit) / 100.0
    barras_p = np.delete(np.arange(nb), referencia)
    barras_q = np.arange(nb)
    ybarra = montar_ybarra(dbar, dlin)
    convergiu = False
    maior_erro = np.inf

    for iteracao in range(1, max_iteracoes + 1):
        p, q = calcular_injecoes(vm, va, ybarra)
        erro = np.r_[
            p_esp[barras_p] - p[barras_p],
            q_esp[barras_q] - q[barras_q],
        ]
        maior_erro = float(np.max(np.abs(erro)))
        if maior_erro <= tolerancia:
            convergiu = True
            break

        jacobiana = calcular_jacobiana(
            vm, va, ybarra, p, q, barras_p, barras_q
        )
        try:
            passo = np.linalg.solve(jacobiana, erro)
        except np.linalg.LinAlgError:
            break
        va[barras_p] += passo[: len(barras_p)]
        vm[barras_q] += passo[len(barras_p) :]

    p, q = calcular_injecoes(vm, va, ybarra)
    pg_resultado, qg_resultado = _resultados_geradores(
        dbar, dger, pd, p_deficit, p_mmgd, p, q, qd, q_deficit
    )
    fluxos = calcular_fluxos(vm, va, dlin)
    return {
        "convergiu": convergiu,
        "iteracoes": iteracao,
        "erro": maior_erro,
        "Vm": vm,
        "Va": va,
        "Pg": pg_resultado,
        "Qg": qg_resultado,
        "fluxos": fluxos,
        "perdas_ativas": np.sum(fluxos["Pde"] + fluxos["Ppara"]),
        "perdas_reativas": np.sum(fluxos["Qde"] + fluxos["Qpara"]),
    }


def preparar_controles_tap(dlin):
    controles = []
    candidatos = dlin[
        (dlin["TapMin"] > 0.0)
        & (dlin["TapMax"] > 0.0)
        & (dlin["V_controle"] > 0.0)
    ]
    for (de, para), grupo in candidatos.groupby(["De_original", "Para_original"]):
        controles.append(
            {
                "de": int(de),
                "para": int(para),
                "linhas": grupo.index.to_numpy(int),
                "barra": int(grupo["Barra_controle"].iloc[0]),
                "alvo": float(grupo["V_controle"].iloc[0]),
                "orientacao": float(grupo["Orientacao_controle"].iloc[0]),
                "minimo": float(grupo["TapMin"].max()),
                "maximo": float(grupo["TapMax"].min()),
            }
        )
    return controles


def _aplicar_taps(dlin, taps):
    resultado = dlin.copy()
    resultado["Tap"] = np.asarray(taps, float)
    return resultado


def _derivada_potencias_a(vm, va, dlin, controle):
    nb = len(vm)
    derivada_y = np.zeros((nb, nb), dtype=complex)
    a = 1.0 / float(dlin.loc[controle["linhas"][0], "Tap"])

    for linha_idx in controle["linhas"]:
        linha = dlin.loc[linha_idx]
        de = int(linha["De"])
        para = int(linha["Para"])
        y = 1.0 / complex(linha["R"], linha["X"])
        ysh = 1j * linha["Bsh"] / 2.0
        derivada_y[de, de] += 2.0 * a * (y + ysh)
        derivada_y[de, para] += -y
        derivada_y[para, de] += -y

    tensao = vm * np.exp(1j * va)
    derivada_s = tensao * np.conj(derivada_y @ tensao)
    return derivada_s.real, derivada_s.imag


def _resolver_estado_fixo_com_taps(
    dbar,
    dlin,
    dger,
    pd,
    qd,
    pg_especificado,
    tipos,
    qg_barra,
    p_mmgd,
    p_deficit,
    vm_inicial,
    va_inicial,
    taps_iniciais,
    controles,
    config,
):
    nb = len(dbar)
    referencia = int(np.flatnonzero(dbar["Tipo"].to_numpy(int) == 2)[0])
    vm = (
        dbar["Vm"].to_numpy(float).copy()
        if vm_inicial is None
        else np.asarray(vm_inicial, dtype=float).copy()
    )
    va = (
        dbar["Va"].to_numpy(float).copy()
        if va_inicial is None
        else np.asarray(va_inicial, dtype=float).copy()
    )
    for barra in np.flatnonzero(tipos != 0):
        vm[barra] = float(dbar.loc[barra, "Vm"])

    taps = (
        dlin["Tap"].to_numpy(float).copy()
        if taps_iniciais is None
        else np.asarray(taps_iniciais, float).copy()
    )
    ativos = np.ones(len(controles), dtype=bool)
    fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
    q_deficit = fator_q * p_deficit
    pg_barra = np.zeros(nb)
    for g, gerador in dger.iterrows():
        pg_barra[int(gerador["Barra"])] += pg_especificado[g]

    barras_p = np.delete(np.arange(nb), referencia)
    barras_q = np.flatnonzero(tipos == 0)
    p_esp = (pg_barra + p_mmgd - pd + p_deficit) / 100.0
    q_esp = (qg_barra - qd + q_deficit) / 100.0
    convergiu = False
    maior_erro = np.inf

    partida = _resolver_estado_fixo(
        dbar,
        _aplicar_taps(dlin, taps),
        dger,
        pd,
        qd,
        pg_especificado,
        tipos,
        qg_barra,
        {},
        p_mmgd,
        p_deficit,
        vm,
        va,
        config.TOLERANCIA_NR,
        config.MAX_ITERACOES_NR,
    )
    if not partida[0]:
        return {
            "convergiu": False,
            "iteracoes": partida[1],
            "erro": partida[2],
            "Vm": partida[3],
            "Va": partida[4],
            "P": partida[5],
            "Q": partida[6],
            "taps": taps,
            "erro_tap": np.inf,
            "controles_tap_ativos": ativos,
        }
    vm = partida[3]
    va = partida[4]

    for iteracao in range(1, config.MAX_ITERACOES_NR + 1):
        dlin_atual = _aplicar_taps(dlin, taps)
        ybarra = montar_ybarra(dbar, dlin_atual)
        p, q = calcular_injecoes(vm, va, ybarra)
        ativos_idx = np.flatnonzero(ativos)
        erro_base = np.r_[
            p_esp[barras_p] - p[barras_p],
            q_esp[barras_q] - q[barras_q],
        ]
        erro_controle = np.array(
            [controles[idx]["alvo"] - vm[controles[idx]["barra"]] for idx in ativos_idx]
        )
        erro = np.r_[erro_base, erro_controle]
        maior_erro = float(np.max(np.abs(erro)))
        if maior_erro <= config.TOLERANCIA_NR:
            convergiu = True
            break

        jac_base = calcular_jacobiana(vm, va, ybarra, p, q, barras_p, barras_q)
        nbase = jac_base.shape[0]
        jac = np.zeros((nbase + len(ativos_idx), nbase + len(ativos_idx)))
        jac[:nbase, :nbase] = jac_base

        for posicao, idx in enumerate(ativos_idx):
            controle = controles[idx]
            dp_da, dq_da = _derivada_potencias_a(vm, va, dlin_atual, controle)
            jac[:nbase, nbase + posicao] = np.r_[
                dp_da[barras_p],
                dq_da[barras_q],
            ]
            barra = controle["barra"]
            posicoes_v = np.flatnonzero(barras_q == barra)
            if len(posicoes_v) == 0:
                ativos[idx] = False
                break
            jac[nbase + posicao, len(barras_p) + posicoes_v[0]] = 1.0
        else:
            try:
                passo = np.linalg.solve(jac, erro)
            except np.linalg.LinAlgError:
                break
            escala = 1.0
            if len(barras_p):
                escala = min(
                    escala,
                    0.20 / max(float(np.max(np.abs(passo[: len(barras_p)]))), 0.20),
                )
            if len(barras_q):
                escala = min(
                    escala,
                    0.05
                    / max(
                        float(np.max(np.abs(passo[len(barras_p) : nbase]))),
                        0.05,
                    ),
                )
            if len(ativos_idx):
                escala = min(
                    escala,
                    0.01
                    / max(float(np.max(np.abs(passo[nbase:]))), 0.01),
                )
            passo *= escala
            va[barras_p] += passo[: len(barras_p)]
            vm[barras_q] += passo[len(barras_p) : nbase]
            for posicao, idx in enumerate(ativos_idx):
                controle = controles[idx]
                linha = controle["linhas"][0]
                a_atual = 1.0 / taps[linha]
                a_min = 1.0 / controle["maximo"]
                a_max = 1.0 / controle["minimo"]
                a_novo = float(np.clip(
                    a_atual + passo[nbase + posicao],
                    a_min,
                    a_max,
                ))
                tap_novo = 1.0 / a_novo
                taps[controle["linhas"]] = tap_novo
                if (
                    (a_novo == a_min and passo[nbase + posicao] < 0.0)
                    or (a_novo == a_max and passo[nbase + posicao] > 0.0)
                ):
                    ativos[idx] = False
            continue
        continue

    dlin_atual = _aplicar_taps(dlin, taps)
    ybarra = montar_ybarra(dbar, dlin_atual)
    p, q = calcular_injecoes(vm, va, ybarra)
    erros_tap = np.array(
        [item["alvo"] - vm[item["barra"]] for item in controles],
        dtype=float,
    )
    return {
        "convergiu": convergiu,
        "iteracoes": iteracao,
        "erro": maior_erro,
        "Vm": vm,
        "Va": va,
        "P": p,
        "Q": q,
        "taps": taps,
        "erro_tap": float(np.max(np.abs(erros_tap))) if len(erros_tap) else 0.0,
        "controles_tap_ativos": ativos,
    }


def resolver_fluxo_controlado(
    dbar,
    dlin,
    dger,
    pd,
    qd,
    pg_especificado,
    config,
    p_mmgd=None,
    p_deficit=None,
    vm_inicial=None,
    va_inicial=None,
    taps_iniciais=None,
    estado_q_inicial=None,
):
    controles = preparar_controles_tap(dlin)
    nb = len(dbar)
    tipos_originais = dbar["Tipo"].astype(int).to_numpy()
    p_mmgd = np.zeros(nb) if p_mmgd is None else np.asarray(p_mmgd, float)
    p_deficit = np.zeros(nb) if p_deficit is None else np.asarray(p_deficit, float)
    _, qg_base, qmin, qmax, barras_geradoras = _dados_geracao(
        dbar, dger, np.asarray(pg_especificado, float)
    )
    barras_pv = np.array(
        [barra for barra in barras_geradoras if tipos_originais[barra] in (1, 2)],
        dtype=int,
    )
    estado_q = np.zeros(nb, dtype=int)
    if estado_q_inicial is not None:
        estado_q[barras_pv] = np.asarray(estado_q_inicial, int)[barras_pv]
    taps = taps_iniciais
    vm_partida = vm_inicial
    va_partida = va_inicial
    historico_estados = set()
    controle_q_convergiu = False
    resultado = None

    for iteracao_controle in range(1, config.MAX_ITERACOES_PV_PQ + 1):
        tipos = tipos_originais.copy()
        qg_barra = qg_base.copy()
        for barra in barras_pv:
            if estado_q[barra] == -1:
                tipos[barra] = 0
                qg_barra[barra] = qmin[barra]
            elif estado_q[barra] == 1:
                tipos[barra] = 0
                qg_barra[barra] = qmax[barra]

        resultado = _resolver_estado_fixo_com_taps(
            dbar,
            dlin,
            dger,
            pd,
            qd,
            pg_especificado,
            tipos,
            qg_barra,
            p_mmgd,
            p_deficit,
            vm_partida,
            va_partida,
            taps,
            controles,
            config,
        )
        if not resultado["convergiu"]:
            break

        fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
        q_deficit = fator_q * p_deficit
        qg_calculado = resultado["Q"] * 100.0 + qd - q_deficit
        mudancas = []
        for barra in barras_pv:
            if estado_q[barra] == 0:
                if qg_calculado[barra] > qmax[barra] + config.TOLERANCIA_REATIVO_MVAR:
                    mudancas.append((
                        (qg_calculado[barra] - qmax[barra]) / max(abs(qmax[barra]), 1.0),
                        barra,
                        1,
                    ))
                elif qg_calculado[barra] < qmin[barra] - config.TOLERANCIA_REATIVO_MVAR:
                    mudancas.append((
                        (qmin[barra] - qg_calculado[barra]) / max(abs(qmin[barra]), 1.0),
                        barra,
                        -1,
                    ))

        if not mudancas:
            controle_q_convergiu = True
            break
        for _, barra, novo_estado in mudancas:
            estado_q[barra] = novo_estado
        assinatura = tuple(estado_q[barras_pv])
        if assinatura in historico_estados:
            break
        historico_estados.add(assinatura)
        taps = resultado["taps"]
        vm_partida = resultado["Vm"]
        va_partida = resultado["Va"]

    fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
    q_deficit = fator_q * p_deficit
    pg_resultado, qg_resultado = _resultados_geradores(
        dbar,
        dger,
        pd,
        p_deficit,
        p_mmgd,
        resultado["P"],
        resultado["Q"],
        qd,
        q_deficit,
    )
    dlin_final = _aplicar_taps(dlin, resultado["taps"])
    fluxos = calcular_fluxos(resultado["Vm"], resultado["Va"], dlin_final)
    limites_q_ativos = [
        (int(barra), "Qmin" if estado_q[barra] == -1 else "Qmax")
        for barra in barras_pv
        if estado_q[barra] != 0
    ]
    return {
        "convergiu": resultado["convergiu"],
        "controle_q_convergiu": controle_q_convergiu,
        "controle_tap_convergiu": (
            resultado["erro_tap"] <= config.TOLERANCIA_TAP
        ),
        "iteracoes": resultado["iteracoes"],
        "iteracoes_controle_q": iteracao_controle,
        "iteracoes_controle_tap": resultado["iteracoes"],
        "erro": resultado["erro"],
        "erro_controle_tap_pu": resultado["erro_tap"],
        "Vm": resultado["Vm"],
        "Va": resultado["Va"],
        "Pg": pg_resultado,
        "Qg": qg_resultado,
        "tipos_finais": tipos,
        "estado_q": estado_q,
        "limites_q_ativos": limites_q_ativos,
        "taps": resultado["taps"],
        "barras_controle_tap": np.array(
            [item["barra"] for item in controles], dtype=int
        ),
        "alvos_controle_tap_pu": np.array(
            [item["alvo"] for item in controles], dtype=float
        ),
        "fluxos": fluxos,
        "perdas_ativas": np.sum(fluxos["Pde"] + fluxos["Ppara"]),
        "perdas_reativas": np.sum(fluxos["Qde"] + fluxos["Qpara"]),
    }


def resolver_fluxo_proporcional(
    dbar,
    dlin,
    dger,
    pd,
    qd,
    p_mmgd=None,
    p_deficit=None,
    vm_especificado=None,
    vm_inicial=None,
    va_inicial=None,
    tolerancia=1e-8,
    max_iteracoes=50,
):
    p_mmgd = np.zeros(len(dbar)) if p_mmgd is None else np.asarray(p_mmgd, float)
    p_deficit = np.zeros(len(dbar)) if p_deficit is None else np.asarray(p_deficit, float)
    pmin = dger["Pmin"].to_numpy(float)
    pmax = dger["Pmax"].to_numpy(float)
    faixa = pmax - pmin
    alpha = (
        np.sum(pd) - np.sum(p_deficit) - np.sum(p_mmgd) - np.sum(pmin)
    ) / np.sum(faixa)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    resultado = None

    for _ in range(max_iteracoes):
        pg = pmin + alpha * faixa
        resultado = resolver_fluxo(
            dbar,
            dlin,
            dger,
            pd,
            qd,
            pg,
            vm_especificado=vm_especificado,
            p_mmgd=p_mmgd,
            p_deficit=p_deficit,
            vm_inicial=vm_inicial,
            va_inicial=va_inicial,
            tolerancia=tolerancia,
            max_iteracoes=max_iteracoes,
        )
        if not resultado["convergiu"]:
            return resultado
        alpha_novo = (np.sum(resultado["Pg"]) - np.sum(pmin)) / np.sum(faixa)
        if abs(alpha_novo - alpha) <= tolerancia:
            resultado["alpha"] = float(alpha_novo)
            return resultado
        alpha = float(np.clip(alpha_novo, 0.0, 1.0))

    resultado["alpha"] = alpha
    return resultado


def resolver_fluxo_proporcional_controlado(
    dbar,
    dlin,
    dger,
    pd,
    qd,
    config,
    p_mmgd=None,
    p_deficit=None,
    vm_inicial=None,
    va_inicial=None,
    taps_iniciais=None,
):
    p_mmgd = np.zeros(len(dbar)) if p_mmgd is None else np.asarray(p_mmgd, float)
    p_deficit = np.zeros(len(dbar)) if p_deficit is None else np.asarray(p_deficit, float)
    pmin = dger["Pmin"].to_numpy(float)
    pmax = dger["Pmax"].to_numpy(float)
    faixa = pmax - pmin
    alpha = float(np.clip(
        (np.sum(pd) - np.sum(p_deficit) - np.sum(p_mmgd) - np.sum(pmin))
        / np.sum(faixa),
        0.0,
        1.0,
    ))
    resultado = None
    taps = taps_iniciais
    estado_q = None

    for _ in range(12):
        pg = pmin + alpha * faixa
        resultado = resolver_fluxo_controlado(
            dbar,
            dlin,
            dger,
            pd,
            qd,
            pg,
            config,
            p_mmgd=p_mmgd,
            p_deficit=p_deficit,
            vm_inicial=vm_inicial if resultado is None else resultado["Vm"],
            va_inicial=va_inicial if resultado is None else resultado["Va"],
            taps_iniciais=taps,
            estado_q_inicial=estado_q,
        )
        if not resultado["convergiu"]:
            return resultado
        alpha_novo = float(np.clip(
            (np.sum(resultado["Pg"]) - np.sum(pmin)) / np.sum(faixa),
            0.0,
            1.0,
        ))
        if abs(alpha_novo - alpha) <= config.TOLERANCIA_NR:
            resultado["alpha"] = alpha_novo
            return resultado
        alpha = alpha_novo
        taps = resultado["taps"]
        estado_q = resultado["estado_q"]

    resultado["alpha"] = alpha
    return resultado


def buscar_fluxo_inicial(
    dbar,
    dlin,
    dger,
    pd,
    qd,
    p_mmgd,
    vmin,
    vmax,
    tolerancia=1e-8,
    max_iteracoes=50,
):
    barras_geradores = dger["Barra"].astype(int).to_numpy()
    tensoes_base = dbar.loc[barras_geradores, "Vm"].to_numpy(float)
    melhor = None

    for reducao in np.arange(0.0, 0.101, 0.005):
        tensoes = np.clip(tensoes_base - reducao, vmin, vmax)
        especificadas = dict(zip(barras_geradores, tensoes))
        resultado = resolver_fluxo_proporcional(
            dbar,
            dlin,
            dger,
            pd,
            qd,
            p_mmgd=p_mmgd,
            vm_especificado=especificadas,
            tolerancia=tolerancia,
            max_iteracoes=max_iteracoes,
        )
        if not resultado["convergiu"]:
            continue
        melhor = resultado
        if resultado["Vm"].min() >= vmin and resultado["Vm"].max() <= vmax:
            return resultado

    return melhor


def calcular_fluxos(vm, va, dlin, pb=100.0):
    tensao = vm * np.exp(1j * va)
    pde, qde, ppara, qpara, sde, spara = [], [], [], [], [], []

    for _, linha in dlin.iterrows():
        de = int(linha["De"])
        para = int(linha["Para"])
        y = 1.0 / complex(linha["R"], linha["X"])
        ysh = 1j * linha["Bsh"] / 2.0
        tap = linha["Tap"] * np.exp(1j * linha["Phs"])

        yff = (y + ysh) / abs(tap) ** 2
        yft = -y / np.conj(tap)
        ytf = -y / tap
        ytt = y + ysh

        ide = yff * tensao[de] + yft * tensao[para]
        ipara = ytf * tensao[de] + ytt * tensao[para]
        potencia_de = tensao[de] * np.conj(ide) * pb
        potencia_para = tensao[para] * np.conj(ipara) * pb

        pde.append(potencia_de.real)
        qde.append(potencia_de.imag)
        ppara.append(potencia_para.real)
        qpara.append(potencia_para.imag)
        sde.append(abs(potencia_de))
        spara.append(abs(potencia_para))

    limite = dlin["Flim"].to_numpy(float)
    return {
        "Pde": np.asarray(pde),
        "Qde": np.asarray(qde),
        "Ppara": np.asarray(ppara),
        "Qpara": np.asarray(qpara),
        "Sde": np.asarray(sde),
        "Spara": np.asarray(spara),
        "ocupacao": 100.0 * np.maximum(sde, spara) / limite,
    }

