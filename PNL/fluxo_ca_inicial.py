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
    tolerancia=1e-8,
    max_iteracoes=50,
):
    pb = 100.0
    nb = len(dbar)
    ybarra = montar_ybarra(dbar, dlin)
    tipos = dbar["Tipo"].astype(int).to_numpy().copy()
    referencia = int(np.flatnonzero(tipos == 2)[0])

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
    if vm_especificado is not None:
        for barra, valor in vm_especificado.items():
            vm[int(barra)] = float(valor)

    p_mmgd_barra = np.zeros(nb) if p_mmgd is None else np.asarray(p_mmgd, float)
    p_deficit = np.zeros(nb) if p_deficit is None else np.asarray(p_deficit, float)
    fator_q = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0.0)
    q_deficit = fator_q * p_deficit

    pg_barra = np.zeros(nb)
    qg_barra = np.zeros(nb)
    qmin = np.full(nb, -np.inf)
    qmax = np.full(nb, np.inf)
    for g, gerador in dger.iterrows():
        barra = int(gerador["Barra"])
        pg_barra[barra] += pg_especificado[g]
        qg_barra[barra] += gerador["Qg0"]
        qmin[barra] = gerador["Qmin"]
        qmax[barra] = gerador["Qmax"]

    convergiu = False
    maior_erro = np.inf
    limites_q_ativos = []

    for iteracao in range(1, max_iteracoes + 1):
        p, q = calcular_injecoes(vm, va, ybarra)

        barras_controladas = np.flatnonzero((tipos == 1) | (tipos == 2))
        for barra in barras_controladas:
            qg_calculado = q[barra] * pb + qd[barra] - q_deficit[barra]
            if qg_calculado < qmin[barra] - 1e-8:
                qg_barra[barra] = qmin[barra]
                tipos[barra] = 0
                if (barra, "Qmin") not in limites_q_ativos:
                    limites_q_ativos.append((barra, "Qmin"))
            elif qg_calculado > qmax[barra] + 1e-8:
                qg_barra[barra] = qmax[barra]
                tipos[barra] = 0
                if (barra, "Qmax") not in limites_q_ativos:
                    limites_q_ativos.append((barra, "Qmax"))

        barras_p = np.delete(np.arange(nb), referencia)
        barras_q = np.flatnonzero(tipos == 0)
        p_esp = (pg_barra + p_mmgd_barra - pd + p_deficit) / pb
        q_esp = (qg_barra - qd + q_deficit) / pb
        erro = np.r_[p_esp[barras_p] - p[barras_p], q_esp[barras_q] - q[barras_q]]
        maior_erro = np.max(np.abs(erro))

        if maior_erro <= tolerancia:
            convergiu = True
            break

        jacobiana = calcular_jacobiana(vm, va, ybarra, p, q, barras_p, barras_q)
        passo = np.linalg.solve(jacobiana, erro)
        va[barras_p] += passo[: len(barras_p)]
        vm[barras_q] += passo[len(barras_p) :]

    p, q = calcular_injecoes(vm, va, ybarra)
    pg_resultado = np.zeros(len(dger))
    qg_resultado = np.zeros(len(dger))
    for g, gerador in dger.iterrows():
        barra = int(gerador["Barra"])
        pg_resultado[g] = p[barra] * pb + pd[barra] - p_deficit[barra] - p_mmgd_barra[barra]
        qg_resultado[g] = q[barra] * pb + qd[barra] - q_deficit[barra]

    fluxos = calcular_fluxos(vm, va, dlin, pb)

    return {
        "convergiu": convergiu,
        "iteracoes": iteracao,
        "erro": maior_erro,
        "Vm": vm,
        "Va": va,
        "Pg": pg_resultado,
        "Qg": qg_resultado,
        "tipos_finais": tipos,
        "limites_q_ativos": limites_q_ativos,
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
    p_deficit = (
        np.zeros(len(dbar))
        if p_deficit is None
        else np.asarray(p_deficit, float)
    )
    pmin = dger["Pmin"].to_numpy(float, copy=True)
    pmax = dger["Pmax"].to_numpy(float, copy=True)
    # A potencia ativa da barra de referencia e resultado do fluxo, nao um
    # participante do rateio proporcional usado apenas para construir a partida.
    barra_referencia = int(np.flatnonzero(dbar["Tipo"].to_numpy(int) == 2)[0])
    geradores_referencia = dger["Barra"].to_numpy(int) == barra_referencia
    pmin[geradores_referencia] = 0.0
    pmax[geradores_referencia] = 0.0
    faixa = pmax - pmin
    alpha = (
        np.sum(pd)
        - np.sum(p_deficit)
        - np.sum(p_mmgd)
        - np.sum(pmin)
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
