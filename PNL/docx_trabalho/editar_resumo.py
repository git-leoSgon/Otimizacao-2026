from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import shutil

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Resumo_estendido_PNL.docx"
WORK = ROOT / "docx_trabalho" / "Resumo_estendido_PNL_editado.docx"
FINAL = ROOT / "Resumo_estendido_PNL.docx"


def set_text(p, text, bold=None, italic=False):
    p.clear()
    r = p.add_run(text)
    if bold is not None:
        r.bold = bold
    if italic is not None:
        r.italic = italic
    return p


def remove_row(table, index):
    row = table.rows[index]
    table._tbl.remove(row._tr)


def fit_png(src, size=(3253, 1984)):
    im = Image.open(src).convert("RGB")
    canvas = im.resize(size, Image.Resampling.LANCZOS)
    out = BytesIO()
    canvas.save(out, "PNG", optimize=True)
    return out.getvalue()


def span_row(row, text):
    cells = row.cells
    tc0 = cells[0]._tc
    tcPr = tc0.get_or_add_tcPr()
    gridSpan = tcPr.find(qn("w:gridSpan"))
    if gridSpan is None:
        from docx.oxml import OxmlElement
        gridSpan = OxmlElement("w:gridSpan")
        tcPr.append(gridSpan)
    gridSpan.set(qn("w:val"), "4")
    tcW = tcPr.find(qn("w:tcW"))
    if tcW is not None:
        tcW.set(qn("w:w"), "9568")
        tcW.set(qn("w:type"), "dxa")
    trPr = row._tr.get_or_add_trPr()
    for tag in (qn("w:gridAfter"), qn("w:wAfter")):
        for node in list(trPr.findall(tag)):
            trPr.remove(node)
    seen = {id(tc0)}
    for cell in cells[1:]:
        tc = cell._tc
        if id(tc) not in seen and tc.getparent() is not None:
            tc.getparent().remove(tc)
            seen.add(id(tc))
    cells[0].text = text
    return cells[0]


doc = Document(SRC)
p = doc.paragraphs

set_text(p[0], "FLUXO DE POTÊNCIA ÓTIMO AC MULTIPERÍODO COM BATERIAS PARA MITIGAÇÃO DE SOBRETENSÃO E FLUXO REVERSO SOB ALTA PENETRAÇÃO DE MMGD", True)
for run in p[0].runs:
    run.font.name = "Arial"; run.font.size = Pt(18); run.font.color.rgb = RGBColor(31, 78, 121)
set_text(p[4], "A elevada penetração de micro e minigeração distribuída (MMGD) fotovoltaica reduz a carga líquida, pode elevar tensões e inverter fluxos. Este trabalho apresenta um fluxo de potência ótimo em corrente alternada (FPO-AC) multiperíodo com sistemas de armazenamento por baterias (BESS) instalados exclusivamente em barras de carga com MMGD. As tensões das barras geradoras são mantidas nos valores do caso-base e as baterias trocam apenas potência ativa. Foram avaliados os sistemas de 57 e 118 barras nos patamares de 0, 10, 25, 50, 80, 100 e 120% de penetração, com potência e energia dos BESS equivalentes a 80% da MMGD local. Em 120%, o armazenamento reduziu o fluxo reverso crítico de -207,7 para -171,9 MW no sistema de 57 barras e de -232,8 para -166,2 MW no de 118 barras. Na barra de carga mais sensível, a tensão máxima caiu de 1,1082 para 1,1002 pu e de 1,0471 para 1,0411 pu, respectivamente. Os resultados confirmam que o despacho temporal da potência ativa das baterias reduz excedentes fotovoltaicos, perdas e estresse operativo sem recorrer ao controle de tensão pelos conversores.")
set_text(p[5], "Palavras-chave: FPO-AC; MMGD fotovoltaica; BESS; sobretensão; inversão de fluxo.")

set_text(p[7], "A expansão da geração distribuída altera simultaneamente os balanços de potência ativa e reativa. Em períodos de elevada irradiância, a produção fotovoltaica pode superar o consumo local, reduzir a demanda líquida e estabelecer fluxos reversos. Em redes AC, essa condição também modifica magnitudes de tensão, perdas e carregamentos, razão pela qual uma representação completa da rede é necessária para avaliar a segurança operativa.")
set_text(p[8], "Os BESS deslocam energia entre períodos e oferecem flexibilidade rápida. Estudos recentes empregam FPO multiperíodo para coordenar armazenamento e geração fotovoltaica, com restrições de rede, potência e estado de carga. Li et al. [3] mostram que a coordenação de BESS em redes AC com alta penetração fotovoltaica reduz desvios de tensão e perdas. Silva, Belati e López-Lezama [4] formulam um FPO multiperíodo para operação e alocação de BESS junto à geração PV, enquanto Zheng et al. [5] destacam o papel do armazenamento sob incerteza em redes de distribuição.")
set_text(p[9], "Neste trabalho, o BESS é deliberadamente modelado sem injeção ou absorção reativa. Assim, seu efeito sobre a tensão decorre apenas da alteração do balanço de potência ativa e dos fluxos na rede. As tensões das barras geradoras permanecem fixas nos valores especificados no caso-base; a potência reativa convencional ajusta-se para sustentar essas referências.")
set_text(p[10], "O objetivo é quantificar a capacidade de baterias instaladas nas mesmas barras da MMGD de atenuar fluxo reverso, sobretensão e perdas. A avaliação utiliza os sistemas de 57 e 118 barras e sete níveis de penetração, incluindo condições nas quais a potência fotovoltaica instalada supera a demanda nominal do sistema.")

set_text(p[13], "A MMGD é representada como injeção ativa disponível segundo uma curva solar horária, enquanto as demandas ativa e reativa seguem o perfil de carga de 24 horas. A penetração corresponde à razão entre a potência total instalada de MMGD e a demanda ativa nominal do sistema. Essa capacidade é distribuída proporcionalmente à carga das barras hospedeiras, todas do tipo PQ. Em cada uma dessas barras instala-se um BESS com potência nominal e capacidade energética iguais a 80% da MMGD local, considerando duração de uma hora.")
set_text(p[14], "O modelo FPO-AC preserva as equações não lineares de balanço nodal, os fluxos ativos e reativos, as perdas e os limites operacionais. O acoplamento temporal decorre do balanço de energia das baterias. O estado de carga é limitado entre 20 e 90%, com valor inicial e final de 25%. A função objetivo penaliza déficit, perdas, custo e desvio do despacho convencional, variação da carga líquida e movimentação do BESS.")

set_text(p[16], "A formulação reduzida destaca as relações determinantes para o fenômeno analisado. Para cada barra i e hora h, os balanços ativos e reativos igualam as injeções líquidas à soma dos fluxos nos ramos incidentes. A MMGD entra apenas no balanço ativo, assim como a diferença entre descarga e carga da bateria. Como não existe variável reativa do BESS, seu despacho não controla diretamente a tensão terminal.")
set_text(p[17], "Os fluxos AC são calculados a partir das magnitudes V, diferenças angulares e elementos da matriz de admitância. A energia armazenada evolui com as eficiências de carga e descarga, respeitando limites de potência, rampa, SOC e condição cíclica. Variáveis binárias evitam carga e descarga simultâneas; nos estudos apresentados, os modos horários são predefinidos pela disponibilidade solar e as variáveis contínuas são resolvidas pelo IPOPT.")
set_text(p[18], "As equações (1)-(6) resumem o modelo. Os índices g, l e b representam geradores, ramos e baterias. Pch e Pdis são não negativas; u indica o modo de carga. A Tabela 1 reúne os símbolos essenciais.")

# Equações reduzidas em seis linhas, preservando o componente tabular do template.
t0 = doc.tables[0]
eqs = [
    "min  Σh [wdef·Pdef + wloss·Ploss + wPg·(Pg-Pgref)² + wV·(V-1)² + wB·(Pch+Pdis)]        (1)",
    "Σg∈Gi Pg,h + PMMGD,i,h + Σb∈Bi(Pdis,b,h-Pch,b,h) - Pd,i,h + Pdef,i,h = Σl∈i Pl,h        (2)",
    "Σg∈Gi Qg,h - Qd,i,h + Qsh,i(Vi,h)² = Σl∈i Ql,h                                      (3)",
    "E b,h+1 = E b,h + ηch Pch,b,h - Pdis,b,h/ηdis                                      (4)",
    "SOCmin Ebmax ≤ Eb,h ≤ SOCmax Ebmax ; Eb,0 = Eb,T = 0,25 Ebmax                       (5)",
    "0≤Pch≤u Pbmax ; 0≤Pdis≤(1-u)Pbmax ; Pbmax=Ebmax=0,80 PMMGD,bmax                     (6)",
]
for i, eq in enumerate(eqs):
    cell = span_row(t0.rows[i], eq)
    para = cell.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in para.runs:
        run.font.name = "Cambria Math"; run.font.size = Pt(9)
for i in range(len(t0.rows) - 1, 5, -1):
    remove_row(t0, i)

t1 = doc.tables[1]
cell = span_row(t1.rows[0], "PMMGD,i,h = κ · Pd,pico · wi · s(h),  com  Σi wi = 1 e κ ∈ {0; 0,10; 0,25; 0,50; 0,80; 1,00; 1,20}    (7)")
cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in cell.paragraphs[0].runs:
    run.font.name = "Cambria Math"; run.font.size = Pt(9)

# Tabela de símbolos compacta.
t2 = doc.tables[2]
symbols = [
    ("i,h", "índices", "-", "Barra e estágio horário"),
    ("Pg,Qg", "variáveis", "MW/MVAr", "Despachos ativo e reativo dos geradores"),
    ("V,θ", "variáveis", "pu/rad", "Magnitude e ângulo da tensão nodal"),
    ("Pl,Ql", "variáveis", "MW/MVAr", "Fluxos ativos e reativos nos ramos"),
    ("Pd,Qd", "parâmetros", "MW/MVAr", "Demandas ativa e reativa"),
    ("PMMGD", "parâmetro", "MW", "Potência fotovoltaica disponível"),
    ("Pch,Pdis", "variáveis", "MW", "Potências de carga e descarga do BESS"),
    ("E,SOC", "variável/razão", "MWh/%", "Energia armazenada e estado de carga"),
    ("ηch,ηdis", "parâmetros", "-", "Eficiências de carga e descarga"),
    ("u", "variável binária", "-", "Modo de carga do BESS"),
    ("κ", "parâmetro", "%", "Penetração total de MMGD"),
]
for r, vals in enumerate(symbols, start=1):
    for c, val in enumerate(vals): t2.cell(r, c).text = val
for i in range(len(t2.rows) - 1, len(symbols), -1):
    remove_row(t2, i)

set_text(p[22], "O FPO-AC é não convexo e sua solução é local. A consistência física foi verificada por resíduos máximos relativos de 3,16×10⁻⁶ no sistema de 57 barras e 1,83×10⁻⁵ no de 118 barras. Todas as barras com MMGD são PQ, a razão BESS/MMGD é exatamente 80% e o desvio das tensões especificadas nas barras geradoras foi numericamente nulo.")
set_text(p[25], "O problema multiperíodo é montado em Pyomo e resolvido pelo IPOPT. Para cada penetração, calcula-se primeiro o FPO-AC horário sem BESS; essa solução fornece a referência de despacho e o estado inicial do problema acoplado. Em seguida, o BESS é otimizado nas 24 horas com condição cíclica de energia. A solução do patamar anterior é utilizada como partida do seguinte, melhorando a robustez nos casos de 100 e 120%.")
set_text(p[26], "Algoritmo principal\nEntrada: rede AC, perfis de carga e solar, barras de MMGD e penetrações;\n1. Validar barras PQ e distribuir a capacidade de MMGD;\n2. Fixar tensões geradoras e resolver os 24 FPO-AC sem BESS;\n3. Dimensionar cada BESS em 80% da MMGD local;\n4. Montar balanços AC, restrições de SOC e modos de carga/descarga;")
set_text(p[27], "5. Resolver o FPO-AC multiperíodo com IPOPT; 6. Verificar resíduos, tensões, fluxos e SOC;")
set_text(p[28], "Saída: bancos consolidados, tabelas e gráficos comparativos por penetração.")
set_text(p[29], "")
set_text(p[31], "A implementação foi realizada em Python 3.14 com Pyomo. Os subproblemas contínuos foram solucionados pelo IPOPT e a infraestrutura de otimização inteira dispõe do GLPK. Os sistemas possuem 57 barras/80 ramos/7 geradores e 118 barras/186 ramos/54 geradores, com BESS em 12 e 20 barras, respectivamente.")
set_text(p[33], "Foram usados SOC inicial e final de 25%, limites de 20-90%, eficiências de 95% e duração de uma hora. Para que os patamares exploratórios alcançassem 120% sem que a análise fosse interrompida por limites térmicos derivados do caso-base, aplicaram-se margens de estudo de 3× e 6× nos sistemas de 57 e 118 barras. Impedâncias, balanços e perfis não foram modificados.")

set_text(p[35], "No sistema de 57 barras, a barra de carga 31 apresentou a maior flutuação diária. Em 120%, sua tensão máxima passou de 1,1082 pu sem BESS para 1,1002 pu com BESS, enquanto a mínima subiu de 0,9639 para 0,9827 pu. Na linha crítica 1-15, o extremo reverso foi reduzido de -207,7 para -171,9 MW. As perdas diárias caíram de 556,2 para 463,4 MWh (16,7%). A atuação é mais pronunciada entre 9 h e 14 h, período em que a MMGD excede a carga e as baterias absorvem potência ativa.")
set_text(p[36], "No sistema de 118 barras, a maior variação ocorreu na barra 45. No patamar de 120%, a máxima diminuiu de 1,0471 para 1,0411 pu e a mínima aumentou de 0,9860 para 0,9927 pu. Na linha 8-5, a maior inversão foi atenuada de -232,8 para -166,2 MW. As perdas foram reduzidas de 2055,6 para 1710,8 MWh (16,8%) e a maior ocupação da rede caiu de 89,27 para 71,94%. A condição cíclica de SOC foi respeitada e não houve déficit de carga.")
set_text(p[38], "Figura 1 – Resultados do FPO-AC para o sistema de 57 barras")
set_text(p[40], "Figura 2 – Resultados do FPO-AC para o sistema de 118 barras")
set_text(p[44], "")

set_text(p[46], "Os resultados demonstram que BESS puramente ativos, posicionados nas barras com MMGD, reduzem fluxo reverso, perdas e amplitude de tensão em altas penetrações. O efeito cresce acima de 50%, quando a geração fotovoltaica altera de modo significativo o sentido dos intercâmbios. Em 120%, as reduções do extremo reverso foram de 17,2% no sistema de 57 barras e 28,6% no de 118 barras, confirmando maior benefício na rede de maior porte.")
set_text(p[47], "A manutenção das tensões geradoras no caso-base separou o efeito do armazenamento daquele produzido por controle de tensão dos conversores. Como continuidade, recomenda-se incorporar limites reativos estritos com tratamento explícito da transição PV-PQ, degradação eletroquímica, incerteza solar e comparação com estratégias que também utilizem potência reativa do BESS.")

refs = [
    "[1] ONS. Revista PAR/PEL 2025. Disponível em: <https://www.ons.org.br/AcervoDigitalDocumentosEPublicacoes/Revista%20PARPEL%202025-VF.pdf>. Acesso em: 31 out. 2026.",
    "[2] EPE. Caderno MMGD e baterias – PDE 2034. Rio de Janeiro, 2024. Disponível em: <https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-804/topico-709/Caderno_MMGD_Baterias_PDE2034_(20240702).pdf>.",
    "[3] Z. Li, Y. Yan, D. Qi, S. Yan e M. Wang. Distributed Voltage Optimization Control of BESS in AC Distribution Networks with High PV Penetration. Energies, v. 15, 4120, 2022. doi: 10.3390/en15114120.",
    "[4] D. J. da Silva, E. A. Belati e J. M. López-Lezama. Enhancing Distribution Networks with Optimal BESS Sitting and Operation: A Weekly Horizon Optimization Approach. Sustainability, v. 16, 7248, 2024. doi: 10.3390/su16177248.",
    "[5] Y. Zheng et al. Optimal Operation of Battery Energy Storage System Considering Distribution System Uncertainty. IEEE Transactions on Sustainable Energy, v. 9, n. 3, p. 1051-1060, 2018. doi: 10.1109/TSTE.2017.2762364.",
    "[6] EPE. Metodologia para criação de séries horárias de geração distribuída fotovoltaica por subsistema. Informe Técnico, 2021.",
    "[7] ONS Dados Abertos. Disponível em: <https://dados.ons.org.br/>. Acesso em: 31 out. 2026.",
    "[8] W. E. Hart, J.-P. Watson e D. L. Woodruff. Pyomo: modeling and solving mathematical programs in Python. Mathematical Programming Computation, v. 3, p. 219-260, 2011.",
    "[9] W. F. Alves. Proposição de sistemas-teste para análise computacional de sistemas de potência. Niterói, 2007.",
]
for i, text in enumerate(refs, start=49):
    set_text(p[i], text, italic=True)
for i in range(58, 59): set_text(p[i], "")
p[48].paragraph_format.page_break_before = True

set_text(p[61], "☒ UTILIZAMOS ferramentas de inteligência artificial generativa conforme detalhado abaixo.")
p[59].paragraph_format.page_break_before = True
t5 = doc.tables[5]
vals = ["ChatGPT Codex", "GPT-5 / 31 out. 2026", "Revisão textual, pesquisa bibliográfica, edição do DOCX e apoio à análise dos resultados", "Resumo, formulação, resultados, referências, figuras e revisão de formatação"]
for c, val in enumerate(vals): t5.cell(1, c).text = val
for cell in t5.rows[1].cells:
    for para in cell.paragraphs:
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in para.runs: run.font.size = Pt(9)

doc.save(WORK)

# Substitui somente as oito mídias, preservando relações, posições e dimensões.
plots = []
for system in ("B57_TAP_FIXO", "B118_TAP_FIXO"):
    base = ROOT / "resultados_comparativos" / system
    plots.extend([
        base / "01_tensao_barra_critica.png",
        base / "02_fluxo_linha_critica.png",
        base / "03_despacho_geradores.png",
        base / "04_despacho_soc_bess.png",
    ])

tmp = WORK.with_suffix(".tmp.docx")
with ZipFile(WORK, "r") as zin, ZipFile(tmp, "w", ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename.startswith("word/media/image") and item.filename.endswith(".png"):
            idx = int(Path(item.filename).stem.replace("image", "")) - 1
            if 0 <= idx < len(plots):
                data = fit_png(plots[idx])
        zout.writestr(item, data)
tmp.replace(WORK)
shutil.copy2(WORK, FINAL)
print(FINAL)
