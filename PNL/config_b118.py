from pathlib import Path


PB = 100.0
SEED = 123
SISTEMA = "B118"

PENETRACAO_TESTE = 50
PENETRACOES = [0, 20, 30, 40, 50, 60]
BARRAS_MMGD = [
    60, 11, 78, 45, 88, 82, 106, 75, 95, 35,
    41, 3, 79, 96, 47, 98, 13, 118, 115, 67,
]

FATOR_POTENCIA_CARGA = 0.95
VMIN = 0.95
VMAX = 1.05
VMAX_BARRAS = {}

TENSAO_GERADORES = {
    1: 0.95,
    4: 1.00,
    6: 0.99,
    8: 1.01,
    10: 1.05,
    12: 0.99,
    15: 0.97,
    18: 0.97,
    19: 0.96,
    24: 0.99,
    25: 1.05,
    26: 1.01,
    27: 0.97,
    31: 0.97,
    32: 0.96,
    34: 0.98,
    36: 0.98,
    40: 0.97,
    42: 0.98,
    46: 1.00,
    49: 1.02,
    54: 0.95,
    55: 0.95,
    56: 0.95,
    59: 0.98,
    61: 1.00,
    62: 1.00,
    65: 1.00,
    66: 1.05,
    69: 1.03,
    70: 0.98,
    72: 0.98,
    73: 0.99,
    74: 0.96,
    76: 0.94,
    77: 1.01,
    80: 1.04,
    85: 0.98,
    87: 1.01,
    89: 1.00,
    90: 0.98,
    91: 0.98,
    92: 0.99,
    99: 1.01,
    100: 1.02,
    103: 1.01,
    104: 0.97,
    105: 0.96,
    107: 0.95,
    110: 0.97,
    111: 0.98,
    112: 0.97,
    113: 0.99,
    116: 1.00,
}

DEMANDA_REATIVA_MVAR = {
    1: 27.0,
    2: 9.0,
    3: 10.0,
    4: 12.0,
    6: 22.0,
    7: 2.0,
    11: 23.0,
    12: 10.0,
    13: 16.0,
    14: 1.0,
    15: 30.0,
    16: 10.0,
    17: 3.0,
    18: 34.0,
    19: 25.0,
    20: 3.0,
    21: 8.0,
    22: 5.0,
    23: 3.0,
    27: 13.0,
    28: 7.0,
    29: 4.0,
    31: 27.0,
    32: 23.0,
    33: 9.0,
    34: 26.0,
    35: 9.0,
    36: 17.0,
    39: 11.0,
    40: 23.0,
    41: 10.0,
    42: 23.0,
    43: 7.0,
    44: 8.0,
    45: 22.0,
    46: 10.0,
    48: 11.0,
    49: 30.0,
    50: 4.0,
    51: 8.0,
    52: 5.0,
    53: 11.0,
    54: 32.0,
    55: 22.0,
    56: 18.0,
    57: 3.0,
    58: 3.0,
    59: 113.0,
    60: 3.0,
    62: 14.0,
    66: 18.0,
    67: 7.0,
    70: 20.0,
    74: 27.0,
    75: 11.0,
    76: 36.0,
    77: 28.0,
    78: 26.0,
    79: 32.0,
    80: 26.0,
    82: 27.0,
    83: 10.0,
    84: 7.0,
    85: 15.0,
    86: 10.0,
    88: 10.0,
    90: 42.0,
    92: 10.0,
    93: 7.0,
    94: 16.0,
    95: 31.0,
    96: 15.0,
    97: 9.0,
    98: 8.0,
    100: 18.0,
    101: 15.0,
    102: 3.0,
    103: 16.0,
    104: 25.0,
    105: 26.0,
    106: 16.0,
    107: 12.0,
    108: 1.0,
    109: 3.0,
    110: 30.0,
    112: 13.0,
    114: 3.0,
    115: 7.0,
    117: 8.0,
    118: 15.0,
}

SHUNTS_BARRAS_MVAR = {
    5: -40.0,
    34: 14.0,
    37: -25.0,
    44: 10.0,
    45: 10.0,
    46: 10.0,
    48: 15.0,
    74: 12.0,
    79: 20.0,
    82: 20.0,
    83: 10.0,
    105: 20.0,
    107: 6.0,
    110: 6.0,
}
INJECOES_REATIVAS_FIXAS_MVAR = {}
CONTROLES_TAP = {}
TAPS_INICIAIS = {}

TOLERANCIA_TAP = 5.0e-3
TOLERANCIA_REATIVO_MVAR = 1.0e-3
TOLERANCIA_RETORNO_PV = 2.0e-3
MAX_ITERACOES_PV_PQ = 30
FAIXA_CARGA = (0.75, 1.00)
FAIXA_MMGD = (0.90, 1.00)

FRACAO_BESS = 0.80
DURACAO_BESS = 1.0
RAMPA_BESS = 0.25
RAMPA_GERADORES = 0.60
SOC_INICIAL = 0.25
SOC_MIN = 0.20
SOC_MAX = 0.90
ETA_CARGA = 0.95
ETA_DESCARGA = 0.95

PESO_DEFICIT = 1.0e7
PESO_PERDAS = 100.0
PESO_CUSTO_GERACAO = 0.01
PESO_DESVIO_PG = 2000.0
PESO_DESVIO_PG_BASE = 500.0
PESO_PLIQ = 10.0
PESO_TENSAO = 1000.0
PESO_CONTROLE_TAP = 10000.0
PESO_MOVIMENTACAO_BESS = 0.10
PESO_VARIACAO_BESS = 20.0

TOLERANCIA_NR = 1.0e-8
MAX_ITERACOES_NR = 50
TOLERANCIA_IPOPT = 1.0e-6
TOLERANCIA_ACEITAVEL_IPOPT = 1.0e-3
MAX_ITERACOES_IPOPT = 1000
MAX_ITERACOES_MINLP = 50
TEMPO_LIMITE_MINLP = None

PARTIDAS_MINLP = ["solar_30"]
OTIMIZAR_MODOS_BESS = True
USAR_PRE_SOLUCAO_CONTINUA = True
PRE_SOLUCAO_PENETRACAO_MIN = 30.0
MAX_ITERACOES_PRE_SOLUCAO = 10
VERSAO_SOLUCAO_BESS = "minlp_pvpq_desvio_pg500_perf_v6"

PASTA = Path(__file__).resolve().parent
ARQUIVO = PASTA / "dados" / "STB_118.xlsx"
if not ARQUIVO.exists():
    ARQUIVO = PASTA.parent / "dados" / "STB_118.xlsx"
RESULTADOS = PASTA / "resultados" / SISTEMA
