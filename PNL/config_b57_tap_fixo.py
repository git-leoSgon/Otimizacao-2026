from config_b57 import *


SISTEMA = "B57_TAP_FIXO"

# O STB 57 nao possui faixa de regulacao automatica declarada. As relacoes
# originais sao mantidas fixas durante as 24 horas.
CONTROLES_TAP = {}
TAPS_INICIAIS = {}
DESCRICAO_TAPS = "fixos nas posicoes conservadoras originais do caso-base"
# Margem de estudo para que os patamares altos revelem a inversao de fluxo,
# sem a fronteira termica artificial interromper a varredura.
FATOR_LIMITE_TERMICO_ESTUDO = 3.0
PERMITIR_EXPORTACAO_REFERENCIA = True
PESO_TENSAO = 100.0
PESO_PLIQ = 100.0
PESO_VARIACAO_BESS = 2.0
OTIMIZAR_MODOS_BESS = False
USAR_PRE_SOLUCAO_CONTINUA = False
REUTILIZAR_ESTADO_Q_PARTIDA = True
VERSAO_SOLUCAO_BESS = "nlp_tensao_geradores_fixa_bess_p_80pct_sistema_v1"

RESULTADOS = PASTA / "resultados_tap_fixo_pliq100_variacao2" / "B57"
PASTA_INVENTARIO = RESULTADOS
