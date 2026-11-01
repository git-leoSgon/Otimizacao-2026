# FPO-CA com MMGD e BESS — sistemas de 57 e 118 barras

Os patamares são definidos em `PENETRACOES` nos arquivos `config_b57.py` e
`config_b118.py`. A lista padrão é `[0, 10, 25, 50, 80, 100, 120]`.

Penetração é a razão entre a potência instalada total de MMGD e a demanda
ativa nominal total do sistema. A MMGD é distribuída entre barras de carga
proporcionalmente às cargas locais. Há uma BESS puramente ativa em cada barra
com MMGD, com potência e energia nominais iguais a 80% da MMGD local e duração
de uma hora.

As tensões das barras com geradores ficam fixas nos valores do caso-base. A
potência reativa dos geradores sustenta esses valores. A barra de referência
representa a interface com a rede a montante e pode ter Pg negativo (exportação
líquida) nos patamares elevados.

Execute com:

```powershell
python main.py
python gerar_resultados_comparativos.py
```

Os bancos completos ficam em
`resultados_tap_fixo_pliq100_variacao2/<sistema>/etapa_03_penetracoes`. Os
gráficos e CSVs enxutos ficam em `resultados_comparativos`.

`FATOR_LIMITE_TERMICO_ESTUDO` amplia os limites térmicos apenas para que a
varredura exploratória alcance 120% sem interromper a observação de fluxo
reverso. Impedâncias e balanços de potência não são alterados.
