# Contrato do template

- Referência: `C:\Users\Leo_g\OneDrive\Área de Trabalho\PNL\Resumo_estendido_PNL.docx`
- SHA-256: `60F2AC9514891DA9F7DDAE99E561896264B14EAAF0C8865BF1E3A3E06D336B02`
- Extensão: 6 páginas, 1 seção, 66 parágrafos, 6 tabelas, 8 imagens.
- Evidência visual: `docx_trabalho/template_render/page-1.png` a `page-6.png`.

## Sistema de páginas

- A4 retrato, 21,0 x 29,7 cm.
- Margens: superior 0,714 cm; inferior 0,675 cm; esquerda/direita 0,794 cm.
- Cabeçalho/rodapé: 0,318 cm.
- Cabeçalho azul com disciplina à esquerda e identificação do template à direita.
- Página 6 reservada integralmente ao apêndice de declaração de uso de IA.

## Tipografia e componentes

- Título centralizado, azul escuro, caixa alta e negrito.
- Autoria centralizada em negrito; afiliação centralizada em itálico cinza.
- Títulos `Heading 1` em azul escuro; subtítulos `Heading 2` em cinza escuro.
- Corpo justificado em fonte serifada, com parágrafos compactos.
- Legendas centralizadas e em negrito.
- Tabelas com cabeçalho azul-claro, bordas finas e texto compacto.
- Referências em itálico, numeração entre colchetes.

## Fluxo e slots

1. Página 1: título, autoria, resumo, palavras-chave, introdução e início da descrição.
2. Página 2: formulação matemática reduzida, equações e início da tabela de símbolos.
3. Página 3: continuação da tabela, validação, método e algoritmo.
4. Página 4: implementação, resultados B57 e primeira figura composta.
5. Página 5: resultados B118, segunda figura composta, conclusões e referências.
6. Página 6: declaração de IA e tabela de registro.

Slots editáveis: todos os parágrafos do corpo, tabelas de equações/símbolos/IA e as oito imagens de resultados. Cabeçalho, geometria, autoria e afiliação devem ser preservados. As imagens mantêm posição, tamanho e estrutura 2x2 das figuras originais, recebendo os novos gráficos B57/B118.

## Preservação

- Preservar estilos, tema, numeração, cabeçalhos, rodapés, relações e geometria da seção.
- Preservar as oito relações de imagem, substituindo apenas os bytes das mídias.
- Preservar seis páginas e a página final exclusiva para IA.
- Não adicionar comentários ou controle de alterações.

## Portões de fidelidade

- Renderizar as seis páginas após cada lote relevante.
- Conferir ausência de sobreposição, texto cortado, tabelas quebradas e figuras ilegíveis.
- Confirmar que a página 6 continua sendo o apêndice de IA.
- Confirmar que as páginas 4 e 5 exibem resultados novos dos sistemas B57 e B118.
