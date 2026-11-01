Preciso que você me ajude a executar o fpo-ac com mmgd e baterias nas barras com MMGD. A ideia é mostrar que em altas penetrações de mmgd, a bateria é capaz de reduzir fenômenos como a inversão de fluxo e também reduzir sobre tensões decorrentes da redução da carga líquida. Porém, da maneira como o código está estruturado atualmente, os resultados mostrados estão pouco satisfatórios, pois não está sendo mostrado a inversão de fluxo e nem grandes mudanças de tensão ao longo do dia devido à MMGD. Então, você será designado para as seguintes tarefas:

- Rodar o fpo-ac garantindo que as baterias estejam em barras com mmgd e que estas injetem/absorvam apenas potência ativa. Preciso que você selecione patamares elevados de penetração, como 0, 10, 25, 50, 80, 100 e 120% para fins de comparação.

- As tensões nas barras com geradores devem permanecer em seus valores especificados no caso base, ou seja, elas não têm liberdade para variar de acordo com a solução do fpo.

- Quero focar a execução apenas nos aqruivos do sistemas de 57 e 118 barras, logo, você pode deletar os arquivos dos demais sistemas.

- Como resultados, quero observar apenas um comparativo entre os cenários em função da penetração, envolvendo tensão nas barras onde houve maior flutuação de valor durante o dia, fluxo nas linhas onde houve a maior inversão, despachos ativo e reativo dos geradores, despachos e SOC das baterias.

- Se as baterias estiverem com capacidade de controle de tensão nas barras terminais quero que você desabilite essa modelagem.

- Quero que você limpe ao máximo arquivos que não serão mais úteis às simulações. Além disso, deixe facilitado a mudança dos cenários de penetração de mmgd para que eu mesmo possa fazer nos arquivos de dados de entrada para o programa.

- Faça com que a capacidade das baterias seja de 80% do valor máximo da capacidade instalada de mmgd nas barras.

- Cheque se todas as barras com mmgd são barras de carga (tipo = 0).