### Conclusões

O objetivo do trabalho era demonstrar um processo confiável de evolução de modelos, e não apenas obter a maior métrica.
Nesse sentido, o pipeline cumpriu o que se propôs: executou **7 ciclos completos sem nenhuma falha e sem intervenção
humana** depois do disparo inicial. Em cada ciclo, incorporou um lote novo, validou o contrato de dados, refez a votação
de features, treinou os dois algoritmos com o mesmo protocolo, registrou tudo no MLflow e decidiu sozinho entre promover
ou manter o modelo vigente. Todas as decisões podem ser auditadas pelo `cycle_id`, pela `data_version` e pelo manifesto.

**A regra de promoção funcionou como uma proteção, não como um atalho.** Foram 3 promoções e 4 manutenções. Nos ciclos 3
e 5 o candidato não superou o vigente pelo limiar de 0,01 de ROC-AUC (diferenças de −0,0003 e +0,0093), e o modelo
existente foi mantido: trocar um modelo por um ganho dentro da variação normal da validação só adicionaria risco. Nos
ciclos 6 e 7, sem dados novos, o pipeline reprocessou os mesmos dados, chegou aos mesmos números (o processo é
determinístico) e manteve o vigente, que é o comportamento esperado. Já no ciclo 4, quando o lote trouxe um período
com taxa de conversão de 27,6%, o candidato superou o vigente em +0,085 e foi promovido.

**A competição entre os algoritmos mudou de vencedor conforme os dados cresceram.** No primeiro ciclo, com dados de um
período quase sem variação econômica e sem campanhas anteriores, a Regressão Logística venceu: com pouco sinal, o
modelo mais simples generaliza melhor. A partir do segundo ciclo, o HistGradientBoosting venceu todos os ciclos,
porque a chegada de períodos diferentes trouxe interações não lineares entre os indicadores econômicos e o histórico
de campanhas que um modelo linear não captura.

**Uma promoção correta pelo protocolo piorou o desempenho no teste reservado.** No ciclo 2, o candidato superou o
vigente na validação (0,7073 contra 0,6947) e foi promovido, mas a ROC-AUC no teste caiu de 0,716 para 0,644. A
validação só contém períodos já incorporados, enquanto o teste reservado cobre toda a série, inclusive regimes
econômicos que o modelo ainda não tinha visto. O novo modelo se ajustou melhor ao passado conhecido e extrapolou pior
para o futuro. O efeito foi revertido no ciclo 4, quando esses regimes entraram no treino e a AUC no teste subiu para
0,811. Consideramos esse o principal aprendizado metodológico do trabalho: **a regra de promoção é tão boa quanto o
conjunto em que ela compara os modelos**. A validação por hash foi escolhida porque um corte temporal media só a ruptura
entre regimes (AUC ≈ 0,5 para qualquer modelo), mas o caso do ciclo 2 mostra que vale combinar os dois sinais.

**A votação de features revelou o que de fato explica a conversão.** Sete atributos foram eleitos em todos os ciclos:
os cinco indicadores macroeconômicos (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`),
o mês e o canal de contato. O resultado de uma ligação depende mais do momento econômico do que do perfil do cliente.
A votação também se adaptou aos dados: `poutcome` e `contatado_antes` passaram a ser eleitas quando chegaram lotes com
campanhas anteriores, e atributos de perfil como `job`, `education` e `faixa_etaria` deixaram de ser. Isso só foi
possível porque a seleção é refeita a cada ciclo, em vez de fixada uma única vez.

**O monitoramento de drift (critério 10) mostrou valor, mas também seus limites.** Ele alertou nos 4 lotes novos, com
9 a 15 atributos acima do limiar e PSI máximo entre 8,7 e 11,5, e abriu uma issue automaticamente em cada caso. O
alerta antecipou corretamente a mudança de cenário dos ciclos 4 e 5, quando a taxa de conversão do lote novo saltou
para 27,6% e 49,2%, contra 6,5% e 8,2% nos dados já incorporados. Por outro lado, no ciclo 3 o alerta disparou mesmo com a taxa de conversão estável (6,5% para 6,7%)
e sem ganho de desempenho. Os indicadores macroeconômicos são quase constantes dentro de cada período, então qualquer
mudança de período gera PSI muito alto. Concluímos que o drift é um sinal para **investigar**, não um critério para
decidir sozinho, e que os limiares deveriam ser calibrados separadamente para atributos de contexto econômico e de
perfil do cliente.

**O modelo vigente final é útil para o problema de negócio.** O HistGradientBoosting promovido no ciclo 4 alcançou, no
teste reservado, ROC-AUC de 0,811, recall de 69% e precisão de 33% para a classe positiva. Como só 11,3% dos clientes do
teste convertem, a precisão do modelo é quase três vezes a taxa base: ordenar a lista de ligações pela probabilidade
prevista concentra os contatos em quem tem mais chance de contratar.

**Do ponto de vista de engenharia, a principal lição foi não depender de um único mecanismo.** O serviço de agendamento
do GitHub não criou nenhuma execução para a conta do projeto, nem num repositório de teste mínimo, sem nenhum erro
visível. Só percebemos porque monitoramos se os ciclos realmente aconteciam. A contingência, um job no mesmo workflow
que dispara o ciclo seguinte, preservou a automação e a rastreabilidade, e cada execução registra de onde veio. A
persistência do estado no próprio repositório também se mostrou robusta: os runners são descartados a cada execução, e
mesmo assim cada ciclo encontrou o modelo vigente e o histórico deixados pelo anterior.

### Trabalhos futuros

- Combinar, na regra de promoção, a validação por hash com uma fatia temporal recente, exigindo que o candidato não piore em nenhuma das duas.
- Calibrar as probabilidades e escolher o limiar de decisão pelo custo de uma ligação contra o ganho de uma conversão, em vez do limiar fixo de 0,5.
- Separar os limiares de drift por tipo de atributo e usar o alerta para priorizar a revisão do modelo.
- Monitorar a ausência de execuções agendadas, gerando um alerta quando um ciclo esperado não acontecer.
- Em produção, substituir o SQLite versionado no Git por um servidor MLflow com armazenamento de objetos e expor o modelo vigente por uma API.
