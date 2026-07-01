# 🍷 Projeto Aplicado: Agrupamento Inteligente de Vinhos (Clusterização)

Este repositório contém o projeto final da disciplina de Aprendizado de Máquina. O objetivo é aplicar conceitos de aprendizagem não supervisionada, redução de dimensionalidade e métricas de validação para resolver um problema de negócios envolvendo categorização automatizada de vinhos com base em suas propriedades químicas.

---

## 1. 🎯 Problema e Motivação
**Contexto de Negócios:** Uma grande distribuidora de bebidas adquire lotes de vinhos de múltiplos pequenos produtores na Itália. A empresa possui análises químicas laboratoriais detalhadas de cada garrafa, mas frequentemente perde a rastreabilidade da origem exata ou deseja criar "Categorias de Sabor/Perfil" de forma automatizada para o seu catálogo, sem depender de degustadores humanos ou rótulos prévios.

**Motivação:** A clusterização automatizada permite à distribuidora organizar seu portfólio, criar sistemas de recomendação precisos (ex: *"clientes que gostam do grupo A também gostarão desta nova garrafa"*), além de facilitar a identificação de anomalias, falsificações ou inconsistências em lotes de fornecedores.

---

## 2. 📊 Descrição do Dataset
Utilizamos o **Wine Dataset**, disponibilizado nativamente pela biblioteca `scikit-learn`. 
* **Tamanho:** 178 amostras de vinho.
* **Características:** 13 variáveis químicas contínuas (ex: Álcool, Ácido Málico, Alcalinidade, Magnésio, Fenóis Totais, Flavonoides, Intensidade da Cor, Matiz, Prolina, etc.).
* **Abordagem:** Para simular o cenário real da distribuidora, a variável alvo original (`target` - que indica o cultivador) foi propositalmente ignorada. O modelo atua em um cenário **100% não supervisionado**.

---

## 3. ⚙️ Preparação dos Dados
A consistência dos dados é garantida através de uma etapa crítica de pré-processamento:
* **Padronização (Z-Score):** Através do `StandardScaler`, todas as 13 variáveis foram padronizadas para possuírem média 0 e desvio padrão 1. Isso é estritamente necessário porque as grandezas variam muito (ex: a Prolina pode passar de 1.000, enquanto os Fenóis variam de 1 a 3). Sem essa etapa, algoritmos baseados em distância (como os que utilizamos) dariam um peso desproporcional à Prolina.

---

## 4. 🤖 Modelos Comparados
Para definir a melhor estratégia de agrupamento, colocamos em competição dois algoritmos clássicos de naturezas diferentes, variando o número de *clusters* ($k$) de 2 a 8:
1. **K-Means (Agrupamento Particional):** Busca particionar os dados encontrando centróides geométricos que minimizem a variância dentro de cada grupo.
2. **Agglomerative Clustering (Agrupamento Hierárquico):** Adota uma abordagem *bottom-up*, onde cada amostra começa como seu próprio cluster e pares de clusters são mesclados iterativamente com base na distância entre eles.

---

## 5. 📉 Análise Não Supervisionada e PCA
Trabalhar com 13 dimensões impossibilita a visualização e pode inserir ruído excessivo. Aplicamos a **Análise de Componentes Principais (PCA)** com `n_components=2`.
* **Benefício:** Comprimimos as 13 características químicas em apenas 2 eixos principais (Componente Principal 1 e 2) que capturam a maior parte da variância dos dados originais.
* **Resultado:** Os dados puderam ser projetados em um plano geométrico 2D limpo, que serve de entrada para os modelos de clusterização e facilita a interpretação visual por parte das áreas de negócio.

---

## 6. 📏 Métrica Principal e Justificativa
O desempenho dos modelos foi avaliado exclusivamente pelo **Silhouette Score (Coeficiente de Silhueta)**.
* **Justificativa:** Como não temos gabaritos (rótulos) para calcular acurácia, precisamos de uma métrica intrínseca. O *Silhouette Score* (que varia de -1 a 1) mede a "qualidade" do grupo formado. Ele avalia a **coesão** (distância de um vinho para os outros vinhos do mesmo grupo) versus a **separação** (distância desse vinho para os vinhos do grupo vizinho mais próximo). Valores mais altos indicam fronteiras de decisão mais nítidas.

---

## 7. 🏆 Interpretação dos Resultados
O script executa um pipeline automatizado que itera sobre diferentes valores de $k$ e extrai o *Silhouette Score* de cada modelo:
* Empiricamente, o ecossistema dos dados aponta para um número ideal de **k=3** agrupamentos, maximizando a silhueta.
* Historicamente, nesses dados reduzidos, o **K-Means** tende a apresentar fronteiras ligeiramente mais otimizadas (maior score) que o modelo hierárquico.
* O algoritmo vencedor é eleito de forma automatizada pelo script, e o resultado final é plotado mostrando a distribuição visual das categorias sugeridas pela máquina. Curiosamente, os 3 clusters definidos pela química espelham perfeitamente as 3 fazendas originais de cultivo, comprovando o sucesso do algoritmo.

---

## 8. ⚠️ Limitações
Apesar dos excelentes resultados, a abordagem possui algumas limitações técnicas:
1. **Perda de Informação (Variância):** A compressão de 13 dimensões para 2 componentes (PCA) descarta uma porcentagem da informação química original, ignorando sutilezas que poderiam estar nas dimensões excluídas.
2. **Explicabilidade:** O espaço do PCA é abstrato. Ao informar que um vinho pertence ao "Cluster 1", a equipe de negócios não sabe imediatamente qual a regra química exata (ex: "é porque tem mais álcool?"). Para contornar isso no futuro, seria necessário cruzar os clusters formados com as médias das variáveis no dataset original.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python instalado (recomenda-se a versão 3.8 ou superior). Instale as dependências executando o comando abaixo:

```bash
pip install pandas numpy matplotlib scikit-learn