import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Carregamento e Descrição dos Dados
print("--- PROJETO FINAL: AGRUPAMENTO DE VINHOS ---")
wine = load_wine()
X_raw = pd.DataFrame(wine.data, columns=wine.feature_names)
print(f"Dataset carregado com {X_raw.shape[0]} amostras e {X_raw.shape[1]} características químicas.\n")

# 2. Preparação dos Dados (Padronização)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Análise Não Supervisionada (Redução de Dimensionalidade com PCA)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

variancia_explicada = pca.explained_variance_ratio_.sum() * 100
print(f"Variância mantida após PCA (2 componentes): {variancia_explicada:.2f}%\n")

# 4 e 6. Modelos Comparados e Métrica Principal
ks = range(2, 9) # Testando de 2 a 8 clusters
silhouette_kmeans = []
silhouette_agglo = []

print("Comparativo de Silhouette Score (Quanto mais perto de 1.0, melhor):")
for k in ks:
    # Treina K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_pca)
    score_k = silhouette_score(X_pca, labels_kmeans)
    silhouette_kmeans.append(score_k)
    
    # Treina Agglomerative (Hierárquico)
    agglo = AgglomerativeClustering(n_clusters=k)
    labels_agglo = agglo.fit_predict(X_pca)
    score_a = silhouette_score(X_pca, labels_agglo)
    silhouette_agglo.append(score_a)
    
    print(f"K={k} | K-Means: {score_k:.4f} | Hierárquico: {score_a:.4f}")

# 7. Interpretação dos Resultados (Escolha do melhor modelo e K)
melhor_k_kmeans = ks[np.argmax(silhouette_kmeans)]
melhor_k_agglo = ks[np.argmax(silhouette_agglo)]
max_score_kmeans = max(silhouette_kmeans)
max_score_agglo = max(silhouette_agglo)

print("\n--- CONCLUSÃO DOS MODELOS ---")
print(f"Melhor K-Means: K={melhor_k_kmeans} (Score: {max_score_kmeans:.4f})")
print(f"Melhor Hierárquico: K={melhor_k_agglo} (Score: {max_score_agglo:.4f})")

# Vamos treinar o modelo vencedor final para plotar
if max_score_kmeans >= max_score_agglo:
    print("\nModelo Vencedor: K-Means!")
    modelo_final = KMeans(n_clusters=melhor_k_kmeans, random_state=42, n_init=10)
else:
    print("\nModelo Vencedor: Agglomerative Clustering!")
    modelo_final = AgglomerativeClustering(n_clusters=melhor_k_agglo)

clusters_finais = modelo_final.fit_predict(X_pca)

# Visualização Final
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: A evolução do Silhouette
ax1.plot(ks, silhouette_kmeans, marker='o', label='K-Means')
ax1.plot(ks, silhouette_agglo, marker='s', label='Hierárquico')
ax1.set_title("Comparação de Desempenho (Silhouette Score)")
ax1.set_xlabel("Número de Clusters (k)")
ax1.set_ylabel("Silhouette Score")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Gráfico 2: A separação final no PCA
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_finais, cmap='viridis', s=60, edgecolor='k', alpha=0.8)
ax2.set_title(f"Agrupamento Final (Vencedor, K={len(np.unique(clusters_finais))}) no Espaço PCA")
ax2.set_xlabel("Componente Principal 1")
ax2.set_ylabel("Componente Principal 2")
plt.colorbar(scatter, ax=ax2, label="Cluster ID")

plt.tight_layout()
plt.show()