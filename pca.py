import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_preprocess_data(file_path):
    """
    Carga y preprocesa el dataset de salud y estilo de vida
    """
    # Cargar datos
    df = pd.read_csv(file_path)
    print("Forma del dataset:", df.shape)
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    print("\nInformación del dataset:")
    print(df.info())
    
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Verificar valores faltantes
    print("\nValores faltantes:")
    print(df.isnull().sum())
    
    return df

def prepare_data_for_pca(df):
    """
    Prepara los datos para el análisis PCA
    """
    # Crear una copia del dataframe
    df_pca = df.copy()
    
    # Codificar variable categórica 'gender'
    le = LabelEncoder()
    df_pca['gender_encoded'] = le.fit_transform(df_pca['gender'])
    
    # Seleccionar variables numéricas para PCA (excluyendo id y gender original)
    numeric_columns = [
        'age', 'bmi', 'daily_steps', 'sleep_hours', 'water_intake_l',
        'calories_consumed', 'smoker', 'alcohol', 'resting_hr',
        'systolic_bp', 'diastolic_bp', 'cholesterol', 'family_history',
        'disease_risk', 'gender_encoded'
    ]
    
    # Verificar que todas las columnas existen
    available_columns = [col for col in numeric_columns if col in df_pca.columns]
    print(f"\nColumnas disponibles para PCA: {available_columns}")
    
    X = df_pca[available_columns]
    
    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, available_columns, scaler

def perform_pca_analysis(X_scaled, feature_names, n_components=None):
    """
    Realiza el análisis de componentes principales
    """
    if n_components is None:
        n_components = min(X_scaled.shape)
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Información sobre los componentes principales
    print(f"\nNúmero de componentes principales: {pca.n_components_}")
    print(f"Varianza explicada por cada componente: {pca.explained_variance_ratio_}")
    print(f"Varianza explicada acumulada: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Crear DataFrame con los componentes principales
    pc_columns = [f'PC{i+1}' for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(X_pca, columns=pc_columns)
    
    return pca, X_pca, df_pca

def visualize_pca_results(pca, X_pca, df_pca, feature_names):
    """
    Visualiza los resultados del PCA
    """
    # 1. Gráfico de varianza explicada
    plt.figure(figsize=(15, 12))
    
    # Subplot 1: Varianza explicada por cada componente
    plt.subplot(2, 3, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_)
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    
    # Subplot 2: Varianza explicada acumulada
    plt.subplot(2, 3, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% varianza')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90% varianza')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Acumulada')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Scree plot
    plt.subplot(2, 3, 3)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'ro-')
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Scree Plot')
    plt.grid(True)
    
    # Subplot 4: Biplot de los primeros 2 componentes
    plt.subplot(2, 3, 4)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
    plt.title('Biplot PC1 vs PC2')
    plt.grid(True)
    
    # Subplot 5: Loadings de las variables
    plt.subplot(2, 3, 5)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Mostrar loadings para PC1 y PC2
    for i, (feature, loading) in enumerate(zip(feature_names, loadings)):
        plt.arrow(0, 0, loading[0], loading[1], 
                 head_width=0.01, head_length=0.01, fc='blue', ec='blue')
        plt.text(loading[0]*1.1, loading[1]*1.1, feature, 
                fontsize=8, ha='center', va='center')
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Loadings de Variables')
    plt.grid(True)
    
    # Subplot 6: Heatmap de componentes principales
    plt.subplot(2, 3, 6)
    # Mostrar solo los primeros 5 componentes para mejor visualización
    n_components_show = min(5, pca.n_components_)
    components_df = pd.DataFrame(
        pca.components_[:n_components_show].T,
        columns=[f'PC{i+1}' for i in range(n_components_show)],
        index=feature_names
    )
    
    sns.heatmap(components_df, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', cbar_kws={'label': 'Loading'})
    plt.title('Loadings de Variables por Componente')
    plt.ylabel('Variables')
    plt.xlabel('Componentes Principales')
    
    plt.tight_layout()
    plt.show()

def analyze_components(pca, feature_names, n_components=5):
    """
    Analiza las contribuciones de las variables a cada componente principal
    """
    print("\n" + "="*50)
    print("ANÁLISIS DE COMPONENTES PRINCIPALES")
    print("="*50)
    
    for i in range(min(n_components, pca.n_components_)):
        print(f"\nComponente Principal {i+1}:")
        print(f"Varianza explicada: {pca.explained_variance_ratio_[i]:.4f} ({pca.explained_variance_ratio_[i]*100:.2f}%)")
        
        # Obtener loadings y ordenar por valor absoluto
        loadings = pca.components_[i]
        feature_loadings = list(zip(feature_names, loadings))
        feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("Variables más influyentes:")
        for j, (feature, loading) in enumerate(feature_loadings[:5]):
            print(f"  {j+1}. {feature}: {loading:.4f}")

def recommend_components(pca, threshold=0.8):
    """
    Recomienda el número de componentes basado en la varianza explicada
    """
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components_80 = np.argmax(cumvar >= 0.8) + 1
    n_components_90 = np.argmax(cumvar >= 0.9) + 1
    
    print(f"\n" + "="*50)
    print("RECOMENDACIONES")
    print("="*50)
    print(f"Componentes para explicar 80% de varianza: {n_components_80}")
    print(f"Componentes para explicar 90% de varianza: {n_components_90}")
    print(f"Varianza total explicada con {pca.n_components_} componentes: {cumvar[-1]:.4f}")
    
    return n_components_80, n_components_90

def reduce_dataset(df, n_samples=10000, random_state=42):
    """
    Reduce el dataset a un número específico de filas
    """
    if len(df) > n_samples:
        df_reduced = df.sample(n=n_samples, random_state=random_state)
        print(f"Dataset reducido de {len(df)} a {n_samples} filas")
    else:
        df_reduced = df.copy()
        print(f"Dataset mantenido con {len(df)} filas (menor a {n_samples})")
    
    return df_reduced.reset_index(drop=True)

def elbow_method(X_scaled, max_k=15, random_state=42):
    """
    Aplica el método del codo para determinar el número óptimo de clusters
    """
    print("\n" + "="*50)
    print("MÉTODO DEL CODO PARA K-MEANS")
    print("="*50)
    
    # Calcular WCSS (Within-Cluster Sum of Squares) para diferentes valores de k
    wcss = []
    k_range = range(1, max_k + 1)
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
        # Calcular silhouette score (solo para k > 1)
        if k > 1:
            silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
            print(f"K={k}: WCSS={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.4f}")
        else:
            silhouette_scores.append(0)
            print(f"K={k}: WCSS={kmeans.inertia_:.2f}")
    
    return k_range, wcss, silhouette_scores

def plot_elbow_method(k_range, wcss, silhouette_scores):
    """
    Visualiza el método del codo y los silhouette scores
    """
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Método del codo
    plt.subplot(1, 3, 1)
    plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Método del Codo')
    plt.grid(True, alpha=0.3)
    
    # Agregar anotaciones
    for i, (k, w) in enumerate(zip(k_range, wcss)):
        if i % 2 == 0:  # Mostrar solo algunos valores para evitar saturación
            plt.annotate(f'{w:.0f}', (k, w), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
    # Subplot 2: Silhouette Scores
    plt.subplot(1, 3, 2)
    plt.plot(k_range[1:], silhouette_scores[1:], 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores por K')
    plt.grid(True, alpha=0.3)
    
    # Encontrar el mejor k según silhouette score
    best_k_silhouette = k_range[1:][np.argmax(silhouette_scores[1:])]
    plt.axvline(x=best_k_silhouette, color='red', linestyle='--', alpha=0.7)
    plt.text(best_k_silhouette + 0.1, max(silhouette_scores[1:]) * 0.9, 
             f'Mejor k={best_k_silhouette}', rotation=90)
    
    # Subplot 3: Comparación combinada
    plt.subplot(1, 3, 3)
    
    # Normalizar valores para comparar en la misma escala
    wcss_norm = np.array(wcss) / max(wcss)
    silhouette_norm = np.array(silhouette_scores) / max(silhouette_scores[1:]) if max(silhouette_scores[1:]) > 0 else np.array(silhouette_scores)
    
    plt.plot(k_range, wcss_norm, 'b-', label='WCSS (normalizado)', linewidth=2)
    plt.plot(k_range, silhouette_norm, 'r-', label='Silhouette (normalizado)', linewidth=2)
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Valores Normalizados')
    plt.title('Comparación de Métrica')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return best_k_silhouette

def recommend_optimal_k(k_range, wcss, silhouette_scores):
    """
    Recomienda el número óptimo de clusters basado en diferentes criterios
    """
    print("\n" + "="*50)
    print("RECOMENDACIONES PARA NÚMERO ÓPTIMO DE CLUSTERS")
    print("="*50)
    
    # 1. Método del codo (buscar el "codo" en la curva)
    # Calcular las diferencias de segundo orden
    diffs = np.diff(wcss)
    second_diffs = np.diff(diffs)
    elbow_k = np.argmax(second_diffs) + 2  # +2 porque perdemos 2 elementos con las diferencias
    
    # 2. Mejor k según silhouette score
    if len(silhouette_scores) > 1 and max(silhouette_scores[1:]) > 0:
        best_silhouette_k = k_range[1:][np.argmax(silhouette_scores[1:])]
    else:
        best_silhouette_k = 2
    
    print(f"Recomendación método del codo: k = {elbow_k}")
    print(f"Recomendación silhouette score: k = {best_silhouette_k}")
    print(f"Mejor silhouette score: {max(silhouette_scores[1:]):.4f}")
    
    # Mostrar top 3 k con mejores silhouette scores
    silhouette_with_k = [(k, score) for k, score in zip(k_range[1:], silhouette_scores[1:])]
    silhouette_with_k.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 3 k con mejores silhouette scores:")
    for i, (k, score) in enumerate(silhouette_with_k[:3]):
        print(f"  {i+1}. k={k}: {score:.4f}")
    
    return elbow_k, best_silhouette_k

def perform_clustering_analysis(X_scaled, optimal_k, feature_names, random_state=42):
    """
    Realiza clustering con el k óptimo y analiza los resultados
    """
    print(f"\n" + "="*50)
    print(f"CLUSTERING CON K={optimal_k}")
    print("="*50)
    
    # Aplicar K-means con k óptimo
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Métricas de evaluación
    wcss = kmeans.inertia_
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    print(f"WCSS: {wcss:.2f}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Mostrar distribución de clusters
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nDistribución de clusters:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} puntos ({count/len(cluster_labels)*100:.1f}%)")
    
    return kmeans, cluster_labels

# Modificar la función main() para incluir el nuevo análisis
def main():
    """
    Función principal que ejecuta todo el análisis PCA y clustering
    """
    print("ANÁLISIS PCA Y CLUSTERING - DATASET DE SALUD Y ESTILO DE VIDA")
    print("="*70)
    
    # 1. Cargar y explorar datos
    file_path = 'health_lifestyle_dataset.csv'
    df = load_and_preprocess_data(file_path)
    
    # 2. Reducir dataset a 10,000 filas
    df = reduce_dataset(df, n_samples=10000)
    
    # 3. Preparar datos para PCA
    X_scaled, feature_names, scaler = prepare_data_for_pca(df)
    
    # 4. Realizar análisis PCA
    pca, X_pca, df_pca = perform_pca_analysis(X_scaled, feature_names)
    
    # 5. Visualizar resultados PCA
    visualize_pca_results(pca, X_pca, df_pca, feature_names)
    
    # 6. Analizar componentes principales
    analyze_components(pca, feature_names)
    
    # 7. Recomendaciones PCA
    n_80, n_90 = recommend_components(pca)
    
    # 8. Método del codo para clustering
    k_range, wcss, silhouette_scores = elbow_method(X_scaled, max_k=15)
    
    # 9. Visualizar método del codo
    best_k_silhouette = plot_elbow_method(k_range, wcss, silhouette_scores)
    
    # 10. Recomendar k óptimo
    elbow_k, silhouette_k = recommend_optimal_k(k_range, wcss, silhouette_scores)
    
    # 11. Realizar clustering con k óptimo
    optimal_k = silhouette_k  # Usar el k con mejor silhouette score
    kmeans, cluster_labels = perform_clustering_analysis(X_scaled, optimal_k, feature_names)
    
    # 12. Guardar resultados
    df_pca.to_csv('pca_results.csv', index=False)
    
    # Agregar labels de cluster a los resultados
    df_results = df.copy()
    df_results['cluster'] = cluster_labels
    df_results.to_csv('clustering_results.csv', index=False)
    
    print(f"\nResultados guardados:")
    print(f"  - PCA: 'pca_results.csv'")
    print(f"  - Clustering: 'clustering_results.csv'")
    
    # 13. Crear PCA reducido con componentes recomendados
    print(f"\n" + "="*50)
    print("ANÁLISIS PCA REDUCIDO (Componentes Óptimos)")
    print("="*50)
    
    pca_reduced = PCA(n_components=n_80)
    X_pca_reduced = pca_reduced.fit_transform(X_scaled)
    
    print(f"Usando {n_80} componentes principales:")
    print(f"Varianza explicada total: {sum(pca_reduced.explained_variance_ratio_):.4f}")
    
    # Guardar PCA reducido
    pc_columns_reduced = [f'PC{i+1}' for i in range(n_80)]
    df_pca_reduced = pd.DataFrame(X_pca_reduced, columns=pc_columns_reduced)
    df_pca_reduced['cluster'] = cluster_labels
    df_pca_reduced.to_csv('pca_reduced_results.csv', index=False)
    
    return df, pca, X_pca, feature_names, scaler, kmeans, cluster_labels

if __name__ == "__main__":
    # Ejecutar análisis completo
    df, pca, X_pca, feature_names, scaler, kmeans, cluster_labels = main()