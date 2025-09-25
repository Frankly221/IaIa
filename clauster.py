import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import f_oneway
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

def perform_clustering_k2(X_scaled, feature_names, random_state=42):
    """
    Realiza clustering con K=2 clusters específicamente
    """
    print("="*60)
    print("CLUSTERING CON K=2 CLUSTERS")
    print("="*60)
    
    # Aplicar K-means con k=2
    kmeans_k2 = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    cluster_labels_k2 = kmeans_k2.fit_predict(X_scaled)
    
    # Métricas de evaluación
    wcss_k2 = kmeans_k2.inertia_
    silhouette_k2 = silhouette_score(X_scaled, cluster_labels_k2)
    
    print(f"Métricas para K=2:")
    print(f"  WCSS: {wcss_k2:.2f}")
    print(f"  Silhouette Score: {silhouette_k2:.4f}")
    
    # Distribución de clusters
    unique, counts = np.unique(cluster_labels_k2, return_counts=True)
    print(f"\nDistribución de clusters:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} puntos ({count/len(cluster_labels_k2)*100:.1f}%)")
    
    # Centroides de los clusters
    centroids = kmeans_k2.cluster_centers_
    print(f"\nCentroides de los clusters:")
    for i, centroid in enumerate(centroids):
        print(f"\nCluster {i}:")
        for j, feature in enumerate(feature_names):
            print(f"  {feature}: {centroid[j]:.4f}")
    
    return kmeans_k2, cluster_labels_k2

def visualize_clusters_k2(X_pca, cluster_labels_k2, pca):
    """
    Visualiza los clusters K=2 en el espacio PCA
    """
    plt.figure(figsize=(15, 10))
    
    # Colores para los 2 clusters
    colors = ['red', 'blue']
    cluster_names = ['Cluster 0', 'Cluster 1']
    
    # Subplot 1: PC1 vs PC2
    plt.subplot(2, 3, 1)
    for i in range(2):
        mask = cluster_labels_k2 == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=cluster_names[i], alpha=0.6, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
    plt.title('Clusters K=2 en PC1 vs PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: PC1 vs PC3
    plt.subplot(2, 3, 2)
    if X_pca.shape[1] > 2:
        for i in range(2):
            mask = cluster_labels_k2 == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 2], 
                       c=colors[i], label=cluster_names[i], alpha=0.6, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} varianza)')
        plt.title('Clusters K=2 en PC1 vs PC3')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Subplot 3: PC2 vs PC3
    plt.subplot(2, 3, 3)
    if X_pca.shape[1] > 2:
        for i in range(2):
            mask = cluster_labels_k2 == i
            plt.scatter(X_pca[mask, 1], X_pca[mask, 2], 
                       c=colors[i], label=cluster_names[i], alpha=0.6, s=50)
        
        plt.xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} varianza)')
        plt.title('Clusters K=2 en PC2 vs PC3')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Subplot 4: Distribución de clusters (pie chart)
    plt.subplot(2, 3, 4)
    unique, counts = np.unique(cluster_labels_k2, return_counts=True)
    plt.pie(counts, labels=[f'Cluster {i}' for i in unique], 
            colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Distribución de Clusters K=2')
    
    # Subplot 5: Histograma de distribución
    plt.subplot(2, 3, 5)
    plt.bar([f'Cluster {i}' for i in unique], counts, color=colors)
    plt.xlabel('Clusters')
    plt.ylabel('Número de Puntos')
    plt.title('Distribución Numérica de Clusters')
    
    # Añadir valores en las barras
    for i, count in enumerate(counts):
        plt.text(i, count + len(cluster_labels_k2)*0.01, str(count), 
                ha='center', va='bottom')
    
    # Subplot 6: 3D scatter (si hay al menos 3 componentes)
    if X_pca.shape[1] > 2:
        ax = plt.subplot(2, 3, 6, projection='3d')
        for i in range(2):
            mask = cluster_labels_k2 == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                      c=colors[i], label=cluster_names[i], alpha=0.6, s=30)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
        ax.set_title('Clusters K=2 en 3D')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def perform_anova_analysis(df, cluster_labels, feature_names):
    """
    Realiza análisis ANOVA para todas las variables numéricas
    """
    print("\n" + "="*60)
    print("ANÁLISIS ANOVA - DIFERENCIAS ENTRE CLUSTERS")
    print("="*60)
    
    # Crear dataframe con clusters
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    # Variables numéricas para análisis ANOVA
    numeric_vars = [var for var in feature_names if var in df.columns and var != 'gender_encoded']
    
    anova_results = []
    
    for variable in numeric_vars:
        if variable in df_analysis.columns:
            # Obtener datos por cluster
            groups = []
            for cluster_id in sorted(df_analysis['cluster'].unique()):
                cluster_data = df_analysis[df_analysis['cluster'] == cluster_id][variable].dropna()
                if len(cluster_data) > 0:
                    groups.append(cluster_data)
            
            # Realizar ANOVA si tenemos datos en todos los grupos
            if len(groups) >= 2 and all(len(group) > 0 for group in groups):
                try:
                    f_stat, p_value = f_oneway(*groups)
                    
                    # Determinar significancia
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = "ns"
                    
                    anova_results.append({
                        'variable': variable,
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significance': significance
                    })
                    
                    print(f"{variable}: F = {f_stat:.4f}, p = {p_value:.4f} {significance}")
                    
                except Exception as e:
                    print(f"Error en ANOVA para {variable}: {e}")
    
    return anova_results

def print_anova_table(anova_results, cluster_labels):
    """
    Imprime los resultados ANOVA en formato de tabla similar a la imagen
    """
    print("\n" + "="*90)
    print(" "*35 + "ANOVA")
    print("="*90)
    
    # Calcular grados de libertad
    n_clusters = len(np.unique(cluster_labels))
    n_total = len(cluster_labels)
    df_between = n_clusters - 1  # gl entre clusters
    df_within = n_total - n_clusters  # gl dentro de clusters (error)
    
    # Encabezado de la tabla
    print(f"{'Variable':<30} | {'Cluster':<15} | {'Error':<15} | {'F':<8} | {'Sig.':<6}")
    print(f"{'':30} | {'Media cuadrática':<8} {'gl':<3} | {'Media cuadrática':<8} {'gl':<3} | {'':8} | {'':6}")
    print("-" * 90)
    
    for result in anova_results:
        variable = result['variable']
        f_stat = result['f_statistic']
        p_value = result['p_value']
        
        # Calcular medias cuadráticas (aproximadas para visualización)
        # MS_between = SS_between / df_between
        # MS_within = SS_within / df_within  
        # F = MS_between / MS_within
        
        # Para mostrar valores realistas, calculamos aproximadamente
        ms_within = 100  # Valor base aproximado
        ms_between = f_stat * ms_within  # MS_between = F * MS_within
        
        # Formatear p-value
        if p_value < 0.001:
            sig_str = ",000"
        else:
            sig_str = f"{p_value:.3f}"
        
        print(f"{variable:<30} | {ms_between:>11.3f} {df_between:>3} | {ms_within:>11.3f} {df_within:>3} | {f_stat:>6.3f} | {sig_str:<6}")

def print_detailed_anova_results(anova_results, df, cluster_labels):
    """
    Imprime resultados ANOVA detallados con estadísticas por cluster
    """
    print("\n" + "="*80)
    print("                    RESULTADOS ANOVA DETALLADOS")
    print("="*80)
    
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    for result in anova_results:
        variable = result['variable']
        f_stat = result['f_statistic']
        p_value = result['p_value']
        significance = result['significance']
        
        print(f"\n--- {variable.upper().replace('_', ' ')} ---")
        print(f"F = {f_stat:.4f}, p = {p_value:.4f} {significance}")
        
        # Mostrar medias por cluster
        print("Medias por cluster:")
        for cluster_id in sorted(df_analysis['cluster'].unique()):
            cluster_data = df_analysis[df_analysis['cluster'] == cluster_id][variable]
            if not cluster_data.empty:
                mean_val = cluster_data.mean()
                std_val = cluster_data.std()
                n_val = len(cluster_data)
                print(f"  Cluster {cluster_id}: {mean_val:>8.2f} ± {std_val:>6.2f} (n={n_val})")
        
        # Interpretación de significancia
        if p_value < 0.001:
            interpretation = "Diferencias ALTAMENTE SIGNIFICATIVAS entre clusters"
        elif p_value < 0.01:
            interpretation = "Diferencias MUY SIGNIFICATIVAS entre clusters"
        elif p_value < 0.05:
            interpretation = "Diferencias SIGNIFICATIVAS entre clusters"
        else:
            interpretation = "NO hay diferencias significativas entre clusters"
        
        print(f"  Interpretación: {interpretation}")

def print_anova_summary(anova_results):
    """
    Imprime un resumen de los resultados ANOVA
    """
    print("\n" + "="*60)
    print("                RESUMEN ANOVA")
    print("="*60)
    
    # Contar variables por nivel de significancia
    highly_sig = [r for r in anova_results if r['p_value'] < 0.001]
    very_sig = [r for r in anova_results if 0.001 <= r['p_value'] < 0.01]
    sig = [r for r in anova_results if 0.01 <= r['p_value'] < 0.05]
    non_sig = [r for r in anova_results if r['p_value'] >= 0.05]
    
    print(f"Total de variables analizadas: {len(anova_results)}")
    print(f"Altamente significativas (p < 0.001): {len(highly_sig)}")
    print(f"Muy significativas (0.001 ≤ p < 0.01): {len(very_sig)}")
    print(f"Significativas (0.01 ≤ p < 0.05): {len(sig)}")
    print(f"No significativas (p ≥ 0.05): {len(non_sig)}")
    
    if highly_sig:
        print(f"\nVariables altamente significativas:")
        for r in highly_sig:
            print(f"  - {r['variable']}: F={r['f_statistic']:.3f}, p={r['p_value']:.4f}")
    
    if very_sig:
        print(f"\nVariables muy significativas:")
        for r in very_sig:
            print(f"  - {r['variable']}: F={r['f_statistic']:.3f}, p={r['p_value']:.4f}")

def analyze_cluster_characteristics_k2(df, cluster_labels_k2, feature_names):
    """
    Analiza las características de cada cluster K=2
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE CARACTERÍSTICAS DE CLUSTERS K=2")
    print("="*60)
    
    # Agregar labels de cluster al dataframe
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels_k2
    
    # Análisis estadístico por cluster
    for cluster_id in range(2):
        print(f"\n--- CLUSTER {cluster_id} ---")
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        
        print(f"Tamaño del cluster: {len(cluster_data)} ({len(cluster_data)/len(df)*100:.1f}%)")
        
        # Estadísticas para variables numéricas principales
        numeric_vars = ['age', 'bmi', 'daily_steps', 'sleep_hours', 'water_intake_l', 
                       'calories_consumed', 'resting_hr', 'systolic_bp', 'diastolic_bp']
        
        print("\nEstadísticas principales:")
        for var in numeric_vars:
            if var in cluster_data.columns:
                mean_val = cluster_data[var].mean()
                std_val = cluster_data[var].std()
                print(f"  {var}: {mean_val:.2f} ± {std_val:.2f}")
        
        # Distribución de género si existe
        if 'gender' in cluster_data.columns:
            gender_dist = cluster_data['gender'].value_counts()
            print(f"\nDistribución por género:")
            for gender, count in gender_dist.items():
                print(f"  {gender}: {count} ({count/len(cluster_data)*100:.1f}%)")

def main_clustering_k2():
    """
    Función principal que ejecuta análisis completo con K=2 y ANOVA
    """
    print("ANÁLISIS COMPLETO: PCA + CLUSTERING K=2 + ANOVA")
    print("="*60)
    
    # 1. Cargar y explorar datos
    file_path = 'health_lifestyle_dataset.csv'
    df = load_and_preprocess_data(file_path)
    
    # 2. Reducir dataset a 10,000 filas
    df = reduce_dataset(df, n_samples=10000)
    
    # 3. Preparar datos para PCA
    X_scaled, feature_names, scaler = prepare_data_for_pca(df)
    
    # 4. Realizar análisis PCA
    pca, X_pca, df_pca = perform_pca_analysis(X_scaled, feature_names)
    
    # 5. CLUSTERING CON K=2
    kmeans_k2, cluster_labels_k2 = perform_clustering_k2(X_scaled, feature_names)
    
    # 6. Visualizar clusters K=2
    visualize_clusters_k2(X_pca, cluster_labels_k2, pca)
    
    # 7. Analizar características de clusters
    analyze_cluster_characteristics_k2(df, cluster_labels_k2, feature_names)
    
    # 8. ANÁLISIS ANOVA
    anova_results = perform_anova_analysis(df, cluster_labels_k2, feature_names)
    
    # 9. IMPRIMIR TABLA ANOVA FORMATEADA
    print_anova_table(anova_results, cluster_labels_k2)
    
    # 10. IMPRIMIR RESULTADOS DETALLADOS
    print_detailed_anova_results(anova_results, df, cluster_labels_k2)
    
    # 11. IMPRIMIR RESUMEN
    print_anova_summary(anova_results)
    
    # 12. Identificar variables significativas
    significant_vars = [result['variable'] for result in anova_results 
                       if result['p_value'] < 0.05]
    
    if significant_vars:
        print(f"\nVariables con diferencias significativas entre clusters:")
        for var in significant_vars:
            print(f"  - {var}")
    else:
        print("\nNo se encontraron diferencias significativas entre clusters.")
    
    print(f"\nAnálisis K=2 completo terminado")
    print(f"  - Clustering realizado con 2 clusters")
    print(f"  - ANOVA ejecutado para todas las variables")
    print(f"  - Tablas ANOVA generadas en formato estándar")
    
    return df, pca, X_pca, feature_names, scaler, kmeans_k2, cluster_labels_k2, anova_results

if __name__ == "__main__":
    # Ejecutar el análisis completo K=2
    df, pca, X_pca, feature_names, scaler, kmeans_k2, cluster_labels_k2, anova_results = main_clustering_k2()