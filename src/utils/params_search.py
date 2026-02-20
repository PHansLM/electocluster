"""
Buscador de parámetros optimos para clustering

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def compute_kdistance_data(df: np.ndarray, k: int) -> np.ndarray:
    """
    Calcula las distancias al k-ésimo vecino más cercano para cada punto.

    Args:
        df:  Matriz de features (n_samples, n_features).
        k:   Número de vecinos. Corresponde al min_samples candidato.

    Return:
        Array de distancias ordenadas de mayor a menor.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(df)  # +1 porque incluye el propio punto
    distances, _ = nbrs.kneighbors(df)
    k_distances = distances[:, k]  # distancia al k-ésimo vecino
    return np.sort(k_distances)[::-1]  # ordenar descendentemente


def find_elbow(distances: np.ndarray) -> tuple:
    """
    Detecta el codo de la curva k-distance usando máxima curvatura.

    Args:
        distances: Array de distancias ordenadas descendentemente.

    Return:
        tuple: (índice del codo, valor de eps en el codo)
    """
    n = len(distances)
    # Vector desde el primer al último punto de la curva
    p1 = np.array([0, distances[0]])
    p2 = np.array([n - 1, distances[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # Distancia perpendicular de cada punto a la línea
    perp_distances = []
    for i in range(n):
        point = np.array([i, distances[i]])
        vec_to_point = point - p1
        proj = np.dot(vec_to_point, line_vec_norm) * line_vec_norm
        perp = vec_to_point - proj
        perp_distances.append(np.linalg.norm(perp))

    elbow_idx = np.argmax(perp_distances)
    return elbow_idx, distances[elbow_idx]


def analyze_dbscan_params(df: pd.DataFrame,
                           output_path: str = None) -> dict:
    """
    Analiza el espacio de parámetros para DBSCAN y obtiene un rango apropiado.

    Genera k-distance plots para múltiples valores de k y detecta el codo de cada curva.

    Args:
        df:           DataFrame preprocesado.
        output_path:  Ruta para guardar el gráfico (opcional).

    Return:
        dict: Espacio de búsqueda recomendado.
    """
    X = df.values
    n_features = X.shape[1]
    n_samples  = X.shape[0]

    print("=" * 65)
    print("ANÁLISIS DE PARÁMETROS DBSCAN")
    print("=" * 65)
    print(f"Dataset: {n_samples} muestras × {n_features} features")

    # --------------------------------------------------------
    # Rango de k (min_samples) basado en dimensionalidad
    # Regla: min_samples >= n_features
    # Se exploran valores alrededor del umbral
    # --------------------------------------------------------
    
    # Limites inferior y superior para k
    k_min = max(5, n_features // 2)      
    k_max = n_features * 2               

    k_candidates = [
        k_min,
        n_features,
        int(n_features * 1.5),
        k_max
    ]
    # Remover duplicados y ordenar
    k_candidates = sorted(set(k_candidates))

    print(f"\nRegla empírica para min_samples (>= n_features = {n_features}):")
    print(f"  k candidatos a explorar: {k_candidates}")

    # --------------------------------------------------------
    # Calcular k-distances y detectar codos
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, len(k_candidates),
                              figsize=(5 * len(k_candidates), 5),
                              sharey=False)
    if len(k_candidates) == 1:
        axes = [axes]

    resultados = {}

    print(f"\nCodos detectados (eps candidatos por valor de k):")
    print(f"  {'k (min_samples)':<20} {'eps en codo':<15} {'interpretación'}")
    print(f"  {'-'*55}")

    for ax, k in zip(axes, k_candidates):
        distances = compute_kdistance_data(X, k)
        elbow_idx, eps_codo = find_elbow(distances)

        resultados[k] = {
            'eps_codo': round(float(eps_codo), 4),
            'elbow_idx': int(elbow_idx),
            'distancia_min': round(float(distances[-1]), 4),
            'distancia_max': round(float(distances[0]), 4),
            'distancia_media': round(float(np.mean(distances)), 4),
        }

        # Gráfico
        ax.plot(range(len(distances)), distances, linewidth=1.2, color='steelblue')
        ax.axvline(x=elbow_idx, color='red', linestyle='--', linewidth=1,
                   label=f'Codo (eps≈{eps_codo:.3f})')
        ax.axhline(y=eps_codo, color='orange', linestyle=':', linewidth=1)
        ax.set_title(f'k = {k}', fontsize=12)
        ax.set_xlabel('Puntos (ordenados)', fontsize=10)
        ax.set_ylabel('Distancia al k-ésimo vecino', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        interpretacion = (
            "conservador (más clusters, menos ruido)"
            if k == k_min else
            "agresivo (menos clusters, más ruido)"
            if k == k_max else
            "recomendado"
        )
        print(f"  k={k:<18} eps≈{eps_codo:<13.4f} {interpretacion}")

    plt.suptitle(
        f'K-Distance Plot — {n_features} features, {n_samples} muestras\n'
        f'El codo (línea roja) indica el eps natural del dataset para cada k',
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    if output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Gráfico guardado en: {output_path}")

    plt.show()

    # --------------------------------------------------------
    # Proponer espacio de búsqueda
    # --------------------------------------------------------
    eps_valores = [v['eps_codo'] for v in resultados.values()]
    eps_min_sugerido = round(min(eps_valores) * 0.7, 3)
    eps_max_sugerido = round(max(eps_valores) * 1.3, 3)
    eps_step = round((eps_max_sugerido - eps_min_sugerido) / 6, 3)

    espacio_sugerido = {
        'eps': {
            'min': eps_min_sugerido,
            'max': eps_max_sugerido,
            'step_sugerido': eps_step,
            'valores': [
                round(eps_min_sugerido + i * eps_step, 3)
                for i in range(7)
            ],
            'justificacion': (
                f'Rango derivado de los codos del k-distance plot '
                f'(±30% alrededor de los valores detectados automáticamente).'
            )
        },
        'min_samples': {
            'valores': k_candidates,
            'justificacion': (
                f'Basado en la regla empírica min_samples >= n_features ({n_features}). '
                f'Sander et al. (1998) recomiendan este umbral para datos de alta '
                f'dimensionalidad para evitar clusters triviales.'
            )
        }
    }

    print(f"\n{'=' * 65}")
    print(f"ESPACIO DE BÚSQUEDA RECOMENDADO")
    print(f"{'=' * 65}")
    print(f"  eps:          {espacio_sugerido['eps']['valores']}")
    print(f"  min_samples:  {espacio_sugerido['min_samples']['valores']}")
    print(f"\n  Combinaciones totales: "
          f"{len(espacio_sugerido['eps']['valores']) * len(k_candidates)}")
    print(f"{'=' * 65}")

    return {
        'analisis_por_k': resultados,
        'espacio_sugerido': espacio_sugerido,
        'n_features': n_features,
        'n_samples': n_samples
    }


def build_dbscan_configs(analisis: dict) -> list:
    """
    Genera la lista de configuraciones para el grid search,
    usando el espacio derivado del k-distance analysis.

    Args:
        analisis: Output de analyze_dbscan_params().

    Return:
        list: Lista de dicts {'eps': ..., 'min_samples': ...}
    """
    espacio = analisis['espacio_sugerido']
    return [
        {'eps': eps, 'min_samples': ms}
        for eps in espacio['eps']['valores']
        for ms in espacio['min_samples']['valores']
    ]