# Preexperimentos

Esta carpeta conserva una copia de los notebooks de exploración y pruebas preliminares de ElectroCluster, junto con sus imágenes auxiliares. Es un respaldo histórico del trabajo anterior a las iteraciones del prototipo. Los archivos se incorporaron con sus salidas guardadas y sin volver a ejecutar las celdas.

## Procedencia

- Repositorio original: https://github.com/PHansLM/electocluster-preexperiment
- Carpeta de origen: `notebooks/`.
- Commit de origen: `fbdc680d85a4f7de596d27f508fa82775bf84b50`.
- Fecha de incorporación a este repositorio: 27 de noviembre de 2026.

Los seis archivos procedentes de esa carpeta se copiaron sin modificar su contenido. El README se añadió para explicar su función dentro del proyecto. La fecha de incorporación y la fecha del commit no deben interpretarse como fechas de ejecución de las celdas. Las salidas y los contadores conservados no acreditan por sí solos cuándo se ejecutaron.

## Contenido

| Archivo | Contenido |
| --- | --- |
| `01_prueba_entorno.ipynb` | Comprobación inicial del entorno y de las versiones de NumPy, pandas y scikit-learn. |
| `02_pruebas_preprocesamiento.ipynb` | Pruebas con datos sintéticos de normalización, clustering, métricas y visualización. El bloque «PRUEBA 10» registra la lectura de LAPOP Bolivia 2023 y una comprobación preliminar de variables de interés. |
| `test_scatter.png` | Gráfico de dispersión de las pruebas preliminares. |
| `test_boxplot.png` | Diagrama de cajas de las pruebas preliminares. |
| `test_dendrograma.png` | Dendrograma de las pruebas preliminares. |
| `test_heatmap.png` | Mapa de correlaciones de las pruebas preliminares. |

Las figuras de estas pruebas corresponden a los datos sintéticos del notebook. No representan segmentos de la muestra LAPOP. La salida guardada de «PRUEBA 10» muestra la carga de 1706 registros y 208 variables, junto con la disponibilidad de algunas variables. Ese bloque no constituye una validación completa de las 40 variables fuente de la selección final.

## Consulta y ejecución

Los notebooks pueden consultarse con sus salidas originales. No es necesario ejecutarlos para revisar esta evidencia histórica.

Se conservaron las rutas y configuraciones originales. En particular, «PRUEBA 10» utiliza `../data/raw/lapop_bolivia_2023.dta`, una ruta relativa a la ubicación que tenía el notebook en el repositorio de origen. Desde esta carpeta, la ubicación del archivo base del proyecto sería `../../data/raw/lapop_bolivia_2023.dta`. Si se desea repetir la prueba, debe prepararse una copia de trabajo y ajustar esa ruta, conservando este respaldo sin cambios.

El dataset no se incluye en esta carpeta. Cada usuario debe obtenerlo directamente de LAPOP y colocarlo en la ubicación correspondiente. Las dependencias del entorno original están documentadas en el `requirements.txt` del repositorio de origen. No se actualizaron las salidas para adaptarlas a las versiones actuales de las bibliotecas.

## Relación con el desarrollo posterior

La exploración y las comprobaciones posteriores se encuentran en:

- `../enviroment_validation/03_validacion_dataset.ipynb`: contraste de la muestra con el reporte técnico, revisión de rangos y faltantes y comprobación de una selección preliminar de 39 variables.
- `../Iteration1/01_preprocesamiento_pasos_dataset.ipynb`: carga, limpieza y transformación por etapas, con una salida de selección de 40 variables fuente.
- `../pruebas_adicionales/02_ficha_metodologica_28_variables.ipynb`: descripción de las características y correspondencia entre las 40 variables fuente y las 28 características finales.
- `../pruebas_adicionales/03_descripcion_y_calidad_muestra.ipynb`: descripción de la muestra, distribuciones y examen de faltantes e inaplicabilidad.

Las configuraciones, selecciones y métricas de los preexperimentos corresponden a esa etapa preliminar. Las configuraciones de referencia del prototipo se documentan en las iteraciones posteriores.
