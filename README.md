# Electocluster

Prototipo de clustering con ponderación aplicada para la segmentación de votantes.

## Descripción

Este repositorio contiene la estructura del proyecto para el desarrollo del prototipo y sus respectivos avances.

## Directorios principales

- `data/raw/` - Donde se suben los datasets (descargar de la fuente especificada).
- `data/processed/` - Salidas normalizadas.
- `notebooks/` - Notebooks interactivos de cada iteración y de la validación del entorno.
- `src/` - Código fuente del prototipo; comprende módulos de clustering, preprocesamiento, ponderación y reportes.

## Datos (LAPOP)

Como muestra de referencia se usa el dataset del Barómetro de las Américas para Bolivia 2023 proporcionado por LAPOP.

El archivo de origen no se versiona. Una vez obtenido mediante la vía documentada
en los notebooks de las Iteraciones 1 y 2, debe ubicarse en:

```text
data/raw/lapop_bolivia_2023.dta
```

El preprocesamiento genera el archivo consumido por los algoritmos de clustering:

```text
data/processed/lapop_bolivia_2023_processed.csv
```

Ambas rutas están excluidas de Git. Los notebooks de Iteración 1 documentan la
validación y el preprocesamiento; los de Iteración 2 documentan las pruebas con
el dataset procesado. No se deben sustituir esos archivos por datos de otra
edición de LAPOP si se busca reproducir la evidencia canónica.

## Dependencias e instalación (PowerShell)

Se recomienda crear un entorno virtual limpio. En PowerShell (Windows):

```powershell
cd C:\electocluster
# Crear entorno virtual (usar nombre que prefieras)
python -m venv .venv

# Activar el entorno
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# (Opcional) Si ya usas otro virtualenv como 'venv_clustering', actívalo en lugar de crear uno nuevo.
```

## Ejecutar notebooks

Con el entorno activado:

```powershell
pip install jupyterlab notebook  # si no están instalados
jupyter lab
# o
jupyter notebook
```

## Ejecutar la interfaz

La interfaz del prototipo está construida con Streamlit y su punto de entrada es `src/app.py`.

Con el entorno activado, ejecuta desde la raíz del proyecto:

```powershell
cd C:\electocluster
streamlit run src/app.py
```

Alternativa equivalente:

```powershell
python -m streamlit run src/app.py
```

Luego abre en el navegador la URL local que Streamlit mostrará en consola, normalmente `http://localhost:8501`.

### Ejecutar notebook de validación del entorno

Abre `notebooks/01_pruebas_validacion` y ejecuta las celdas. Todos los recursos generados por las pruebas se guardarán allí mismo.

## Validación y reproducibilidad

Los notebooks de cada iteración son la evidencia principal de las pruebas
metodológicas. Cada uno documenta si trabaja con datos sintéticos, con el
dataset LAPOP crudo o con el dataset ya procesado.

Como verificación técnica complementaria, desde la raíz del repositorio pueden
ejecutarse las siguientes comprobaciones:

```powershell
# Contratos de módulos y artefactos locales ya existentes.
python tests/validate_iteration5.py --quick

# Ejercita los tres algoritmos sobre una muestra temporal del dataset procesado.
python tests/validate_iteration5.py --sample --sample-size 300

# Reproduce preprocesamiento en una salida temporal y la batería canónica completa.
python tests/validate_iteration5.py --full

# Comprueba la carga de las páginas de Streamlit; requiere resultados y dataset procesado locales.
python tests/validate_iteration5.py --streamlit
```

Las opciones `--sample`, `--full` y `--streamlit` requieren los datos locales
indicados arriba. Ninguna prueba versiona datos ni sustituye los resultados
históricos. La comprobación de instalación desde un clon limpio queda pendiente
de validación manual.

## Batería canónica de Iteración 5

En la página **Ejecución**, selecciona los tres algoritmos y usa la fuente
**Canónica** en todos: WKMedoids, W-Hierarchical Clustering y W-DBSCAN. Al
ejecutarlos como conjunto, la aplicación conserva los runs individuales y
genera una evidencia consolidada en `results/suites/`.

La evidencia incluye parámetros, trazabilidad, tiempos, métricas, cobertura,
ruido y perfiles semánticos. Puede abrirse y descargarse desde **Resultados**.
W-DBSCAN se conserva con su espacio y población de evaluación propios; por
ello la interfaz no produce un ranking global cuando las métricas no son
directamente comparables.
