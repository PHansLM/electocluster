# Electocluster

Prototipo de clustering con ponderación aplicada para la segmentación de votantes.

## Descripción

Este repositorio contiene la estructura del proyecto para el desarrollo del prototipo y sus respecticos avances.

## Directorios principales

- `data/raw/` — Donde se suben los dataset (Descargar de la fuente especificada).
- `data/processed/` — Salidas normalizadas.
- `notebooks/` — Notebooks interactivos de cada iteración y de la validacion del entorno.
- `src/` — Código fuente del prototipo, comprende modulos de clustering, preprocesamiento, ponderación y reportes.

## Datos (LAPOP)

Como muestra de referencia se usa el dataset del Barometro de las Americas para Bolivia 2023 proporcionado por LAPOP.


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

### Ejecutar notebook de validación del entorno 

Abre `notebooks/01_pruebas_validacion` y ejecuta las celdas.Todos los recursos generados por las pruebas se guardaran alli mismo.
