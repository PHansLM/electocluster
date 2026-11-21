# Resultados generados localmente

Este directorio contiene las salidas producidas por las ejecuciones del prototipo.
Como evidencia técnica de la Iteración 5, el repositorio conserva el
paquete canónico generado el 3 de septiembre de 2026: su resumen consolidado, los
tres registros individuales de métricas y las tres asignaciones asociadas. También
conserva la evidencia agregada de sensibilidad a la ponderación descrita más abajo.
Las demás salidas locales permanecen excluidas mediante `.gitignore`.

Las ejecuciones pueden generar, según el flujo utilizado, archivos de asignaciones,
métricas, comparaciones, figuras, reportes, perfiles y resúmenes de la suite
canónica. Estos artefactos dependen del dataset disponible, la configuración, el
entorno de ejecución y la versión del código, por lo que deben regenerarse en cada
entorno cuando se requiera reproducibilidad.

Los siete archivos canónicos versionados comparten los mismos identificadores de ejecución.
El resumen se encuentra en `suites/`, la procedencia y las métricas en `metrics/`,
y las etiquetas de cluster en `assignments/`. No contienen el dataset original ni
las 28 características procesadas por observación.

Para ejecutar la comprobación canónica en un entorno con el dataset procesado:

```powershell
python tests\validate_iteration5.py --sample --sample-size 300
```

Para una ejecución completa, se requiere además el dataset original de LAPOP en la
ruta documentada bajo `data/raw/`:

```powershell
python tests\validate_iteration5.py --full
```

La procedencia de cada resultado persistido incluye hashes del dataset y de la
configuración, parámetros, entorno y condiciones de evaluación. Consulte el manual
técnico para el procedimiento completo de instalación y reproducción.

## Sensibilidad a la ponderación

`pruebas_sensibilidad_ponderacion/` conserva los resultados agregados de ocho
escenarios con parámetros fijos: pesos, métricas, comparación de asignaciones,
tamaños, cobertura y perfiles, junto con sus figuras y manifiesto. El procedimiento
y la síntesis se encuentran en
`notebooks/pruebas_sensibilidad_ponderacion/01_sensibilidad_ponderaciones_parametros_fijos.ipynb`.
El notebook verifica la referencia mediante las tres asignaciones canónicas de
`assignments/`; esos archivos forman parte de sus entradas.

El manifiesto distingue todas las salidas generadas localmente de los archivos
seleccionados para publicación. `etiquetas.csv`, con asignaciones por observación,
y `sintesis_metodologica.md`, cuyo contenido se muestra también en el notebook,
permanecen locales. No se distribuyen los datos originales ni procesados.
Para reejecutar el análisis se requiere el dataset procesado y el entorno indicados
en el notebook; las salidas locales se regeneran y se comprueban sus hashes.
