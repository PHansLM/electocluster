# Resultados generados localmente

Este directorio contiene las salidas producidas por las ejecuciones del prototipo.
Como evidencia técnica de la Iteración 5, el repositorio conserva únicamente el
paquete canónico generado el 3 de septiembre de 2026: su resumen consolidado, los
tres registros individuales de métricas y las tres asignaciones asociadas. El resto
de los resultados locales permanece excluido mediante `.gitignore`.

Las ejecuciones pueden generar, según el flujo utilizado, archivos de asignaciones,
métricas, comparaciones, figuras, reportes, perfiles y resúmenes de la suite
canónica. Estos artefactos dependen del dataset disponible, la configuración, el
entorno de ejecución y la versión del código, por lo que deben regenerarse en cada
entorno cuando se requiera reproducibilidad.

Los siete archivos versionados comparten los mismos identificadores de ejecución.
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
