# Resultados generados localmente

Este directorio contiene las salidas producidas por las ejecuciones del prototipo.
Se conserva en el repositorio únicamente este archivo para crear la estructura
esperada después de clonar el proyecto; los resultados concretos no se versionan.

Las ejecuciones pueden generar, según el flujo utilizado, archivos de asignaciones,
métricas, comparaciones, figuras, reportes, perfiles y resúmenes de la suite
canónica. Estos artefactos dependen del dataset disponible, la configuración, el
entorno de ejecución y la versión del código, por lo que deben regenerarse en cada
entorno cuando se requiera reproducibilidad.

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
