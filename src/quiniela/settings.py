#############
# libraries #
#############

"""
Archivo de configuración global del proyecto Quiniela.

Define las rutas y parámetros generales utilizados por los distintos módulos del proyecto.
Todas las rutas se gestionan mediante objetos Path para asegurar compatibilidad entre sistemas operativos.
"""

from pathlib import Path

# Directorio de trabajo actual
CWD = Path()

# Ruta al archivo de base de datos SQLite
DATABASE_PATH = CWD / "laliga.sqlite"

# Ruta al directorio donde se almacenarán los modelos entrenados
MODELS_PATH = CWD / "models"

# Ruta al directorio donde se almacenarán los registros de ejecución
LOGS_PATH = CWD / "logs"

# Creación automática de los directorios necesarios si no existen
MODELS_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)

