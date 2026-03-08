'''
    Clase para gestionar pesos asignados a cada característica.
'''

from datetime import datetime
import json
from pathlib import Path
from typing import Optional

from src.utils import constants

# Constantes
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "schemas" / "feature_weights.json"
DEFAULT_WEIGHT = 0.3

class WeightManager:
    
    '''
        WeightManager: Clase para gestionar pesos asignados. Permite:
        - Cargar, modificar y persistir pesos en un archivo JSON
        - Validar pesos 
    '''

    def __init__(self, weight_path_: Optional [str | Path] = None):

        '''
            Inicializa la clase cargando el JSON o creandolo

            Args:
                weight_path_ (str | Path, opcional): Ruta al JSON de pesos. Si no se proporciona, se usa DEFAULT_WEIGHTS_PATH.
        '''
        self.weights_path_: Path = Path(weight_path_) if weight_path_ else DEFAULT_WEIGHTS_PATH
        self.weights_: dict = {}
        self.features_: list = []
        self.initialized_: bool = False

        self._load_or_initialize()

    
    '''
        Carga el JSON de pesos si existe, sino se inicializa a partir de constants.py
    '''
    def _load_or_initialize(self) -> None:
        if self.weights_path_.exists():
            self._load_from_json()
        else:
            self._initialize_from_constants()

    
    '''
        Inicializa el JSON de pesos a partir de constants.py
    '''
    def _initialize_from_constants(self) -> None:
        todas = constants.TODAS_VARIABLES

        data = {
            "features": {
                feature: info.get("peso", DEFAULT_WEIGHT)
                for feature, info in todas.items()
            },
            "metadata": {
                "total_features": len(todas),
                "initialized_from": "constants",
                "default_weight": DEFAULT_WEIGHT,
                "initialization_time": datetime.now().isoformat(),
            },
        }

        self._save_json(data)
        self.features_ = list(todas.keys())
        self.weights_ = dict(data["features"])
        self.initialized_ = True
        print(f"JSON de pesos inicializado en {self.weights_path_} con {len(todas)} características.")

    '''    
        Carga el JSON de pesos
    ''' 
    def _load_from_json(self) -> None:
            try:
                with open(self.weights_path_, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                self.features_ = list(data["features"].keys())
                self.weights_ = dict(data["features"])
                self.initialized_ = True
                print(f"Pesos cargados desde {self.weights_path_}, total de características: {len(self.features_)}.")
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Error al cargar el JSON en '{self.weights_path_}': {e}") from e
            
    
    
    ''' 
        Operaciones sobre pesos guardados
        - Obtener peso de una característica
        - Obtener todos los pesos
        - Modificar peso de una característica
        - Modificar varios pesos a la vez
    '''
    def get_weight(self, feature: str) -> float:
        self._assert_initialized()
        if feature not in self.weights_:
            raise KeyError(f"Característica '{feature}' no encontrada en el archivo de pesos.")
        return self.weights_[feature]
    
    def get_weights(self) -> dict:
        self._assert_initialized()
        return dict(self.weights_) 
    
    def set_weight(self, feature: str, weight: float) -> None:
        self._assert_initialized()
        if feature not in self.weights_:
            raise KeyError(f"Característica '{feature}' no encontrada en el archivo de pesos.")
        self._validate_weight(feature, weight)

        self.weights_[feature] = weight
        self._persist_weight(feature, weight)

    def set_weights(self, weights: dict) -> None:
        self._assert_initialized()
        unknown_features = set(weights) - set(self.weights_)
        if unknown_features:
            raise KeyError(f"Características desconocidas: {unknown_features}")
        
        for feature, weight in weights.items():
            self._validate_weight(feature, weight)
            
        self.weights_.update(weights)
        self._persist_all_weights()

    
    ''' 
        Reinicio de pesos a los valores por defecto definidos en constants.py
    '''
    def reset_weights_values(self) -> None:
        self._assert_initialized()
        todas = constants.TODAS_VARIABLES

        self.weights_ = {
            f: todas[f].get("peso", DEFAULT_WEIGHT) if f in todas else DEFAULT_WEIGHT
            for f in self.features_
        }
        self._persist_all_weights()
        print("Pesos reiniciados a valores de constants.py.")

    ''' 
        Helper (para algun algoritmo): 
        - Obtener los pesos en formato array
        - Obtener los pesos normalizados (suma 1) en formato dict
    '''

    def get_weights_array(self, feature_order: Optional[list] = None) -> list:
        self._assert_initialized()
        ordenado = feature_order if feature_order is not None else self.features_

        return [self.weights_[f] for f in ordenado]
    
    def get_normalized_weights(self) -> dict:
        self._assert_initialized()
        total = sum(self.weights_.values())
        if total == 0:
            raise ValueError("La suma de pesos es 0, no se puede normalizar.")
        return {f: round(w / total, 6) for f, w in self.weights_.items()}
    
    '''
        Imprime un resumen de los pesos actuales.
    '''
    def print_summary(self) -> None:
        self._assert_initialized()
        print(f"\n{'='*50}")
        print(f"  WeightManager — {self.weights_path_}")
        print(f"  Features: {len(self.features_)}")
        print(f"{'='*50}")
        for f in self.features_:
            bar = "■" * int(self.weights_[f] * 100)
            print(f"  {f:<20} {self.weights_[f]:.2f}  {bar}")
        print(f"{'='*50}\n")

    '''
        Operaciones de persistencia sobre JSON:
        - Persistir un solo peso modificado
        - Persistir todos los pesos
    '''
    def _persist_weight(self, feature: str, weight: float) -> None:
        with open(self.weights_path_, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["features"][feature] = weight
        self._save_json(data)

    def _persist_all_weights(self) -> None:
        with open(self.weights_path_, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["features"].update(self.weights_)
        self._save_json(data)

    def _save_json(self, data: dict) -> None:
        """Escribe el dict al JSON, creando directorios si no existen."""
        self.weights_path_.parent.mkdir(parents=True, exist_ok=True)
        with open(self.weights_path_, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    
    ''' 
        Validaciones:
        - Validar que el peso esté en el rango [0.0, 1.0]
        - Validar que el WeightManager esté inicializado antes de operar
    '''
    def _validate_weight(self, feature: str, weight: float) -> None:
        if not (0.0 <= weight <= 1.0):
            raise ValueError(
                f"Peso de '{feature}' debe estar en [0.0, 1.0]. Recibido: {weight}"
            )
    def _assert_initialized(self) -> None:
        if not self.initialized_:
            raise RuntimeError("WeightManager no está inicializado correctamente.")