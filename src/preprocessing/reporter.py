"""
Módulo para generación de reportes del preprocesamiento
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path


class PreprocessingReporter:
    """Genera reportes consolidados del preprocesamiento"""
    
    def __init__(self):
        self.sections = {}
    
    def _convert_to_serializable(self, obj):
        """
        Convierte objetos no serializables a JSON a tipos compatibles
        
        Args:
            obj: Objeto a convertir
            
        Return:
            Objeto serializable
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
        
    def add_section(self, section_name: str, content: dict):
        """
        Agrega una sección al reporte
        
        Args:
            section_name: Nombre de la sección
            content: Contenido de la sección
        """
        self.sections[section_name] = self._convert_to_serializable(content)
        
    def generate_full_report(self, output_path: str = None) -> dict:
        """
        Genera reporte completo
        
        Args:
            output_path: Ruta donde guardar JSON (opcional)
            
        Return:
            dict: Reporte completo
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'sections': self.sections
        }
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
        return report
    
    def get_summary(self) -> dict:
        """Resumen ejecutivo"""
        return {
            'loading': self.sections.get('loading', {}),
            'cleaning': self.sections.get('cleaning', {}),
            'transformation': self.sections.get('transformation', {})
        }