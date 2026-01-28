# Genesis-007: Material Avanzado para Almacenamiento de Hidrógeno

##  Resumen del Proyecto
Este repositorio contiene el pipeline de descubrimiento y validación computacional para **Genesis-007**, un marco metal-orgánico (MOF) diseñado mediante inteligencia artificial para superar los objetivos del Departamento de Energía de EE.UU. (DOE) para el almacenamiento de hidrógeno vehicular.

##  Especificaciones Técnicas (Validado)

| Propiedad | Valor | Estándar DOE 2025 | Estado |
| :--- | :--- | :--- | :--- |
| **Capacidad Gravimétrica** | **6.15 wt%** | 5.5 wt% |  Superado |
| **Área Superficial** | **3075 m²/g** | N/A | Excelente |
| **Densidad Cristalina** | 0.81 g/cm³ | > 0.70 g/cm³ | Óptimo |
| **Fracción de Vacío** | 76.18% | - | Alta Porosidad |

##  Composición Química
* **Nodos Metálicos:** Clústeres de Zirconio ($Zr_6O_8$) para estabilidad térmica y química.
* **Linkers:** Conectores basados en Cobre (Cu) con puentes orgánicos.
* **Topología:** Estructura jerárquica con "zurcido" (stitching) computacional para garantizar integridad mecánica.

##  Estructura del Repositorio
* `src/`: Código fuente para generación, reparación y análisis.
* `top_candidates/`: Archivos cristalográficos (.xyz) y reportes de simulación.
* `Genesis_007_Datasheet.txt`: Reporte técnico detallado generado automáticamente.

##  Reproducción
Para reproducir el análisis de capacidad:

```bash
python src/analyze_hydrogen.py
