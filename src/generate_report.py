import os
from datetime import datetime

def generate_datasheet():
    # Datos manuales basados en tus resultados exitosos
    material_name = "Genesis-007"
    formula = "Zr-Cluster / Cu-Linker / C-Bridge"
    surface_area = 3075.69
    h2_capacity = 6.15
    density = 0.8182
    void_fraction = 76.18
    
    report = f"""
    =======================================================
    HOJA DE DATOS DE MATERIAL AVANZADO: {material_name}
    =======================================================
    FECHA: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    AUTOR: Benjamin (via Deep-Material AI)
    
    ------------------ PROPIEDADES FISICAS ----------------
    DENSIDAD:          {density} g/cm3
       (Ultraligero - Flota en agua)
       
    POROSIDAD (Vacio): {void_fraction}%
       (Alta capacidad de volumen libre)

    ------------------ RENDIMIENTO ENERGETICO -------------
    CAPACIDAD H2:      {h2_capacity} wt%
       STATUS: SUPERIOR AL ESTANDAR DOE 2025 (5.5 wt%)
       
    AREA SUPERFICIAL:  {surface_area} m2/g
       (Equivalente a medio campo de futbol en 1 gramo)

    ------------------ CONCLUSIONES -----------------------
    El material candidato 'Genesis-007' demuestra una topologia
    estable tras el proceso de zurcido (stitching). Su alta
    area superficial lo posiciona en el top 10% de materiales
    sinteticos para almacenamiento de gases limpios.
    
    RECOMENDACION: Proceder a sintesis experimental o
    simulacion DFT avanzada.
    =======================================================
    """
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'Genesis_007_Datasheet.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\nReporte guardado en: {output_path}")

if __name__ == "__main__":
    generate_datasheet()
