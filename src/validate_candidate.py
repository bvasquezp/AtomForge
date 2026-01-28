import os
import numpy as np
from scipy.spatial import cKDTree

# Radios de Van der Waals aproximados (en Angstroms)
VDW_RADII = {
    'Zr': 2.18,
    'Cu': 1.96,
    'C': 1.70,
    'O': 1.52,
    'H': 1.20,
    'N': 1.55,
    'default': 1.70
}

MASSES = {
    'Zr': 91.22, 'Cu': 63.55, 'C': 12.01, 'O': 15.99, 'H': 1.01
}

def parse_xyz(filepath):
    """Lee el archivo XYZ y extrae posiciones y elementos."""
    coords = []
    elements = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Ignorar las primeras 2 lineas (numero de atomos y comentario)
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                elements.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(elements), np.array(coords)

def calculate_porosity_monte_carlo(elements, coords, num_samples=50000):
    """
    Estima la porosidad lanzando puntos aleatorios dentro de la caja envolvente (Bounding Box).
    """
    # 1. Definir la caja envolvente (Bounding Box) del cristal
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Agregamos un 'padding' para no cortar los atomos del borde
    padding = 2.0
    box_min = min_coords - padding
    box_max = max_coords + padding
    volume_box = np.prod(box_max - box_min)
    
    # 2. Generar puntos de sonda aleatorios
    probe_points = np.random.uniform(low=box_min, high=box_max, size=(num_samples, 3))
    
    # 3. Asignar radios a cada atomo del cristal
    radii = np.array([VDW_RADII.get(el, VDW_RADII['default']) for el in elements])
    
    # 4. Usar KDTree para busqueda rapida de vecinos
    tree = cKDTree(coords)
    
    # Consultar el arbol: para cada punto sonda, encontrar la distancia al atomo mas cercano
    dists, indices = tree.query(probe_points, k=1)
    
    # 5. Determinar colisiones
    # Un punto esta 'ocupado' si su distancia a un atomo es menor que el radio de ese atomo
    occupied_mask = dists < radii[indices]
    occupied_count = np.sum(occupied_mask)
    
    void_fraction = 1.0 - (occupied_count / num_samples)
    
    return void_fraction, volume_box

def calculate_density(elements, volume_angstrom3):
    """Calcula densidad aproximada en g/cm3."""
    total_mass_amu = sum([MASSES.get(el, 12.0) for el in elements])
    
    # Conversion: 1 amu/A^3 = 1.66054 g/cm^3
    density = (total_mass_amu / volume_angstrom3) * 1.66054
    return density

def validate_structure(filename):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(project_root, 'top_candidates', filename)
    
    if not os.path.exists(filepath):
        print(f"Error: Archivo no encontrado: {filepath}")
        return

    print(f"--- VALIDANDO: {filename} ---")
    
    try:
        elements, coords = parse_xyz(filepath)
        num_atoms = len(elements)
        
        # Calcular metricas
        void_fraction, box_vol = calculate_porosity_monte_carlo(elements, coords)
        density = calculate_density(elements, box_vol)
        
        print(f"Atomos:              {num_atoms}")
        print(f"Volumen de caja:     {box_vol:.2f} A^3")
        print(f"Fraccion de Vacio:   {void_fraction*100:.2f}%")
        print(f"Densidad Estimada:   {density:.4f} g/cm3")
        
        # Evaluacion
        print("\nDIAGNOSTICO:")
        if void_fraction < 0.10:
            print("[CRITICO] Estructura demasiado densa. El gas no entrara.")
        elif void_fraction > 0.80:
            print("[ADVERTENCIA] Demasiado vacio. Probablemente inestable o disociada.")
        else:
            print("[EXITO] Porosidad dentro del rango operativo para MOFs (10-80%).")
            
    except Exception as e:
        print(f"Error en validacion: {e}")

if __name__ == "__main__":
    # Cambia esto por el nombre de tu archivo optimizado
    target_file = "crystal_gen_007_hollow_stitched.xyz" 
    validate_structure(target_file)