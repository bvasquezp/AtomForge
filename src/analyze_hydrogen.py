import os
import numpy as np
from scipy.spatial import cKDTree

# Parametros Fisicos
PROBE_RADIUS_N2 = 1.82  # Radio cinetico del Nitrogeno (para medir superficie)
VDW_RADII = {
    'Zr': 2.18, 'Cu': 1.96, 'C': 1.70, 'O': 1.52, 'H': 1.20, 'default': 1.70
}
MASSES = {
    'Zr': 91.22, 'Cu': 63.55, 'C': 12.01, 'O': 15.99, 'H': 1.01
}

def get_radius(atom_type):
    return VDW_RADII.get(atom_type, VDW_RADII['default'])

def calculate_surface_area_and_uptake(filename, num_samples=100000):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(project_root, 'top_candidates', filename)
    
    if not os.path.exists(filepath):
        print(f"Error: No se encuentra {filepath}")
        return

    # 1. Leer archivo
    atoms = []
    coords = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    coords = np.array(coords)
    radii = np.array([get_radius(a) for a in atoms])
    
    # 2. Calcular Masa Total (para normalizar por gramo)
    total_mass_amu = sum([MASSES.get(a, 12.0) for a in atoms])
    total_mass_g = total_mass_amu * 1.66054e-24 # Convertir AMU a gramos
    
    print(f"--- ANALISIS DE CAPACIDAD: {filename} ---")
    print(f"Masa de la celda unitaria: {total_mass_amu:.2f} amu")

    # 3. Metodo de Monte Carlo para Area Superficial
    # Generamos puntos en la superficie de cada atomo y vemos si son accesibles
    accessible_points = 0
    total_points_tried = 0
    
    # Usamos KDTree para busqueda rapida
    tree = cKDTree(coords)
    
    print("Calculando superficie accesible (esto puede tardar unos segundos)...")
    
    surface_area_total_A2 = 0.0
    
    for i, center in enumerate(coords):
        # Radio expandido donde "tocaria" la sonda de Nitrogeno
        expanded_radius = radii[i] + PROBE_RADIUS_N2
        
        # Generar puntos aleatorios en la esfera de superficie
        # Algoritmo de Fibonacci para distribucion uniforme en esfera
        n_points_atom = 200 # Puntos por atomo
        
        indices = np.arange(0, n_points_atom, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/n_points_atom)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = expanded_radius * np.cos(theta) * np.sin(phi)
        y = expanded_radius * np.sin(theta) * np.sin(phi)
        z = expanded_radius * np.cos(phi)
        
        sphere_points = np.column_stack((x, y, z)) + center
        
        # Verificar colisiones para estos puntos
        # Un punto es accesible si no choca con NINGUN otro atomo vecino
        dists, _ = tree.query(sphere_points, k=2) # k=2 porque el mas cercano es el atomo propio
        
        # dists[:, 1] es la distancia al vecino mas cercano (que no soy yo mismo)
        # Si esa distancia es mayor que el radio del vecino + radio sonda, es libre
        # (Simplificacion aproximada)
        
        # Enfoque mas robusto: Chequear contra todos los vecinos cercanos
        # Para hacerlo rapido, asumimos fraccion de area expuesta
        # Un punto es accesible si dist > vecino_radius + probe
        
        neighbors = tree.query_ball_point(sphere_points, r=4.0) # Buscar vecinos en 4A
        
        atom_accessible_count = 0
        for p_idx, point in enumerate(sphere_points):
            is_blocked = False
            # Revisar vecinos cercanos
            local_neighbors = neighbors[p_idx]
            for n_idx in local_neighbors:
                if n_idx == i: continue # Ignorar atomo propio
                
                dist = np.linalg.norm(point - coords[n_idx])
                if dist < (radii[n_idx] + PROBE_RADIUS_N2):
                    is_blocked = True
                    break
            
            if not is_blocked:
                atom_accessible_count += 1
        
        fraction_accessible = atom_accessible_count / n_points_atom
        atom_surface_area = 4 * np.pi * (expanded_radius**2) * fraction_accessible
        surface_area_total_A2 += atom_surface_area

    # 4. Resultados Finales
    # Convertir A^2 a m^2 (1 A^2 = 1e-20 m^2)
    sa_m2 = surface_area_total_A2 * 1e-20
    
    # Area Superficial Especifica (m^2/g)
    ssa = sa_m2 / total_mass_g
    
    # Prediccion de H2 (Regla de Chahine: 1 wt% por cada 500 m2/g)
    predicted_h2_wt = ssa / 500.0
    
    print("\nRESULTADOS:")
    print(f"Area Superficial (Est.): {ssa:.2f} m2/g")
    print(f"Capacidad H2 Predicha:   {predicted_h2_wt:.2f} wt%")
    
    print("\nEVALUACION DOE (Target 2025: 5.5 wt%):")
    if predicted_h2_wt > 5.5:
        print("🌟 SUPERIOR AL ESTANDAR (Material Revolucionario)")
    elif predicted_h2_wt > 4.0:
        print("✅ EXCELENTE CANDIDATO (Alta competitividad)")
    elif predicted_h2_wt > 2.0:
        print("⚠️ PROMEDIO (Bueno, pero mejorable)")
    else:
        print("❌ BAJO RENDIMIENTO (Demasiado pesado o poca superficie)")

if __name__ == "__main__":
    # Cambia esto por tu archivo final
    target_file = "crystal_gen_007_hollow_stitched.xyz"
    calculate_surface_area_and_uptake(target_file)