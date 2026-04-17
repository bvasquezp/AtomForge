"""
Heurística de Estabilidad MOF inspirada en Moosavi et al. 2020
(Nature Communications - "Understanding the diversity of the metal-organic framework ecosystem")

Evalúa 4 dimensiones:
1. Estabilidad Térmica      → basada en metal + coordinación
2. Estabilidad Química      → resistencia a agua/ácidos via ligando
3. Estabilidad Mecánica     → bulk modulus + densidad
4. Sintetizabilidad         → SAscore + diversidad elemental

Score final: 0-100 (>60 = candidato viable para síntesis)
"""

import os
import csv
import json
import warnings
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

# =============================================================================
# Tablas de Conocimiento Químico (Moosavi-inspired)
# =============================================================================

# Estabilidad térmica por metal (°C aproximado de descomposición MOF típico)
METAL_THERMAL_STABILITY = {
    "Zr": 95, "Al": 90, "Fe": 85, "Cr": 85, "Ti": 80,
    "Cu": 70, "Zn": 65, "Co": 70, "Ni": 72, "Mn": 60,
    "Cd": 40, "In": 50, "Mg": 55, "Ca": 45,
}

# Resistencia química del metal al agua (0-10)
METAL_WATER_STABILITY = {
    "Zr": 9, "Al": 8, "Fe": 7, "Cr": 8, "Ti": 8,
    "Cu": 5, "Zn": 4, "Co": 5, "Ni": 6, "Mn": 4,
    "Cd": 3, "In": 5, "Mg": 3, "Ca": 2,
}

# Grupos funcionales del ligando y su efecto en estabilidad
# (substructura SMARTS -> delta_stability)
LIGAND_STABILITY_GROUPS = {
    "c1ccc(cc1)C(=O)O": +2.0,   # BDC (tereftalato) - muy estable
    "c1cc2ccc(cc2cc1)C(=O)O": +2.5,  # NDC (naftalendicarboxilato)
    "C(=O)O": +1.0,              # Carboxilato genérico
    "n1ccnc1": +1.5,             # Imidazolato (ZIF)
    "C(=O)[NH]": -0.5,           # Amida - menos estable
    "O": -0.3,                   # Éter - moderado
    "S": -0.8,                   # Tioéter - menos estable
}

# Número de coordinación óptimo por metal
OPTIMAL_COORDINATION = {
    "Zr": 8, "Al": 6, "Fe": 6, "Cr": 6, "Ti": 6,
    "Cu": 4, "Zn": 4, "Co": 6, "Ni": 6, "Mn": 6,
    "Cd": 6, "In": 6, "Mg": 6, "Ca": 8,
}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class StabilityProfile:
    structure_id: str
    thermal_score: float        # 0-25
    chemical_score: float       # 0-25
    mechanical_score: float     # 0-25
    synthetic_score: float      # 0-25
    total_score: float          # 0-100
    grade: str                  # A/B/C/D
    metals_found: List[str]     # Metales detectados
    dominant_metal: str
    ligand_fragments: int
    best_sascore: float
    bulk_modulus_gpa: float
    density_gcm3: float
    recommendation: str
    flags: List[str] = field(default_factory=list)


# =============================================================================
# Motor de Evaluación
# =============================================================================

def detect_metals(cif_path: str) -> Tuple[List[str], Dict[str, int]]:
    """Detecta metales y su count en la estructura."""
    from pymatgen.core import Structure
    
    struct = Structure.from_file(cif_path)
    metals = list(METAL_THERMAL_STABILITY.keys())
    
    found = {}
    for site in struct:
        sym = str(site.specie.symbol)
        if sym in metals:
            found[sym] = found.get(sym, 0) + 1

    return list(found.keys()), found


def estimate_coordination_number(cif_path: str, metal: str) -> float:
    """Estima número de coordinación promedio del metal dominante."""
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import CrystalNN
    
    try:
        struct = Structure.from_file(cif_path)
        cnn = CrystalNN()
        cn_list = []
        
        for i, site in enumerate(struct):
            if str(site.specie.symbol) == metal:
                try:
                    cn = cnn.get_cn(struct, i)
                    cn_list.append(cn)
                except Exception:
                    continue
                if len(cn_list) >= 5:  # Muestra de 5 sitios es suficiente
                    break
                    
        return np.mean(cn_list) if cn_list else 0.0
    except Exception:
        return 0.0


def score_thermal_stability(metals: List[str], cif_path: str) -> Tuple[float, List[str]]:
    """
    Score 0-25 basado en:
    - Estabilidad térmica del metal
    - Número de coordinación vs óptimo
    """
    flags = []
    
    if not metals:
        return 5.0, ["Sin metales detectados"]
    
    # Metal dominante = más estable térmicamente
    dominant = max(metals, key=lambda m: METAL_THERMAL_STABILITY.get(m, 50))
    base_temp = METAL_THERMAL_STABILITY.get(dominant, 50)
    
    # Normalizar a 0-15 (300°C = 15 pts, 100°C = 5 pts)
    temp_score = min(15.0, max(0.0, (base_temp - 40) * 15 / 60))
    
    # Bonus por número de coordinación
    optimal_cn = OPTIMAL_COORDINATION.get(dominant, 6)
    actual_cn = estimate_coordination_number(cif_path, dominant)
    
    if actual_cn > 0:
        cn_diff = abs(actual_cn - optimal_cn)
        cn_score = max(0.0, 10.0 - cn_diff * 2.0)
    else:
        cn_score = 5.0
        flags.append("CN no calculable")
    
    # Penalización por múltiples metales (complejidad sintética)
    if len(metals) > 2:
        temp_score *= 0.85
        flags.append(f"MOF heterometálico ({len(metals)} metales) → síntesis compleja")

    total = min(25.0, temp_score + cn_score)
    return round(total, 2), flags


def score_chemical_stability(metals: List[str], sascore_csv: Optional[str] = None,
                              structure_id: str = "") -> Tuple[float, List[str]]:
    """
    Score 0-25 basado en:
    - Resistencia al agua del metal
    - Grupos funcionales del ligando (via SAscore CSV)
    """
    flags = []
    
    if not metals:
        return 5.0, ["Sin metales"]
    
    dominant = max(metals, key=lambda m: METAL_WATER_STABILITY.get(m, 5))
    water_score = METAL_WATER_STABILITY.get(dominant, 5)
    
    # Normalizar 0-10 → 0-20
    chem_score = water_score * 2.0
    
    # Bonus ligando via SAscore CSV
    ligand_bonus = 0.0
    if sascore_csv and Path(sascore_csv).exists():
        with open(sascore_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if structure_id in row.get("ID", ""):
                    sa = float(row.get("best_sascore", 10))
                    smiles = row.get("best_smiles", "")
                    
                    # Carboxilatos aromáticos = muy estables
                    if "C(=O)O" in smiles and "c1" in smiles:
                        ligand_bonus = 5.0
                        flags.append("Ligando carboxilato aromático → alta estabilidad")
                    elif "C(=O)O" in smiles:
                        ligand_bonus = 3.0
                    elif "n1ccnc1" in smiles:
                        ligand_bonus = 4.0
                        flags.append("Ligando imidazolato → ZIF-like, muy estable")
                    break

    # Penalización Zn/Cd en agua
    if "Zn" in metals or "Cd" in metals:
        flags.append("Zn/Cd sensibles a humedad → considerar síntesis en atmósfera inerte")
    
    total = min(25.0, chem_score + ligand_bonus)
    return round(total, 2), flags


def score_mechanical_stability(bulk_modulus_gpa: float, 
                                density_gcm3: float) -> Tuple[float, List[str]]:
    """
    Score 0-25 basado en:
    - Bulk modulus (GPa): MOFs típicos 1-30 GPa
    - Densidad: MOFs típicos 0.3-1.5 g/cm³
    """
    flags = []
    
    # Bulk modulus score (0-15)
    # Rango ideal para MOF funcional: 5-25 GPa
    if bulk_modulus_gpa <= 0:
        bm_score = 3.0
        flags.append("Bulk modulus no calculado")
    elif bulk_modulus_gpa < 1.0:
        bm_score = 2.0
        flags.append("Bulk modulus muy bajo → estructura frágil")
    elif bulk_modulus_gpa > 50:
        bm_score = 8.0
        flags.append("Bulk modulus anormalmente alto")
    else:
        bm_score = min(15.0, bulk_modulus_gpa * 0.6)
    
    # Densidad score (0-10)
    # Ultra-poroso: 0.2-0.5 g/cm³ (bueno para adsorción pero frágil)
    # Estable: 0.8-1.5 g/cm³
    if 0.5 <= density_gcm3 <= 1.5:
        den_score = 10.0
    elif 0.2 <= density_gcm3 < 0.5:
        den_score = 7.0
        flags.append("Ultra-poroso → alta capacidad pero posible fragilidad")
    elif density_gcm3 > 1.5:
        den_score = 6.0
        flags.append("Alta densidad → posible baja porosidad accesible")
    else:
        den_score = 3.0
        flags.append("Densidad anormal")
    
    total = min(25.0, bm_score + den_score)
    return round(total, 2), flags


def score_synthetic_accessibility(sascore: float, 
                                   n_fragments: int,
                                   n_metals: int) -> Tuple[float, List[str]]:
    """
    Score 0-25 basado en:
    - SAscore del ligando principal
    - Complejidad (número de fragmentos y metales)
    """
    flags = []
    
    # SAscore → 0-20
    if sascore <= 0 or sascore > 9.9:
        sa_score = 5.0
        flags.append("SAscore no disponible")
    else:
        # SA=1 → 20pts, SA=6 → 0pts (lineal)
        sa_score = max(0.0, 20.0 * (6.0 - sascore) / 5.0)
    
    if sascore > 6.0:
        flags.append(f"SAscore={sascore:.1f} → ligando difícil de sintetizar")
    elif sascore < 3.0:
        flags.append(f"SAscore={sascore:.1f} → ligando comercialmente disponible")
    
    # Penalización por complejidad
    complexity_penalty = 0.0
    if n_fragments > 5:
        complexity_penalty += 2.0
        flags.append(f"{n_fragments} fragmentos ligando → purificación compleja")
    if n_metals > 1:
        complexity_penalty += 1.5 * (n_metals - 1)
    
    # Bonus simplicidad (1 metal, 1-2 ligandos = MOF clásico)
    simplicity_bonus = 5.0 if (n_metals == 1 and 1 <= n_fragments <= 3) else 0.0
    
    total = min(25.0, max(0.0, sa_score + simplicity_bonus - complexity_penalty))
    return round(total, 2), flags


def assign_grade(score: float) -> Tuple[str, str]:
    """Asigna grado y recomendación."""
    if score >= 80:
        return "A", "PRIORIDAD ALTA: Candidato excelente para síntesis inmediata"
    elif score >= 65:
        return "B", "VIABLE: Proceder con síntesis, monitorear estabilidad química"
    elif score >= 50:
        return "C", "MARGINAL: Requiere optimización de ligando o condiciones de síntesis"
    else:
        return "D", "DESCARTAR: Inviable técnica o económicamente"


# =============================================================================
# Pipeline Principal
# =============================================================================

def evaluate_candidate(
    cif_path: str,
    sascore_csv: Optional[str] = None,
    bulk_modulus_gpa: float = 0.0,
    density_gcm3: float = 0.0,
) -> StabilityProfile:
    
    name = Path(cif_path).stem
    all_flags = []

    # 1. Detectar metales
    try:
        metals, metal_counts = detect_metals(cif_path)
        dominant = max(metal_counts, key=metal_counts.get) if metal_counts else "Unknown"
    except Exception as e:
        metals, metal_counts, dominant = [], {}, "Unknown"
        all_flags.append(f"Error detectando metales: {e}")

    # 2. Leer SAscore del CSV si existe
    sascore = 10.0
    n_fragments = 0
    if sascore_csv and Path(sascore_csv).exists():
        with open(sascore_csv) as f:
            for row in csv.DictReader(f):
                if name in row.get("ID", ""):
                    sascore = float(row.get("best_sascore", 10))
                    n_fragments = int(row.get("n_fragments", 0))
                    break

    # 3. Calcular scores
    t_score, t_flags = score_thermal_stability(metals, cif_path)
    c_score, c_flags = score_chemical_stability(metals, sascore_csv, name)
    m_score, m_flags = score_mechanical_stability(bulk_modulus_gpa, density_gcm3)
    s_score, s_flags = score_synthetic_accessibility(sascore, n_fragments, len(metals))

    all_flags = t_flags + c_flags + m_flags + s_flags
    total = t_score + c_score + m_score + s_score
    grade, recommendation = assign_grade(total)

    return StabilityProfile(
        structure_id=name,
        thermal_score=t_score,
        chemical_score=c_score,
        mechanical_score=m_score,
        synthetic_score=s_score,
        total_score=round(total, 2),
        grade=grade,
        metals_found=metals,
        dominant_metal=dominant,
        ligand_fragments=n_fragments,
        best_sascore=sascore,
        bulk_modulus_gpa=bulk_modulus_gpa,
        density_gcm3=density_gcm3,
        recommendation=recommendation,
        flags=all_flags,
    )


def run_stability_pipeline(
    candidates_dir: str,
    sascore_csv: Optional[str] = None,
    porosity_csv: Optional[str] = None,
    output_csv: str = "results/stability_report.csv",
    output_json: str = "results/stability_report.json",
):
    cifs = sorted(Path(candidates_dir).glob("*_relaxed.cif"))
    if not cifs:
        cifs = sorted(Path(candidates_dir).glob("*.cif"))

    print(f"Evaluando estabilidad de {len(cifs)} candidatos...\n")

    # Cargar densidades desde porosity CSV si existe
    density_map = {}
    bm_map = {}
    if porosity_csv and Path(porosity_csv).exists():
        with open(porosity_csv) as f:
            for row in csv.DictReader(f):
                rid = row.get("ID", "")
                density_map[rid] = float(row.get("Density", 0) or 0)

    profiles = []
    for cif in cifs:
        name = cif.stem
        print(f"  [{name}]", end=" ", flush=True)
        
        density = density_map.get(name, 0.0)
        bm = bm_map.get(name, 0.0)

        profile = evaluate_candidate(
            str(cif),
            sascore_csv=sascore_csv,
            bulk_modulus_gpa=bm,
            density_gcm3=density,
        )
        profiles.append(profile)
        print(f"Score={profile.total_score:.1f} ({profile.grade}) | "
              f"Metal={profile.dominant_metal} | SA={profile.best_sascore:.2f}")

    # Ordenar por score
    profiles.sort(key=lambda p: p.total_score, reverse=True)

    # Guardar CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        fieldnames = [
            "ID", "grade", "total_score", "thermal_score", "chemical_score",
            "mechanical_score", "synthetic_score", "dominant_metal",
            "best_sascore", "bulk_modulus_gpa", "density_gcm3",
            "ligand_fragments", "recommendation", "flags"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in profiles:
            writer.writerow({
                "ID": p.structure_id,
                "grade": p.grade,
                "total_score": p.total_score,
                "thermal_score": p.thermal_score,
                "chemical_score": p.chemical_score,
                "mechanical_score": p.mechanical_score,
                "synthetic_score": p.synthetic_score,
                "dominant_metal": p.dominant_metal,
                "best_sascore": p.best_sascore,
                "bulk_modulus_gpa": p.bulk_modulus_gpa,
                "density_gcm3": p.density_gcm3,
                "ligand_fragments": p.ligand_fragments,
                "recommendation": p.recommendation,
                "flags": " | ".join(p.flags),
            })

    # Guardar JSON detallado
    with open(output_json, "w") as f:
        json.dump([p.__dict__ for p in profiles], f, indent=2)

    # Resumen consola
    print(f"\n{'='*60}")
    print(f"RANKING DE ESTABILIDAD")
    print(f"{'='*60}")
    print(f"{'ID':<35} {'Score':>6} {'Grade':>5} {'Metal':>5} {'SA':>5}")
    print("-" * 60)
    for p in profiles:
        print(f"{p.structure_id:<35} {p.total_score:>6.1f} {p.grade:>5} "
              f"{p.dominant_metal:>5} {p.best_sascore:>5.2f}")
    
    grade_a = [p for p in profiles if p.grade == "A"]
    grade_b = [p for p in profiles if p.grade == "B"]
    print(f"\nGrado A (síntesis inmediata): {len(grade_a)}")
    print(f"Grado B (viables):            {len(grade_b)}")
    print(f"\nReportes: {output_csv}")
    print(f"          {output_json}")

    return profiles


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_dir", default="results/candidates")
    parser.add_argument("--sascore_csv", default="results/sascore_report.csv")
    parser.add_argument("--porosity_csv", default="results/summary_report.csv")
    parser.add_argument("--output_csv", default="results/stability_report.csv")
    parser.add_argument("--output_json", default="results/stability_report.json")
    args = parser.parse_args()

    run_stability_pipeline(
        candidates_dir=args.candidates_dir,
        sascore_csv=args.sascore_csv,
        porosity_csv=args.porosity_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
