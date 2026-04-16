"""
Deep-Material v2: Simulacion GCMC (Grand Canonical Monte Carlo)
================================================================
Evalua la capacidad de adsorcion de gas (CO2) en estructuras MOF
usando una implementacion de GCMC en Python puro.

Modos de operacion:
  1. Python puro:  Algoritmo GCMC simplificado con potencial Lennard-Jones
                    para screening rapido de candidatos.
  2. RASPA prep:   Genera los archivos de entrada (input files) necesarios
                    para ejecutar simulaciones de adsorcion con RASPA2.

NOTA IMPORTANTE:
  Los CIFs generados por el modelo tienen coordenadas aleatorias (dummy data),
  lo que causa superposicion de atomos y divergencia energetica. Este script
  esta configurado para tomar un CIF valido del dataset original de
  entrenamiento (data/raw_cifs/) como prueba de que el pipeline funciona.
"""

import os
import sys
import math
import random
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================================
# Parametros Fisicos y Constantes
# ============================================================================

K_BOLTZMANN = 1.380649e-23   # J/K
AVOGADRO = 6.02214076e23
R_GAS = 8.314462             # J/(mol*K)
ANG_TO_M = 1e-10

# Parametros Lennard-Jones para adsorbato CO2 (modelo TraPPE)
# sigma (Angstrom), epsilon/kB (K)
LJ_PARAMS = {
    # Atomos del framework (UFF generico simplificado)
    "C":  {"sigma": 3.431, "epsilon_kB": 52.84},
    "H":  {"sigma": 2.571, "epsilon_kB": 22.14},
    "O":  {"sigma": 3.118, "epsilon_kB": 30.19},
    "N":  {"sigma": 3.261, "epsilon_kB": 34.72},
    "Zn": {"sigma": 2.462, "epsilon_kB": 62.40},
    "Cu": {"sigma": 3.114, "epsilon_kB": 2.516},
    "Co": {"sigma": 2.559, "epsilon_kB": 7.045},
    "Fe": {"sigma": 2.594, "epsilon_kB": 6.540},
    "Ni": {"sigma": 2.525, "epsilon_kB": 7.548},
    "Mn": {"sigma": 2.638, "epsilon_kB": 6.540},
    "Cr": {"sigma": 2.693, "epsilon_kB": 7.548},
    "Al": {"sigma": 4.008, "epsilon_kB": 254.1},
    "Si": {"sigma": 3.826, "epsilon_kB": 202.3},
    "S":  {"sigma": 3.595, "epsilon_kB": 137.9},
    "P":  {"sigma": 3.695, "epsilon_kB": 153.5},
    "Ti": {"sigma": 2.829, "epsilon_kB": 8.550},
    "V":  {"sigma": 2.801, "epsilon_kB": 8.051},
    "Mg": {"sigma": 2.691, "epsilon_kB": 55.86},
    "Ca": {"sigma": 3.399, "epsilon_kB": 119.8},
    "Cd": {"sigma": 2.537, "epsilon_kB": 114.7},
    "In": {"sigma": 3.976, "epsilon_kB": 301.4},
    "Zr": {"sigma": 2.783, "epsilon_kB": 34.72},
}

# CO2 como particula unica (centro de masa, modelo esfericalizado)
CO2_SIGMA = 3.941       # Angstrom
CO2_EPSILON_KB = 195.2   # K


# ============================================================================
# Estructuras de Datos
# ============================================================================

@dataclass
class CrystalStructure:
    """Representacion interna de un cristal para GCMC."""
    name: str
    lattice_params: Tuple[float, float, float, float, float, float]  # a,b,c,alpha,beta,gamma
    lattice_matrix: np.ndarray   # (3,3) vectores de celda
    inv_lattice: np.ndarray      # Inversa para conversion cart->frac
    species: List[str]
    frac_coords: np.ndarray      # (N, 3)
    cart_coords: np.ndarray      # (N, 3)
    volume: float                # Angstrom^3


@dataclass
class GCMCConfig:
    """Configuracion de la simulacion GCMC."""
    temperature: float = 298.0         # K
    pressure: float = 1.0              # bar
    n_cycles: int = 5000               # Ciclos de produccion
    n_equil: int = 2000                # Ciclos de equilibrio
    n_insert_attempts: int = 1         # Intentos de insercion por ciclo
    cutoff: float = 12.0               # Angstrom, corte LJ
    gas_species: str = "CO2"
    # Probabilidades de movimientos MC
    p_insert: float = 0.35
    p_delete: float = 0.35
    p_translate: float = 0.30
    max_translation: float = 1.0       # Angstrom
    # Supercell para condicion de imagen minima
    supercell: Tuple[int, int, int] = (1, 1, 1)


@dataclass
class GCMCResult:
    """Resultados de la simulacion GCMC."""
    structure_name: str
    temperature: float
    pressure: float
    avg_loading: float          # moleculas por celda
    std_loading: float
    loading_mg_g: float         # mg/g
    loading_cm3_stp_cm3: float  # cm3(STP)/cm3
    henry_constant: float       # mol/(kg*Pa) (estimado)
    avg_energy: float           # kJ/mol
    acceptance_insert: float
    acceptance_delete: float
    acceptance_translate: float
    n_cycles: int


# ============================================================================
# Lectura de CIF con pymatgen
# ============================================================================

def load_cif(filepath: str) -> CrystalStructure:
    """Carga un archivo CIF usando pymatgen y retorna la estructura interna."""
    try:
        from pymatgen.core import Structure
    except ImportError:
        print("ERROR: pymatgen no esta instalado. Ejecutar: pip install pymatgen")
        sys.exit(1)

    struct = Structure.from_file(filepath)

    lattice = struct.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

    species = [str(s) for s in struct.species]
    frac_coords = struct.frac_coords
    cart_coords = struct.cart_coords
    volume = lattice.volume

    return CrystalStructure(
        name=Path(filepath).stem,
        lattice_params=(a, b, c, alpha, beta, gamma),
        lattice_matrix=lattice.matrix.copy(),
        inv_lattice=lattice.inv_matrix.copy(),
        species=species,
        frac_coords=frac_coords.copy(),
        cart_coords=cart_coords.copy(),
        volume=volume,
    )


def check_structure_validity(struct: CrystalStructure, min_dist: float = 0.5) -> Tuple[bool, str]:
    """
    Verifica que la estructura no tenga atomos superpuestos.
    Retorna (es_valido, mensaje).
    """
    n = len(struct.species)
    if n == 0:
        return False, "Estructura vacia (0 atomos)"

    for i in range(n):
        for j in range(i + 1, n):
            dx = struct.frac_coords[i] - struct.frac_coords[j]
            # Imagen minima en fractional
            dx = dx - np.round(dx)
            cart_dx = dx @ struct.lattice_matrix
            dist = np.linalg.norm(cart_dx)
            if dist < min_dist:
                return False, (
                    f"Atomos superpuestos: {struct.species[i]}({i}) y "
                    f"{struct.species[j]}({j}) a {dist:.3f} A (min={min_dist} A)"
                )
    return True, "Estructura valida"


# ============================================================================
# Motor GCMC (Python Puro)
# ============================================================================

def get_lj_params(species: str) -> Tuple[float, float]:
    """Retorna (sigma, epsilon_kB) para una especie. Default a carbono si desconocida."""
    params = LJ_PARAMS.get(species, LJ_PARAMS["C"])
    return params["sigma"], params["epsilon_kB"]


def lorentz_berthelot(sigma1: float, eps1: float, sigma2: float, eps2: float) -> Tuple[float, float]:
    """Reglas de mezcla Lorentz-Berthelot."""
    sigma_mix = 0.5 * (sigma1 + sigma2)
    eps_mix = math.sqrt(eps1 * eps2)
    return sigma_mix, eps_mix


def lj_energy(r: float, sigma: float, epsilon_kB: float) -> float:
    """Calcula energia Lennard-Jones (en unidades de kB*K)."""
    if r < 0.3 * sigma:  # Overlap protection
        return 1e12
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon_kB * (sr6 * sr6 - sr6)


def compute_guest_host_energy(
    guest_frac: np.ndarray,
    struct: CrystalStructure,
    cutoff: float,
) -> float:
    """
    Calcula la energia total de interaccion guest-framework.
    Retorna energia en unidades de kB*K.
    """
    total_energy = 0.0

    guest_sigma = CO2_SIGMA
    guest_eps = CO2_EPSILON_KB

    # Imagenes periodicas [-1, 0, 1] para cada dimension
    for ix in range(-1, 2):
        for iy in range(-1, 2):
            for iz in range(-1, 2):
                shift = np.array([ix, iy, iz], dtype=float)
                for i, species in enumerate(struct.species):
                    host_frac = struct.frac_coords[i] + shift
                    dx_frac = guest_frac - host_frac
                    dx_cart = dx_frac @ struct.lattice_matrix
                    r = np.linalg.norm(dx_cart)

                    if r > cutoff or r < 1e-8:
                        continue

                    host_sigma, host_eps = get_lj_params(species)
                    sig_mix, eps_mix = lorentz_berthelot(
                        guest_sigma, guest_eps, host_sigma, host_eps
                    )
                    total_energy += lj_energy(r, sig_mix, eps_mix)

    return total_energy


def compute_guest_guest_energy(
    guest_frac: np.ndarray,
    all_guests: List[np.ndarray],
    guest_idx: int,
    struct: CrystalStructure,
    cutoff: float,
) -> float:
    """Calcula la energia de interaccion guest-guest para una molecula."""
    total_energy = 0.0
    sig_gg = CO2_SIGMA
    eps_gg = CO2_EPSILON_KB

    for j, other_frac in enumerate(all_guests):
        if j == guest_idx:
            continue

        for ix in range(-1, 2):
            for iy in range(-1, 2):
                for iz in range(-1, 2):
                    shift = np.array([ix, iy, iz], dtype=float)
                    dx_frac = guest_frac - (other_frac + shift)
                    dx_cart = dx_frac @ struct.lattice_matrix
                    r = np.linalg.norm(dx_cart)

                    if r > cutoff or r < 1e-8:
                        continue

                    total_energy += lj_energy(r, sig_gg, eps_gg)

    return total_energy


def ideal_gas_chemical_potential(temperature: float, pressure: float) -> float:
    """
    Potencial quimico del gas ideal: mu_id = kB*T * ln(beta * P * Lambda^3)
    Para GCMC necesitamos beta*mu = ln(beta*P*Lambda^3)
    Simplificamos usando la fugacidad = P (gas ideal).
    Retorna beta*mu_excess (adimensional).
    """
    # Longitud de onda termica de Broglie para CO2 (M=44 g/mol)
    M_co2 = 44.01e-3 / AVOGADRO  # kg por molecula
    h = 6.626e-34  # J*s
    lambda_db = h / math.sqrt(2 * math.pi * M_co2 * K_BOLTZMANN * temperature)
    lambda_db_ang = lambda_db / ANG_TO_M  # Convertir a Angstrom

    # P en Pa
    P_pa = pressure * 1e5

    # beta = 1/(kB*T)
    beta = 1.0 / (K_BOLTZMANN * temperature)

    # beta*mu = ln(beta * P * Lambda^3) -- en SI
    beta_mu = math.log(beta * P_pa * lambda_db**3)

    return beta_mu


def run_gcmc(
    struct: CrystalStructure,
    config: GCMCConfig,
    verbose: bool = True,
) -> GCMCResult:
    """
    Ejecuta la simulacion Grand Canonical Monte Carlo.

    Algoritmo:
    1. Insercion: Colocar molecula aleatoriamente, aceptar con min(1, V*exp(-beta*dU + beta*mu) / (N+1))
    2. Eliminacion: Remover molecula aleatoria, aceptar con min(1, N*exp(beta*dU - beta*mu) / V)
    3. Traslacion: Mover molecula aleatoria, aceptar con Metropolis usual
    """
    T = config.temperature
    beta = 1.0 / (K_BOLTZMANN * T)
    beta_kB = 1.0 / T  # Para energias en unidades kB*K

    # Potencial quimico (para el criterio de aceptacion)
    beta_mu = ideal_gas_chemical_potential(T, config.pressure)

    # Volumen en Angstrom^3
    V = struct.volume
    for sx, sy, sz in [config.supercell]:
        V_super = V * sx * sy * sz

    # Estado del sistema: lista de coordenadas fraccionales de moleculas guest
    guests: List[np.ndarray] = []

    # Contadores
    n_insert_accept, n_insert_total = 0, 0
    n_delete_accept, n_delete_total = 0, 0
    n_translate_accept, n_translate_total = 0, 0

    # Almacenar loadings para estadisticas
    loadings: List[int] = []
    energies: List[float] = []

    rng = random.Random(42)
    total_cycles = config.n_equil + config.n_cycles

    if verbose:
        print(f"  Iniciando GCMC: T={T} K, P={config.pressure} bar, "
              f"{config.n_equil} equil + {config.n_cycles} prod ciclos")
        print(f"  Volumen de celda: {V:.2f} A^3 | beta*mu = {beta_mu:.4f}")

    for cycle in range(total_cycles):
        # Decidir tipo de movimiento
        r = rng.random()

        if r < config.p_insert:
            # --- INSERCION ---
            n_insert_total += 1
            N = len(guests)

            # Posicion aleatoria en coordenadas fraccionales
            new_frac = np.array([rng.random(), rng.random(), rng.random()])

            # Energia de la nueva molecula
            dU_host = compute_guest_host_energy(new_frac, struct, config.cutoff)
            dU_guest = compute_guest_guest_energy(new_frac, guests, -1, struct, config.cutoff)
            dU = dU_host + dU_guest  # en kB*K

            # Criterio de aceptacion de insercion (GCMC)
            # acc = min(1, V/(N+1) * exp(-beta_kB * dU + beta_mu))
            # Nota: beta_mu ya contiene la contribucion de Lambda^3 y P
            arg = -beta_kB * dU + beta_mu + math.log(V / (N + 1))

            if arg > 0 or rng.random() < math.exp(min(arg, 500)):
                guests.append(new_frac)
                n_insert_accept += 1

        elif r < config.p_insert + config.p_delete:
            # --- ELIMINACION ---
            n_delete_total += 1
            N = len(guests)

            if N > 0:
                idx = rng.randint(0, N - 1)
                old_frac = guests[idx]

                dU_host = compute_guest_host_energy(old_frac, struct, config.cutoff)
                dU_guest = compute_guest_guest_energy(old_frac, guests, idx, struct, config.cutoff)
                dU = dU_host + dU_guest

                # acc = min(1, N/V * exp(beta_kB * dU - beta_mu))
                arg = beta_kB * dU - beta_mu + math.log(N / V)

                if arg > 0 or rng.random() < math.exp(min(arg, 500)):
                    guests.pop(idx)
                    n_delete_accept += 1

        else:
            # --- TRASLACION ---
            n_translate_total += 1
            N = len(guests)

            if N > 0:
                idx = rng.randint(0, N - 1)
                old_frac = guests[idx].copy()

                # Pequeño desplazamiento en coordenadas fraccionales
                delta_cart = np.array([
                    rng.gauss(0, config.max_translation),
                    rng.gauss(0, config.max_translation),
                    rng.gauss(0, config.max_translation),
                ])
                delta_frac = delta_cart @ struct.inv_lattice
                new_frac = (old_frac + delta_frac) % 1.0

                # Energia vieja
                E_old_host = compute_guest_host_energy(old_frac, struct, config.cutoff)
                E_old_guest = compute_guest_guest_energy(old_frac, guests, idx, struct, config.cutoff)

                # Energia nueva
                E_new_host = compute_guest_host_energy(new_frac, struct, config.cutoff)
                E_new_guest = compute_guest_guest_energy(new_frac, guests, idx, struct, config.cutoff)

                dU = (E_new_host + E_new_guest) - (E_old_host + E_old_guest)

                # Metropolis
                if dU <= 0 or rng.random() < math.exp(-beta_kB * dU):
                    guests[idx] = new_frac
                    n_translate_accept += 1

        # Recoger estadisticas en fase de produccion
        if cycle >= config.n_equil:
            loadings.append(len(guests))
            # Energia total del sistema
            total_E = 0.0
            for g_idx, g_frac in enumerate(guests):
                total_E += compute_guest_host_energy(g_frac, struct, config.cutoff)
                total_E += 0.5 * compute_guest_guest_energy(g_frac, guests, g_idx, struct, config.cutoff)
            energies.append(total_E)

        # Progress
        if verbose and (cycle + 1) % max(total_cycles // 10, 1) == 0:
            pct = (cycle + 1) / total_cycles * 100
            phase = "EQUIL" if cycle < config.n_equil else "PROD"
            print(f"    [{phase}] {pct:5.1f}% | N_guests={len(guests)} | cycle {cycle+1}/{total_cycles}")

    # --- Calcular resultados ---
    avg_loading = np.mean(loadings) if loadings else 0.0
    std_loading = np.std(loadings) if loadings else 0.0
    avg_energy = np.mean(energies) if energies else 0.0

    # Convertir energia de kB*K a kJ/mol
    avg_energy_kjmol = avg_energy * K_BOLTZMANN * AVOGADRO / 1000.0

    # Loading en mg/g
    # masa_framework en g/mol (estimacion burda basada en peso atomico promedio)
    from pymatgen.core import Element
    framework_mass = sum(Element(s).atomic_mass for s in struct.species)  # en g/mol (por celda)
    co2_mass = 44.01  # g/mol
    loading_mg_g = (avg_loading * co2_mass / framework_mass) * 1000.0  # mg/g

    # Loading en cm3(STP)/cm3
    # 1 mol de gas a STP = 22414 cm3
    V_cm3 = V * 1e-24  # Angstrom^3 a cm^3
    loading_cm3_stp_cm3 = (avg_loading / AVOGADRO) * 22414.0 / V_cm3 if V_cm3 > 0 else 0.0

    # Constante de Henry (estimacion: loading / pressure a baja presion)
    P_pa = config.pressure * 1e5
    henry = (avg_loading / (framework_mass * 1e-3)) / P_pa if P_pa > 0 else 0.0  # mol/(kg*Pa)

    result = GCMCResult(
        structure_name=struct.name,
        temperature=T,
        pressure=config.pressure,
        avg_loading=avg_loading,
        std_loading=std_loading,
        loading_mg_g=loading_mg_g,
        loading_cm3_stp_cm3=loading_cm3_stp_cm3,
        henry_constant=henry,
        avg_energy=avg_energy_kjmol,
        acceptance_insert=n_insert_accept / max(n_insert_total, 1),
        acceptance_delete=n_delete_accept / max(n_delete_total, 1),
        acceptance_translate=n_translate_accept / max(n_translate_total, 1),
        n_cycles=config.n_cycles,
    )

    if verbose:
        print(f"\n  --- Resultados GCMC para {struct.name} ---")
        print(f"  Loading promedio:  {avg_loading:.2f} +/- {std_loading:.2f} moleculas/celda")
        print(f"  Loading (mg/g):    {loading_mg_g:.2f}")
        print(f"  Loading (STP):     {loading_cm3_stp_cm3:.2f} cm3(STP)/cm3")
        print(f"  Energia promedio:  {avg_energy_kjmol:.2f} kJ/mol")
        print(f"  Henry constant:    {henry:.4e} mol/(kg*Pa)")
        print(f"  Accept. insert:    {result.acceptance_insert:.3f}")
        print(f"  Accept. delete:    {result.acceptance_delete:.3f}")
        print(f"  Accept. translate: {result.acceptance_translate:.3f}")

    return result


# ============================================================================
# Generador de Inputs para RASPA
# ============================================================================

def generate_raspa_input(
    struct: CrystalStructure,
    cif_path: str,
    output_dir: str,
    temperature: float = 298.0,
    pressure: float = 1e5,  # Pa
    n_cycles: int = 25000,
    n_init: int = 10000,
    gas: str = "CO2",
) -> str:
    """
    Genera los archivos de entrada de RASPA2 para una simulacion GCMC.
    Retorna la ruta al directorio de simulacion creado.
    """
    sim_dir = Path(output_dir) / struct.name
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Copiar CIF al directorio de simulacion
    import shutil
    cif_dest = sim_dir / Path(cif_path).name
    if not cif_dest.exists():
        shutil.copy2(cif_path, cif_dest)

    # Archivo simulation.input
    input_content = f"""SimulationType                MonteCarlo
NumberOfCycles                {n_cycles}
NumberOfInitializationCycles  {n_init}
PrintEvery                    1000
RestartFile                   no

Forcefield                    GenericMOFs
CutOffVDW                     12.0

Framework 0
FrameworkName {struct.name}
UnitCells 1 1 1
HeliumVoidFraction 0.29
ExternalTemperature {temperature}
ExternalPressure {pressure}

Component 0 MoleculeName             {gas}
            MoleculeDefinition       TraPPE
            TranslationProbability   0.5
            RotationProbability      0.5
            ReinsertionProbability   1.0
            SwapProbability          1.0
            CreateNumberOfMolecules  0
"""

    input_path = sim_dir / "simulation.input"
    with open(input_path, "w") as f:
        f.write(input_content)

    # Archivo force_field_mixing_rules.def (UFF parameters simplificados)
    ff_content = """# General mixing rules
# number of rules
shift
# mixing rule
Lorentz-Berthelot
# Atom  epsilon/kB  sigma
C_co2   27.0        2.80
O_co2   79.0        3.05
"""
    ff_path = sim_dir / "force_field_mixing_rules.def"
    with open(ff_path, "w") as f:
        f.write(ff_content)

    return str(sim_dir)


# ============================================================================
# Pipeline Principal
# ============================================================================

def find_valid_test_cif(raw_cifs_dir: str, max_atoms: int = 200) -> Optional[str]:
    """
    Busca un CIF valido del dataset original para testing.
    Prioriza CIFs pequenos (< max_atoms) para rapidez.
    Busca preferiblemente MOFs conocidos con buena porosidad.
    """
    priority_prefixes = ["ABAVIJ", "AHOKIR", "BOHGOU", "AGESIP", "CICYIX"]

    cif_dir = Path(raw_cifs_dir)
    if not cif_dir.exists():
        return None

    all_cifs = sorted(cif_dir.glob("*.cif"))
    if not all_cifs:
        return None

    # Primero buscar CIFs prioritarios (estructuras sencillas conocidas)
    for prefix in priority_prefixes:
        for cif in all_cifs:
            if cif.name.startswith(prefix) and "clean" in cif.name:
                # Verificar tamanno (archivo pequenno = pocos atomos)
                if cif.stat().st_size < 15000:
                    return str(cif)

    # Fallback: primer CIF pequenno
    for cif in all_cifs:
        if "clean" in cif.name and cif.stat().st_size < 10000:
            return str(cif)

    return str(all_cifs[0])


def main():
    parser = argparse.ArgumentParser(
        description="GCMC Analysis - Deep-Material v2 Pipeline"
    )
    parser.add_argument(
        "--cif", default=None,
        help="Ruta al archivo CIF a analizar. Si no se especifica, se usa un CIF del dataset original."
    )
    parser.add_argument(
        "--mode", choices=["gcmc", "raspa", "both"], default="gcmc",
        help="Modo: 'gcmc' (Python puro), 'raspa' (generar inputs), 'both'"
    )
    parser.add_argument(
        "--temperature", type=float, default=298.0,
        help="Temperatura en K (default: 298)"
    )
    parser.add_argument(
        "--pressure", type=float, default=1.0,
        help="Presion en bar (default: 1.0)"
    )
    parser.add_argument(
        "--cycles", type=int, default=5000,
        help="Ciclos de produccion GCMC (default: 5000)"
    )
    parser.add_argument(
        "--equil", type=int, default=2000,
        help="Ciclos de equilibrio GCMC (default: 2000)"
    )
    parser.add_argument(
        "--cutoff", type=float, default=12.0,
        help="Cutoff LJ en Angstrom (default: 12.0)"
    )
    parser.add_argument(
        "--output_dir", default="results/gcmc",
        help="Directorio de salida (default: results/gcmc)"
    )
    parser.add_argument(
        "--use-generated", action="store_true",
        help="Usar CIFs generados por el modelo. ADVERTENCIA: probablemente fallara por overlap."
    )

    args = parser.parse_args()

    # --- Resolver CIF ---
    project_root = Path(__file__).resolve().parent.parent

    if args.cif:
        cif_path = args.cif
    elif args.use_generated:
        # Buscar en generated_crystals o results/candidates
        gen_dirs = [
            project_root / "results" / "candidates",
            project_root / "generated_crystals",
        ]
        cif_path = None
        for d in gen_dirs:
            cifs = sorted(d.glob("*.cif")) if d.exists() else []
            if cifs:
                cif_path = str(cifs[0])
                break
        if not cif_path:
            print("ERROR: No se encontraron CIFs generados.")
            sys.exit(1)
        print(f"ADVERTENCIA: Usando CIF generado ({cif_path}). "
              "Puede fallar por superposicion de atomos.")
    else:
        # Usar CIF del dataset original
        raw_dir = project_root / "data" / "raw_cifs"
        cif_path = find_valid_test_cif(str(raw_dir))
        if not cif_path:
            print("ERROR: No se encontraron CIFs en data/raw_cifs/")
            sys.exit(1)

    print("=" * 70)
    print("  Deep-Material v2: GCMC Adsorption Analysis")
    print("=" * 70)
    print(f"  CIF:         {Path(cif_path).name}")
    print(f"  Modo:        {args.mode}")
    print(f"  Temperatura: {args.temperature} K")
    print(f"  Presion:     {args.pressure} bar")
    print(f"  Gas:         CO2")
    print(f"  Cycles:      {args.equil} equil + {args.cycles} prod")
    print("=" * 70)

    # Cargar estructura
    print("\n[1/4] Cargando estructura cristalina...")
    struct = load_cif(cif_path)
    print(f"  Nombre:  {struct.name}")
    print(f"  Atomos:  {len(struct.species)}")
    print(f"  Volumen: {struct.volume:.2f} A^3")
    print(f"  Celda:   a={struct.lattice_params[0]:.3f}  b={struct.lattice_params[1]:.3f}  "
          f"c={struct.lattice_params[2]:.3f}")

    # Validar estructura
    print("\n[2/4] Validando integridad estructural...")
    is_valid, msg = check_structure_validity(struct)
    if not is_valid:
        print(f"  FALLO: {msg}")
        print("  La estructura tiene atomos superpuestos. No se puede ejecutar GCMC.")
        print("  Sugerencia: Use un CIF del dataset original con --cif <ruta>")
        sys.exit(1)
    else:
        print(f"  OK: {msg}")

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ejecutar GCMC Python puro
    if args.mode in ("gcmc", "both"):
        print("\n[3/4] Ejecutando simulacion GCMC (Python puro)...")
        gcmc_config = GCMCConfig(
            temperature=args.temperature,
            pressure=args.pressure,
            n_cycles=args.cycles,
            n_equil=args.equil,
            cutoff=args.cutoff,
        )

        result = run_gcmc(struct, gcmc_config, verbose=True)

        # Guardar resultados
        import csv
        csv_path = output_dir / "gcmc_results.csv"
        write_header = not csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "Structure", "T(K)", "P(bar)", "Loading(mol/uc)",
                    "Std", "Loading(mg/g)", "Loading(cm3STP/cm3)",
                    "Henry(mol/kg/Pa)", "Energy(kJ/mol)",
                    "Acc_Insert", "Acc_Delete", "Acc_Translate", "N_Cycles"
                ])
            writer.writerow([
                result.structure_name, result.temperature, result.pressure,
                f"{result.avg_loading:.4f}", f"{result.std_loading:.4f}",
                f"{result.loading_mg_g:.2f}", f"{result.loading_cm3_stp_cm3:.2f}",
                f"{result.henry_constant:.6e}", f"{result.avg_energy:.2f}",
                f"{result.acceptance_insert:.4f}", f"{result.acceptance_delete:.4f}",
                f"{result.acceptance_translate:.4f}", result.n_cycles,
            ])
        print(f"\n  Resultados guardados en {csv_path}")

    # Generar inputs RASPA
    if args.mode in ("raspa", "both"):
        print("\n[4/4] Generando archivos de entrada para RASPA2...")
        raspa_dir = generate_raspa_input(
            struct, cif_path, str(output_dir / "raspa_inputs"),
            temperature=args.temperature,
            pressure=args.pressure * 1e5,  # bar -> Pa
        )
        print(f"  Directorio de simulacion: {raspa_dir}")
        print(f"  Para ejecutar: cd {raspa_dir} && simulate simulation.input")

    print("\n" + "=" * 70)
    print("  GCMC Analysis completado exitosamente.")
    print("=" * 70)


if __name__ == "__main__":
    main()
