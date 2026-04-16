"""
Deep-Material v2: Validacion DFT con GFN2-xTB
==============================================
Lee TOP 10 de gcmc_results.csv, relaja con GFN2-xTB
y calcula energia de adsorcion de H2.
Paralelizado con 20 hilos via ProcessPoolExecutor.
"""

import os
import sys
import csv
import warnings
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")


def get_top10(csv_path: str) -> list:
    """Lee gcmc_results.csv y retorna top 10 por loading mg/g."""
    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    results.sort(key=lambda x: float(x["Loading(mg/g)"]), reverse=True)
    return results[:10]


def relax_xtb(cif_path: str, output_path: str, method: str = "GFN2-xTB") -> dict:
    """
    Relaja una estructura con GFN2-xTB via ASE+tblite.
    Retorna dict con energia y path del CIF relajado.
    """
    from ase.io import read, write
    from ase.optimize import BFGS
    from tblite.ase import TBLite

    atoms = read(cif_path)
    atoms.calc = TBLite(method=method, verbosity=0)

    opt = BFGS(atoms, logfile=None)
    try:
        opt.run(fmax=0.1, steps=500)
        converged = True
    except Exception:
        converged = False

    energy = atoms.get_potential_energy()
    write(output_path, atoms)

    return {
        "cif_relaxed": output_path,
        "energy_ev": energy,
        "converged": converged,
        "n_atoms": len(atoms),
    }


def calc_adsorption_energy_h2(mof_cif: str, method: str = "GFN2-xTB") -> float:
    """
    Calcula E_ads(H2) = E(MOF+H2) - E(MOF) - E(H2).
    Coloca H2 en el centroide de la celda unitaria.
    Retorna energia en kJ/mol.
    """
    from ase.io import read
    from ase import Atoms
    from tblite.ase import TBLite

    # E(MOF)
    mof = read(mof_cif)
    mof.calc = TBLite(method=method, verbosity=0)
    e_mof = mof.get_potential_energy()

    # E(H2) libre
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    h2.calc = TBLite(method=method, verbosity=0)
    e_h2 = h2.get_potential_energy()

    # E(MOF+H2): insertar H2 en el centro de la celda
    cell = mof.get_cell()
    center = cell.sum(axis=0) / 2.0
    h2_positions = np.array([
        center + np.array([0, 0, 0.37]),
        center - np.array([0, 0, 0.37]),
    ])

    from ase import Atoms as ASEAtoms
    import ase
    mof_h2 = mof.copy()
    h2_guest = ase.Atoms(
        "H2",
        positions=h2_positions,
        cell=mof.get_cell(),
        pbc=mof.get_pbc()
    )
    mof_h2 = mof + h2_guest
    mof_h2.calc = TBLite(method=method, verbosity=0)
    e_mof_h2 = mof_h2.get_potential_energy()

    # Conversion eV -> kJ/mol (1 eV = 96.485 kJ/mol)
    e_ads = (e_mof_h2 - e_mof - e_h2) * 96.485
    return e_ads


def process_candidate(args: tuple) -> dict:
    """Worker function para paralelizacion."""
    name, cif_path, output_dir = args
    relaxed_path = str(Path(output_dir) / f"{name}_xtb_relaxed.cif")

    print(f"  [{name}] Iniciando relajacion GFN2-xTB...")
    try:
        relax_result = relax_xtb(cif_path, relaxed_path)
        print(f"  [{name}] Relajacion OK | E={relax_result['energy_ev']:.4f} eV | converged={relax_result['converged']}")
    except Exception as e:
        print(f"  [{name}] Error en relajacion: {e}")
        return {"name": name, "error": str(e)}

    print(f"  [{name}] Calculando E_ads(H2)...")
    try:
        e_ads = calc_adsorption_energy_h2(relaxed_path)
        print(f"  [{name}] E_ads(H2) = {e_ads:.2f} kJ/mol")
    except Exception as e:
        print(f"  [{name}] Error en E_ads: {e}")
        e_ads = None

    return {
        "name": name,
        "cif_relaxed": relaxed_path,
        "energy_ev": relax_result["energy_ev"],
        "energy_ev_per_atom": relax_result["energy_ev"] / relax_result["n_atoms"],
        "converged": relax_result["converged"],
        "e_ads_h2_kjmol": e_ads,
        "n_atoms": relax_result["n_atoms"],
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "results" / "gcmc" / "gcmc_results.csv"
    candidates_dir = project_root / "results" / "candidates"
    output_dir = project_root / "results" / "xtb_validated"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: No se encontro {csv_path}")
        return

    print("=" * 60)
    print("  Deep-Material v2: Validacion GFN2-xTB")
    print("=" * 60)

    top10 = get_top10(str(csv_path))
    print(f"TOP 10 candidatos por loading CO2:")
    for i, r in enumerate(top10):
        print(f"  {i+1}. {r['Structure']} | {float(r['Loading(mg/g)']):.2f} mg/g")

    # Preparar argumentos para workers
    tasks = []
    for r in top10:
        name = r["Structure"]
        # Buscar CIF relajado por LJ primero, si no el original
        relaxed_lj = candidates_dir / f"{name}_relaxed.cif"
        original = candidates_dir / f"{name}.cif"
        cif_path = str(relaxed_lj) if relaxed_lj.exists() else str(original)
        tasks.append((name, cif_path, str(output_dir)))

    print(f"\nCoriendo {len(tasks)} candidatos en paralelo (20 workers)...")

    results = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_candidate, t): t[0] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Guardar reporte final
    report_path = project_root / "results" / "xtb_validation_report.csv"
    fieldnames = ["name", "energy_ev", "energy_ev_per_atom", "converged", "e_ads_h2_kjmol", "n_atoms", "error"]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.get("e_ads_h2_kjmol") or 0):
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nReporte guardado en {report_path}")
    print("\nRanking final por E_ads(H2):")
    for r in sorted(results, key=lambda x: x.get("e_ads_h2_kjmol") or 0):
        if "error" not in r:
            print(f"  {r['name']} | E_ads={r.get('e_ads_h2_kjmol', 'N/A'):.2f} kJ/mol | converged={r['converged']}")


if __name__ == "__main__":
    main()
