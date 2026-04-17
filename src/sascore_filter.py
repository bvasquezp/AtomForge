"""
Extrae ligandos orgánicos de CIFs generados y calcula SAscore.
Filtra candidatos sintéticamente accesibles (SAscore < 6).
"""
import os
import sys
import csv
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import RWMol, AllChem, Descriptors
from rdkit.Chem.Draw import MolToImage

# SA Score de Ertl & Schuffenhauer (implementacion RDKit contrib)
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Elementos metalicos (nodos MOF, los excluimos)
METALS = {
    "Zn","Cu","Fe","Co","Ni","Mn","Cr","V","Ti","Zr","Mo","Cd","In",
    "Al","Mg","Ca","Sr","Ba","La","Ce","Eu","Tb","Gd","Y","Sc"
}

def extract_organic_fragments(cif_path: str) -> list:
    """
    Lee CIF con pymatgen y extrae fragmentos orgánicos
    separando nodos metálicos de ligandos.
    Retorna lista de SMILES de fragmentos.
    """
    from pymatgen.core import Structure
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import CrystalNN
    
    try:
        struct = Structure.from_file(cif_path)
    except Exception as e:
        print(f"  Error leyendo CIF: {e}")
        return []

    # Filtrar solo átomos orgánicos (C, H, N, O, S, P, F, Cl, Br)
    organic_elements = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br"}
    organic_indices = [
        i for i, site in enumerate(struct)
        if str(site.specie.symbol) in organic_elements
    ]

    if not organic_indices:
        return []

    # Construir mol RDKit desde coordenadas cartesianas
    mol = RWMol()
    idx_map = {}
    
    for new_idx, old_idx in enumerate(organic_indices):
        site = struct[old_idx]
        atom = Chem.Atom(str(site.specie.symbol))
        mol.AddAtom(atom)
        idx_map[old_idx] = new_idx

    # Agregar enlaces basados en distancias VdW
    from pymatgen.core import Element
    positions = struct.cart_coords
    
    for i, oi in enumerate(organic_indices):
        for j, oj in enumerate(organic_indices):
            if j <= i:
                continue
            dist = struct.get_distance(oi, oj)
            ri = Element(str(struct[oi].specie.symbol)).atomic_radius or 0.7
            rj = Element(str(struct[oj].specie.symbol)).atomic_radius or 0.7
            # Umbral de enlace covalente
            if dist < (float(ri) + float(rj)) * 1.3:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    # Sanitizar y obtener fragmentos
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    
    # Filtrar fragmentos muy pequeños (< 6 átomos pesados)
    smiles_list = []
    for frag in frags:
        if frag.GetNumHeavyAtoms() >= 6:
            smi = Chem.MolToSmiles(frag)
            if smi:
                smiles_list.append(smi)

    return list(set(smiles_list))  # Deduplicar


def calculate_sascore(smiles: str) -> float:
    """Calcula SAscore (1=fácil, 10=imposible). <6 = sintetizable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return 10.0


def calculate_mw(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol else 0.0


def filter_candidates(candidates_dir: str, output_csv: str, sa_threshold: float = 6.0):
    cifs = sorted(Path(candidates_dir).glob("*_relaxed.cif"))
    
    if not cifs:
        # Intentar sin _relaxed
        cifs = sorted(Path(candidates_dir).glob("*.cif"))

    print(f"Analizando {len(cifs)} candidatos...")
    print(f"Umbral SAscore: < {sa_threshold} (sintetizable)\n")

    results = []

    for cif in cifs:
        name = cif.stem
        print(f"  [{name}]", end=" ")
        
        fragments = extract_organic_fragments(str(cif))
        
        if not fragments:
            print("Sin fragmentos orgánicos detectados.")
            results.append({
                "ID": name, "n_fragments": 0,
                "best_smiles": "N/A", "best_sascore": 10.0,
                "avg_sascore": 10.0, "mw": 0.0, "viable": False
            })
            continue

        scores = [(smi, calculate_sascore(smi), calculate_mw(smi)) 
                  for smi in fragments]
        scores.sort(key=lambda x: x[1])  # Mejor (menor) primero

        best_smi, best_sa, best_mw = scores[0]
        avg_sa = sum(s[1] for s in scores) / len(scores)
        viable = best_sa < sa_threshold

        print(f"frags={len(fragments)} | best_SA={best_sa:.2f} | "
              f"avg_SA={avg_sa:.2f} | viable={'✓' if viable else '✗'}")

        results.append({
            "ID": name,
            "n_fragments": len(fragments),
            "best_smiles": best_smi,
            "best_sascore": round(best_sa, 3),
            "avg_sascore": round(avg_sa, 3),
            "mw": round(best_mw, 2),
            "viable": viable
        })

    if not results:
        print("No se encontraron resultados para guardar.")
        return []

    # Guardar CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda x: x["best_sascore"]))

    # Resumen
    viable = [r for r in results if r["viable"]]
    print(f"\n{'='*50}")
    print(f"Viables (SA < {sa_threshold}): {len(viable)}/{len(results)}")
    for r in viable:
        print(f"  {r['ID']} | SA={r['best_sascore']} | MW={r['mw']} | {r['best_smiles'][:50]}")
    print(f"\nReporte: {output_csv}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_dir", default="results/candidates")
    parser.add_argument("--output", default="results/sascore_report.csv")
    parser.add_argument("--threshold", type=float, default=6.0)
    args = parser.parse_args()

    filter_candidates(args.candidates_dir, args.output, args.threshold)
