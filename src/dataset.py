"""
Deep-Material v2: Dataset Pipeline Unificado
=============================================
Convierte archivos CIF a grafos periodicos con:
- Coordenadas fraccionales (periodicidad natural)
- Parametros de celda unitaria (a, b, c, alpha, beta, gamma)
- Tipos atomicos por numero atomico
- Enlaces via CrystalNN (quimicamente informados)
- Edge attributes: distancias de enlace
- Train/val/test split reproducible
"""

import os
import sys
import json
import warnings
import torch
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List, Dict

warnings.filterwarnings("ignore")

# Agregar src al path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_config, resolve_paths, get_project_root, set_seed


def cif_to_periodic_graph(cif_path: str, config: dict) -> Optional[dict]:
    """
    Convierte un archivo CIF a un grafo periodico completo.

    Retorna un diccionario con:
        - x: (N,) tensor de numeros atomicos (long)
        - frac_coords: (N, 3) coordenadas fraccionales (float)
        - cart_coords: (N, 3) coordenadas cartesianas (float)
        - edge_index: (2, E) indices de enlaces (long)
        - edge_attr: (E, 1) distancias de enlace (float)
        - lattice: (3, 3) matriz de red (float)
        - lattice_params: (6,) parametros [a, b, c, alpha, beta, gamma] (float)
        - num_atoms: int
        - formula: str
    """
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import CrystalNN

    try:
        struct = Structure.from_file(cif_path)
    except Exception:
        return None

    num_atoms = len(struct)
    max_atoms = config.get("data", {}).get("max_atoms", 500)
    min_atoms = config.get("data", {}).get("min_atoms", 5)

    if num_atoms > max_atoms or num_atoms < min_atoms:
        return None

    try:
        # --- Tipos atomicos ---
        max_z = config.get("model", {}).get("num_atom_types", 100)
        atomic_numbers = []
        for site in struct:
            z = site.specie.Z
            if z > max_z:
                return None  # Elemento demasiado pesado, saltamos
            atomic_numbers.append(z)

        x = torch.tensor(atomic_numbers, dtype=torch.long)

        # --- Coordenadas ---
        frac_coords = torch.tensor(
            struct.frac_coords, dtype=torch.float32
        )
        cart_coords = torch.tensor(
            struct.cart_coords, dtype=torch.float32
        )

        # --- Celda unitaria ---
        lattice_matrix = torch.tensor(
            struct.lattice.matrix, dtype=torch.float32
        )
        lattice_params = torch.tensor([
            struct.lattice.a, struct.lattice.b, struct.lattice.c,
            struct.lattice.alpha, struct.lattice.beta, struct.lattice.gamma
        ], dtype=torch.float32)

        # --- Enlaces via CrystalNN ---s
        from pymatgen.analysis.local_env import VoronoiNN
        cnn = VoronoiNN(
            cutoff=8.0,
            weight="solid_angle",
            extra_nn_info=False
        )
        edge_indices = []
        edge_attrs = []

        for i in range(num_atoms):
            try:
                neighbors = cnn.get_nn_info(struct, i)
                for n in neighbors:
                    j = n["site_index"]
                    dist = struct[i].distance(struct[j])
                    edge_indices.append([i, j])
                    edge_attrs.append([dist])
            except Exception:
                # CrystalNN puede fallar en sitios problematicos
                continue

        if len(edge_indices) == 0:
            return None

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

        # --- Formula ---
        formula = struct.composition.reduced_formula

        return {
            "x": x,
            "frac_coords": frac_coords,
            "cart_coords": cart_coords,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "lattice": lattice_matrix,
            "lattice_params": lattice_params,
            "num_atoms": num_atoms,
            "formula": formula,
        }

    except Exception:
        return None


def wrapper_process_cif(filename, raw_dir, processed_dir, config):
    """Wrapper para procesar un solo CIF en paralelo."""
    cif_path = os.path.join(raw_dir, filename)
    graph = cif_to_periodic_graph(cif_path, config)
    
    if graph is not None:
        save_name = filename.replace(".cif", ".pt")
        save_path = os.path.join(processed_dir, save_name)
        torch.save(graph, save_path)
        return graph["formula"]
    return None


def process_dataset(config: dict) -> None:
    """Procesa todos los CIFs y genera grafos periodicos."""
    paths = config["paths"]
    raw_dir = paths["raw_cifs"]
    processed_dir = paths["processed"]

    os.makedirs(processed_dir, exist_ok=True)

    cif_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".cif")])

    if not cif_files:
        print(f"No se encontraron archivos .cif en {raw_dir}")
        return

    num_workers = paths.get("num_workers", mp.cpu_count() - 2)
    print(f"🚀 Iniciando procesamiento paralelo con {num_workers} núcleos...")

    # Usar partial para pasar argumentos fijos a la funcion de mapeo
    process_func = partial(
        wrapper_process_cif, 
        raw_dir=raw_dir, 
        processed_dir=processed_dir, 
        config=config
    )

    formulas = []
    success = 0
    skipped = 0

    with mp.Pool(processes=num_workers) as pool:
        # Usamos imap para ver el progreso con tqdm
        results = list(tqdm(pool.imap(process_func, cif_files), total=len(cif_files), desc="📊 Procesando CIFs"))

    # Consolidar resultados
    for res in results:
        if res is not None:
            formulas.append(res)
            success += 1
        else:
            skipped += 1

    print(f"\nCompletado: {success} grafos guardados en '{processed_dir}'")

    # Guardar metadata del dataset
    meta = {
        "total_cifs": len(cif_files),
        "processed": success,
        "skipped": skipped,
        "unique_formulas": len(set(formulas)),
    }
    meta_path = os.path.join(processed_dir, "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata guardada en {meta_path}")


def create_splits(config: dict) -> None:
    """
    Crea splits train/val/test y guarda las listas de archivos.
    Garantiza reproducibilidad via semilla fija.
    """
    set_seed(config.get("project", {}).get("seed", 42))

    processed_dir = config["paths"]["processed"]
    pt_files = sorted([f for f in os.listdir(processed_dir)
                       if f.endswith(".pt") and not f.startswith("_")])

    if not pt_files:
        print("No hay archivos procesados para dividir.")
        return

    n = len(pt_files)
    indices = np.random.permutation(n)

    split_cfg = config.get("data", {}).get("split", {})
    train_ratio = split_cfg.get("train", 0.80)
    val_ratio = split_cfg.get("val", 0.10)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = [pt_files[i] for i in indices[:n_train]]
    val_files = [pt_files[i] for i in indices[n_train:n_train + n_val]]
    test_files = [pt_files[i] for i in indices[n_train + n_val:]]

    splits = {
        "train": sorted(train_files),
        "val": sorted(val_files),
        "test": sorted(test_files),
    }

    splits_path = os.path.join(processed_dir, "_splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Splits creados: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    print(f"Guardado en {splits_path}")


# =============================================================================
# Dataset de PyTorch
# =============================================================================

class MOFDataset(torch.utils.data.Dataset):
    """
    Dataset de MOFs como grafos periodicos.
    Soporta splits via _splits.json.
    """

    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = root

        splits_path = os.path.join(root, "_splits.json")
        if os.path.exists(splits_path):
            with open(splits_path, "r") as f:
                splits = json.load(f)
            self.files = splits.get(split, [])
        else:
            # Sin splits: usar todos los archivos
            self.files = sorted([f for f in os.listdir(root)
                                 if f.endswith(".pt") and not f.startswith("_")])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        filepath = os.path.join(self.root, self.files[idx])
        data = torch.load(filepath, weights_only=False)
        return data

    def get_stats(self) -> dict:
        """Calcula estadisticas del dataset."""
        num_atoms_list = []
        num_edges_list = []
        elements_seen = set()

        for f in self.files:
            filepath = os.path.join(self.root, f)
            data = torch.load(filepath, weights_only=False)
            num_atoms_list.append(data["num_atoms"])
            num_edges_list.append(data["edge_index"].shape[1])
            elements_seen.update(data["x"].tolist())

        return {
            "num_samples": len(self.files),
            "avg_atoms": np.mean(num_atoms_list),
            "max_atoms": max(num_atoms_list),
            "min_atoms": min(num_atoms_list),
            "avg_edges": np.mean(num_edges_list),
            "unique_elements": len(elements_seen),
        }


# =============================================================================
# Collate para batching de grafos de tamano variable
# =============================================================================

def collate_periodic_graphs(batch: List[dict]) -> dict:
    """
    Collate function para batching de grafos periodicos.
    Combina grafos de tamano variable en un batch.
    """
    x_list = []
    frac_coords_list = []
    cart_coords_list = []
    edge_index_list = []
    edge_attr_list = []
    lattice_list = []
    lattice_params_list = []
    num_atoms_list = []
    batch_idx_list = []

    offset = 0
    for i, data in enumerate(batch):
        n = data["num_atoms"]

        x_list.append(data["x"])
        frac_coords_list.append(data["frac_coords"])
        cart_coords_list.append(data["cart_coords"])

        # Offset edge indices para batch
        edge_index = data["edge_index"].clone()
        edge_index += offset
        edge_index_list.append(edge_index)

        edge_attr_list.append(data["edge_attr"])
        lattice_list.append(data["lattice"])
        lattice_params_list.append(data["lattice_params"])
        num_atoms_list.append(n)
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))

        offset += n

    return {
        "x": torch.cat(x_list, dim=0),
        "frac_coords": torch.cat(frac_coords_list, dim=0),
        "cart_coords": torch.cat(cart_coords_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "lattice": torch.stack(lattice_list, dim=0),
        "lattice_params": torch.stack(lattice_params_list, dim=0),
        "num_atoms": num_atoms_list,
        "batch": torch.cat(batch_idx_list, dim=0),
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline de datos Deep-Material v2")
    parser.add_argument("--action", choices=["process", "split", "stats", "all"],
                        default="all", help="Accion a realizar")
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_paths(config)

    if args.action in ("process", "all"):
        process_dataset(config)

    if args.action in ("split", "all"):
        create_splits(config)

    if args.action in ("stats", "all"):
        for split_name in ["train", "val", "test"]:
            ds = MOFDataset(config["paths"]["processed"], split=split_name)
            if len(ds) > 0:
                stats = ds.get_stats()
                print(f"\n{split_name.upper()}: {stats}")
