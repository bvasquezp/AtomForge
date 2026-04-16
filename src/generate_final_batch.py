"""
Deep-Material v2: Script de Generación de Candidatos (Fase Final)
================================================================
Genera el batch final de 20 estructuras para caracterización de porosidad.
"""
import os
import sys
import torch
from pathlib import Path

# Asegurar que el directorio src esté en el path
src_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(src_dir))

from generate_flow import generate_crystals, export_to_cif, export_to_xyz
from utils import load_config, resolve_paths

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generacion de candidatos MOF - Fase Final")
    parser.add_argument("--config", default="config.yaml", help="Ruta a config.yaml")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Ruta al checkpoint .pth (default: models/checkpoints/flow_final.pth)"
    )
    parser.add_argument("--num_samples", type=int, default=20, help="Numero de candidatos a generar")
    parser.add_argument("--num_atoms", type=int, default=64, help="Atomos por estructura")
    parser.add_argument("--num_steps", type=int, default=100, help="Pasos de integracion ODE")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperatura de sampling")
    parser.add_argument("--output_dir", default="results/candidates", help="Directorio de salida")
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_paths(config)

    # Resolver checkpoint: CLI > config > default
    checkpoint_path = args.checkpoint or os.path.join(
        config.get("paths", {}).get("models", "models"),
        "checkpoints", "flow_final.pth"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint no encontrado en {checkpoint_path}")
        return

    print(f"Iniciando Discovery Pipeline...")
    print(f"Usando modelo: {checkpoint_path}")
    print(f"Generando {args.num_samples} candidatos unicos (RK4, {args.num_steps} steps)...")
    
    structures = generate_crystals(
        config=config,
        checkpoint_path=checkpoint_path,
        num_samples=args.num_samples,
        num_atoms=args.num_atoms,
        num_steps=args.num_steps,
        temperature=args.temperature,
    )
    
    for i, structure in enumerate(structures):
        name = f"mof_candidate_{i+1:03d}"
        cif_path = output_dir / f"{name}.cif"
        xyz_path = output_dir / f"{name}.xyz"
        
        export_to_cif(structure, str(cif_path))
        export_to_xyz(structure, str(xyz_path))
        
        a, b, c, _, _, _ = structure["lattice_params"]
        vol = a * b * c
        print(f"  {name} listo | Vol = {vol:.1f} A^3")
        
    print(f"\nGeneracion completada. {args.num_samples} candidatos guardados en '{output_dir}'.")

if __name__ == "__main__":
    main()
