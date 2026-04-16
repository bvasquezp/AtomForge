import os
import sys
import subprocess
from pathlib import Path

def main():
    print("Iniciando batch de relajacion y GCMC para 20 candidatos...")
    
    # Aseguramos que estamos usando el ejecutable actual del entorno
    python_exe = sys.executable
    
    # Obtener la ruta raiz del proyecto (asumiendo que este script esta en src/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Scripts a ejecutar
    relax_script = script_dir / "relax_structure.py"
    gcmc_script = script_dir / "run_gcmc_analysis.py"
    
    for i in range(1, 21):
        num_str = f"{i:03d}"
        cif_path = project_root / "results" / "candidates" / f"mof_candidate_{num_str}.cif"
        relaxed_path = project_root / "results" / "candidates" / f"mof_candidate_{num_str}_relaxed.cif"
        
        if cif_path.exists():
            print("="*60)
            print(f"[{i}/20] Procesando candidato {num_str}...")
            print("="*60)
            
            # 1. Relajar estructura
            print(f"-> Relajando estructura (LJ)...")
            try:
                # Se pasa cwd=project_root para consistencia con los demas scripts
                subprocess.run([python_exe, str(relax_script), str(cif_path), str(relaxed_path)], 
                               check=True, cwd=project_root)
            except subprocess.CalledProcessError as e:
                print(f"Error relajando {cif_path}: {e}")
                continue
                
            # 2. Correr GCMC
            print(f"-> GCMC Simulacion...")
            try:
                subprocess.run([
                    python_exe, str(gcmc_script), 
                    "--cif", str(relaxed_path), 
                    "--cycles", "5000", 
                    "--equil", "2000"
                ], check=True, cwd=project_root)
            except subprocess.CalledProcessError as e:
                print(f"Error en simulacion GCMC para {relaxed_path}: {e}")
                continue

    print("\nBatch completado!")

if __name__ == "__main__":
    main()
