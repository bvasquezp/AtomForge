import os
import glob
import subprocess
import pandas as pd
from pathlib import Path
import shutil

def find_zeo_executable():
    # Primero: miniconda (donde lo instalamos)
    conda_path = os.path.expanduser("~/miniconda3/bin/network")
    if os.path.exists(conda_path):
        return conda_path
    # Segundo: PATH del sistema
    for opt in ["network", "network.exe"]:
        path = shutil.which(opt)
        if path:
            return path
    # Tercero: búsqueda en el proyecto
    pwd = Path(__file__).resolve().parent.parent
    for root, dirs, files in os.walk(pwd):
        for f in files:
            if f in ["network", "network.exe"]:
                full_path = os.path.join(root, f)
                if os.access(full_path, os.X_OK):
                    return full_path
    return None

def parse_res(file_path):
    di, df = 0.0, 0.0
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read().strip()
            if content:
                parts = content.split()
                if len(parts) >= 4:
                    di = float(parts[1])
                    df = float(parts[2])
    return di, df

def parse_sa(file_path):
    asa_a2, asa_m2g, nasa_m2g = 0.0, 0.0, 0.0
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read().strip()
            parts = content.split()
            for i, p in enumerate(parts):
                if p == "ASA_A^2:" and i + 1 < len(parts):
                    asa_a2 = float(parts[i + 1])
                if p == "ASA_m^2/g:" and i + 1 < len(parts):
                    asa_m2g = float(parts[i + 1])
                if p == "NASA_m^2/g:" and i + 1 < len(parts):
                    nasa_m2g = float(parts[i + 1])
    return asa_a2, asa_m2g, nasa_m2g

def parse_vol(file_path):
    av = 0.0
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read().strip()
            parts = content.split()
            for i, p in enumerate(parts):
                if p == "AV_cm^3/g:" and i + 1 < len(parts):
                    av = float(parts[i + 1])
    return av

def main():
    zeo_exe = find_zeo_executable()
    
    candidates_dir = Path("results/candidates_v2")
    report_file = Path("results/summary_report.csv")
    
    cifs = sorted(glob.glob(str(candidates_dir / "*.cif")))
    
    if not cifs:
        print("No se encontraron candidatos .cif en results/candidates_v2/")
        return
        
    print(f"Iniciando analisis de porosidad (Zeo++) para {len(cifs)} archivos...")
    
    if not zeo_exe:
        print("  IMPORTANTE: Ejecutable de Zeo++ ('network' o 'network.exe') no encontrado en el PATH.")
        print("  El script procesará los archivos asumiendo salidas manuales, o dejará todo en 0.")
        
    results = []
    
    for cif in cifs:
        name = Path(cif).stem
        print(f"  ⚡ Analizando {name}...", end=" ", flush=True)
        
        base_path = candidates_dir / name
        res_file = str(base_path) + ".res"
        sa_file = str(base_path) + ".sa"
        vol_file = str(base_path) + ".vol"
        
        if zeo_exe:
            # -res (Diamentro)
            subprocess.run([zeo_exe, "-res", res_file, cif], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # -sa (Area Superficial con sonda de Hidrogeno 1.2A)
            subprocess.run([zeo_exe, "-sa", "1.2", "1.2", "5000", sa_file, cif], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # -vol (Volumen Accesible con sonda de Hidrogeno 1.2A)
            subprocess.run([zeo_exe, "-vol", "1.2", "1.2", "5000", vol_file, cif], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        # Parse output files
        di, df = parse_res(res_file)
        asa_a2, asa_m2g, nasa_m2g = parse_sa(sa_file)
        av = parse_vol(vol_file)
        
        results.append({
            "ID": name,
            "Density": 0.0, # Se calcularía con pymatgen u output extendido de Zeo++
            "Di": round(di, 3),
            "Df": round(df, 3),
            "ASA_A2": round(asa_a2, 2),
            "ASA_m2g": round(asa_m2g, 2),
            "NASA_m2g": round(nasa_m2g, 2),
            "AV": round(av, 4)
        })
        print("Hecho.")
        
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="ASA_m2g", ascending=False)
    
    # Save CSV
    df_results.to_csv(report_file, index=False)
    
    print(f"\n🏆 Resultados Consolidados en '{report_file}':")
    print(df_results.head().to_string())
    
    if not df_results.empty:
        top = df_results.iloc[0]
        print(f"\n🌟 TOP 1 Candidate: {top['ID']} | ASA_m2g: {top['ASA_m2g']:.2f} m^2/g | AV: {top['AV']:.3f} cm^3/g")
        
if __name__ == "__main__":
    main()
