# src/unzip_mofs.py
import os
import zipfile
import shutil

# --- CONFIGURACIÓN ---
# Nombre exacto del archivo que bajaste
ZIP_FILENAME = "CoREMOF2019_public_v2_20241119.zip" 
# Carpeta donde guardaremos los cristales limpios
DEST_DIR = os.path.join("data", "raw_cifs")
# Cantidad de materiales a extraer (para no saturar tu disco)
LIMIT = 500

def main():
    # Verificar si el ZIP existe
    if not os.path.exists(ZIP_FILENAME):
        print(f"ERROR: No encuentro el archivo '{ZIP_FILENAME}'.")
        print("Asegurate de que el ZIP esté en la carpeta principal 'Deep-Material'.")
        return

    print(f"--- INICIANDO EXTRACCIÓN QUIRÚRGICA ---")
    print(f"Objetivo: Extraer {LIMIT} estructuras en '{DEST_DIR}'")

    # Crear carpeta de destino si no existe
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    extracted_count = 0

    try:
        with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
            # Obtener lista de todos los archivos dentro del ZIP
            all_files = zip_ref.namelist()
            
            # Filtrar solo los que terminan en .cif y evitar carpetas ocultas (__MACOSX)
            cif_files = [f for f in all_files if f.endswith('.cif') and "__MACOSX" not in f]
            
            print(f"🔍 Encontrados {len(cif_files)} archivos .cif en el ZIP.")
            
            for file_path in cif_files:
                if extracted_count >= LIMIT:
                    break
                
                # Nombre limpio del archivo (sin carpetas previas)
                filename = os.path.basename(file_path)
                target_path = os.path.join(DEST_DIR, filename)
                
                # Leemos el archivo del ZIP y lo escribimos directamente en el destino
                # (Esto "aplana" la estructura de carpetas)
                with zip_ref.open(file_path) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                
                extracted_count += 1
                
                if extracted_count % 50 == 0:
                    print(f"Procesados: {extracted_count}/{LIMIT}")

    except zipfile.BadZipFile:
        print("ERROR: El archivo ZIP parece estar dañado o incompleto.")
    except Exception as e:
        print(f" Ocurrió un error inesperado: {e}")

    print(f"--- COMPLETADO ---")
    print(f" Se han guardado {extracted_count} cristales en '{DEST_DIR}'.")
    print("¡Listo para procesar!")

if __name__ == "__main__":
    main()