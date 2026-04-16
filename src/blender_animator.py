"""
Script de Blender para importar y animar Multi-Frame XYZ de Deep-Material v2.
Fase 5: Renderizado Cinematico y Transition.
Instrucciones: 
1. Abre Blender.
2. Ve a la pestaña de Scripting.
3. Carga este script y corrige la ruta a tu `movie_render.xyz`.
4. Ejecuta (Run Script).
"""
import bpy
import math
import os

# --- CONGIGURACION ---
XYZ_FILE = r"C:\Users\benja\OneDrive - utem.cl\Documentos\Deep-Material\generated_crystals\trajectories\movie_render.xyz"
ATOM_RADIUS = 0.3
MATERIAL_NAME = "AtomMaterial"

def clean_scene():
    """Limpia todos los objetos en la escena"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def load_xyz_trajectory(filepath):
    """Lee el archivo multiframe XYZ con c_state"""
    frames = []
    with open(filepath, 'r') as f:
        while True:
            line_num = f.readline()
            if not line_num:
                break
            num_atoms = int(line_num.strip())
            comment = f.readline().strip()
            
            frame_atoms = []
            for _ in range(num_atoms):
                parts = f.readline().strip().split()
                if not parts: continue
                elem = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                c_state = float(parts[4])
                frame_atoms.append((elem, (x, y, z), c_state))
                
            frames.append(frame_atoms)
    return frames

def setup_material():
    """Crea un material reactivo al c_state"""
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if not mat:
        mat = bpy.data.materials.new(name=MATERIAL_NAME)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Crear Nodos
    out_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Emission strength animable via Custom Properties
    principled.inputs['Emission Strength'].default_value = 1.0
    
    mat.node_tree.links.new(principled.outputs[0], out_node.inputs[0])
    return mat

def create_animation():
    if not os.path.exists(XYZ_FILE):
        print(f"Error: {XYZ_FILE} no encontrado.")
        return

    frames = load_xyz_trajectory(XYZ_FILE)
    print(f"Cargados {len(frames)} cuadros.")
    
    clean_scene()
    mat = setup_material()
    
    # Colección para mantener organizado
    collection = bpy.data.collections.new("Crystals")
    bpy.context.scene.collection.children.link(collection)
    
    # Instanciar átomos base
    num_atoms = len(frames[0])
    atom_objects = []
    
    # Crear esferas para cada átomo
    for i in range(num_atoms):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=ATOM_RADIUS, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = f"Atom_{i:03d}"
        obj.data.materials.append(mat)
        
        # Mover a colección
        bpy.context.scene.collection.objects.unlink(obj)
        collection.objects.link(obj)
        atom_objects.append(obj)
    
    # Animar por frames
    frame_step = 2 # Configurar la velocidad
    
    for f_idx, frame_data in enumerate(frames):
        blender_frame = (f_idx * frame_step) + 1
        bpy.context.scene.frame_set(blender_frame)
        
        for atom_idx, (elem, pos, c_state) in enumerate(frame_data):
            obj = atom_objects[atom_idx]
            obj.location = pos
            obj.keyframe_insert(data_path="location", index=-1)
            
            # Grabar propiedad customizada c_state para el material (0.0 ruido a 1.0 cristal)
            obj["c_state"] = c_state
            
            # Aqui podrías programar el Driver del material basado en obj["c_state"]
            # Para hacerlos brillar cuando se ordenan.
            
    # Zoom out de camara de cortesia
    bpy.ops.object.camera_add(location=(15, -15, 10))
    cam = bpy.context.active_object
    cam.rotation_euler = (math.radians(60), 0, math.radians(45))
    
    print("Animación cargada con éxito en Blender!")

if __name__ == "__main__":
    create_animation()
