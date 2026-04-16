# AtomForge &mdash; Deep-Material v2

**Descubrimiento generativo de Metal-Organic Frameworks para almacenamiento de hidrogeno verde mediante Riemannian Flow Matching, screening Grand Canonical Monte Carlo y validacion DFT semi-empirica.**

---

## Descripcion General

AtomForge es un pipeline computacional end-to-end que aprende la distribucion de estructuras cristalinas MOF a partir de la base de datos [CoREMOF 2019](https://zenodo.org/record/3677685) y genera candidatos novedosos optimizados para adsorcion de gases. El sistema opera en cuatro etapas:

1. **Aprender** -- Una GNN SE(3)-equivariante aprende a mapear ruido a estructuras cristalinas validas via Conditional Flow Matching en la variedad Riemanniana SPD(3).
2. **Generar** -- Nuevos candidatos MOF se muestrean integrando la ODE aprendida de t=0 (ruido) a t=1 (cristal), produciendo coordenadas atomicas, especies y parametros de celda unitaria simultaneamente.
3. **Cribar** -- Los candidatos se relajan con campos de fuerza Lennard-Jones y se evaluan para capacidad de adsorcion de CO2 usando un motor Grand Canonical Monte Carlo integrado.
4. **Validar** -- Los mejores candidatos pasan por relajacion semi-empirica GFN2-xTB y calculo de energia de adsorcion de H2.

## Arquitectura

```
CIFs de CoREMOF
    |
    v
[dataset.py] -- CIF -> Grafo Periodico (enlaces VoronoiNN, coords fraccionales, matriz de red)
    |
    v
[crystal_model.py] -- CrystalFlowModel
    |   AtomEncoder (Z -> vector denso)
    |   CrystalGNN (message passing SE(3) con armonicos esfericos l=0,1)
    |   Campo conservativo v = -grad(Phi) via autograd
    |   Tensor de esfuerzo de Virial para dinamicas de red
    |
    v
[train_flow.py] -- Conditional Flow Matching
    |   Perdida Sinkhorn OT con annealing t^4 (per-graph, O(N_i^2))
    |   Estimador de divergencia de Hutchinson (teorema de Liouville)
    |   Perdida de red Log-Euclidiana en Sym(3)
    |   Gradient clipping + clamping de divergencia [-10, 10]
    |
    v
[generate_flow.py] -- Integracion de la ODE (RK4 / dopri5)
    |   LogEuclideanExp: difeomorfismo SPD(3) con backward de Daleckii-Krein
    |   Correccion adaptativa de volumen basada en densidad
    |   WyckoffGuidedRecorder para captura de trayectorias
    |   Exportacion CIF/XYZ con celda unitaria completa
    |
    v
[run_gcmc_analysis.py] -- Screening de Adsorcion GCMC
    |   Motor LJ en Python puro (reglas de mezcla UFF + TraPPE)
    |   Movimientos Insert / Delete / Translate con aceptacion Metropolis
    |   Validacion estructural (deteccion de superposicion)
    |   Generacion de archivos de entrada RASPA2
    |
    v
[validate_xtb.py] -- Validacion DFT
    |   Relajacion GFN2-xTB via ASE + tblite
    |   E_ads(H2) = E(MOF+H2) - E(MOF) - E(H2)
    |   Ejecucion paralela (ProcessPoolExecutor)
    |
    v
  Candidatos rankeados -> results/xtb_validation_report.csv
```

## Componentes Matematicos Clave

| Componente | Implementacion | Proposito |
|---|---|---|
| **LogEuclideanExp** | `torch.autograd.Function` personalizado | Mapea Sym(3) -> SPD(3) via exponencial matricial con backward regularizado por Taylor para autovalores degenerados |
| **Sinkhorn-Knopp OT** | Per-graph con wrapping MIC | Perdida proxy-Wasserstein en el toro plano T^3 para matching de coordenadas fraccionales |
| **Divergencia de Hutchinson** | Estimador estocastico de traza | Impone el teorema de Liouville (preservacion de volumen) en el CNF |
| **Tensor de Virial** | Head de fuerzas aprendido + producto diadico | Impulsa la evolucion de parametros de red durante la generacion |
| **Campo Conservativo** | v = -grad(Phi) via `torch.autograd` | Garantiza dinamicas de coordenadas consistentes con la energia |

## Estructura del Proyecto

```
Deep-Material/
├── config.yaml                  # Configuracion central (modelo, entrenamiento, rutas)
├── requirements.txt             # Dependencias Python
├── tracker.html                 # Dashboard interactivo del proyecto
├── TASKS.md                     # Hoja de ruta de desarrollo
│
├── src/
│   ├── utils.py                 # Constantes quimicas, I/O, LogEuclideanExp, distancia MIC
│   ├── dataset.py               # Pipeline CIF -> grafo periodico (VoronoiNN, PBC)
│   ├── crystal_model.py         # CrystalFlowModel (GNN SE(3) + heads de Flow Matching)
│   ├── train_flow.py            # Bucle de entrenamiento (Sinkhorn OT, div Hutchinson)
│   ├── generate_flow.py         # Integracion ODE, dinamicas CNF, exportacion CIF/XYZ
│   ├── generate_final_batch.py  # Generacion batch de 20 candidatos (CLI)
│   ├── relax_structure.py       # Relajacion rapida LJ via ASE
│   ├── run_gcmc_analysis.py     # Motor GCMC + generador de inputs RASPA (771 lineas)
│   ├── run_batch_gcmc.py        # Orquesta relajacion -> GCMC para todos los candidatos
│   ├── run_porosity_analysis.py # Caracterizacion de porosidad Zeo++ (ASA, AV, Di, Df)
│   ├── validate_xtb.py          # Relajacion GFN2-xTB + energia de adsorcion H2
│   ├── export_cinematic_xyz.py  # Trayectoria T^3 -> Cartesianas para Blender
│   ├── blender_animator.py      # Driver de material Blender para peliculas de cristalizacion
│   ├── plot_crystallization.py  # Visualizacion de metricas de entrenamiento
│   └── unzip_mofs.py            # Extraccion del dataset CoREMOF
│
├── data/
│   ├── raw_cifs/                # Archivos CIF originales de CoREMOF
│   └── processed_v2/           # Grafos periodicos serializados (.pt)
│
├── models/
│   ├── flow_best.pth            # Mejor checkpoint de validacion (val loss: 0.6266)
│   ├── flow_final.pth           # Checkpoint de la epoca final
│   └── checkpoints/             # Checkpoints periodicos (cada 50 epocas)
│
└── results/
    ├── candidates/              # 20 CIFs de MOFs generados + variantes relajadas con LJ
    ├── gcmc/
    │   └── gcmc_results.csv     # Resultados del screening de adsorcion GCMC
    └── xtb_validated/           # Estructuras relajadas con GFN2-xTB (pendiente)
```

## Resultados de Entrenamiento

| Metrica | Valor |
|---|---|
| Dataset | CoREMOF 2019 (9,784 train / 1,223 val) |
| Arquitectura | Backbone SchNet, 64 dim oculta, 3 capas, ~100K parametros |
| Entrenamiento | 500 epocas en 8 nucleos CPU (Xeon) |
| Mejor Val Loss | **0.6266** |
| Candidatos Generados | 20 MOFs (RK4, 100 pasos de integracion) |

## Resultados de Screening GCMC (extracto)

| Candidato | Loading (mg/g) | Loading (cm3 STP/cm3) | Henry (mol/kg/Pa) |
|---|---|---|---|
| mof_candidate_014 | **48.02** | 27.10 | 1.09e-05 |
| mof_candidate_006 | 23.44 | 13.23 | 5.33e-06 |
| mof_candidate_001 | 23.37 | 13.19 | 5.31e-06 |
| mof_candidate_002 | 11.75 | 6.63 | 2.67e-06 |
| mof_candidate_004 | 6.60 | 3.72 | 1.50e-06 |

*CO2 a 298 K, 1 bar. 5000 ciclos de produccion, 2000 de equilibracion.*

## Inicio Rapido

```bash
# 1. Configurar entorno
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Preparar datos
python src/unzip_mofs.py
python src/dataset.py --action all

# 3. Entrenar modelo
python -u src/train_flow.py --config config.yaml

# 4. Generar candidatos
python src/generate_final_batch.py --num_samples 20

# 5. Screening de adsorcion de gas (relajar + GCMC)
python src/run_batch_gcmc.py

# 6. Validacion DFT de mejores candidatos (requiere tblite)
python src/validate_xtb.py
```

## Dependencias

- **Core:** PyTorch >= 2.0, PyTorch Geometric >= 2.4, e3nn
- **Quimica:** pymatgen, ASE, tblite (opcional, para DFT)
- **Tracking:** Weights & Biases

## Referencias

| Paper | Venue | Relevancia |
|---|---|---|
| MOFFlow | ICLR 2025 | Riemannian Flow Matching en SE(3) para MOFs |
| FlowMM | ICML 2024 | Riemannian FM para generacion de cristales |
| CDVAE | ICLR 2022 | Modelo de difusion baseline para cristales |

## Licencia

Proyecto academico.
