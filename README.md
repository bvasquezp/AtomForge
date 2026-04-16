# Deep-Material v2 (Genesis-007 Project)

Generacion de Metal-Organic Frameworks (MOFs) mediante **Riemannian Flow Matching** con validacion **DFT** para almacenamiento de hidrogeno verde.

## Arquitectura

```
CIF (CoREMOF) -> Grafos Periodicos -> Flow Matching (SchNet backbone) -> Estructuras 3D -> Validacion DFT (xTB)
```

El modelo genera conjuntamente:
- **Tipos atomicos** (clasificacion categorica)
- **Coordenadas fraccionales** (en toro T^3, periodicidad natural)
- **Parametros de celda unitaria** (a, b, c, alpha, beta, gamma)

## Estructura del Proyecto

```
Deep-Material/
├── config.yaml              # Configuracion central
├── requirements.txt         # Dependencias
├── data/
│   ├── raw_cifs/            # Archivos CIF originales (CoREMOF)
│   └── processed_v2/        # Grafos periodicos (.pt)
├── models/                  # Checkpoints del modelo
├── src/
│   ├── utils.py             # Utilidades compartidas
│   ├── dataset.py           # Pipeline de datos con PBC
│   ├── crystal_model.py     # CrystalFlowModel (SchNet + Flow Matching)
│   ├── train_flow.py        # Entrenamiento Flow Matching
│   └── generate_flow.py     # Generacion de nuevos MOFs
├── results/                 # Resultados de la generacion y validacion GCMC
└── top_candidates/          # Candidatos validados
```

## Uso Rapido

### 1. Preprocesar datos
```bash
python src/dataset.py --action all
```

### 2. Entrenar modelo
```bash
python src/train_flow.py
```

### 3. Generar nuevos MOFs
```bash
python src/generate_flow.py --checkpoint models/flow_best.pth --num_samples 10
```

## Exploracion y Validacion Masiva (v2)

### 4. Orquestar Monte Carlo GCMC
```bash
python src/run_batch_gcmc.py 
```

### 5. Caracterizacion DFT - GFN2-xTB 
```bash
python src/validate_xtb.py
```

## Modelos de Referencia

| Modelo | Venue | Innovacion |
|---|---|---|
| MOFFlow | ICLR 2025 | Riemannian FM en SE(3) para MOFs |
| FlowMM | ICML 2024 | Riemannian FM para cristales |
| CDVAE | ICLR 2022 | Diffusion model para cristales |

## Licencia

Proyecto academico / concurso de innovacion.
