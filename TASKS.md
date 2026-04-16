# AtomForge -- Roadmap de Desarrollo

## Fase 0: Cimientos y Consolidacion -- COMPLETADA
- [x] Unificacion del repositorio, eliminacion de scripts legacy
- [x] Configuracion centralizada (`config.yaml`) y utilidades compartidas (`src/utils.py`)
- [x] Entorno virtual con PyTorch, PyG, e3nn, torchdiffeq
- [x] Arquitectura unificada en `src/crystal_model.py`
- [x] Pipeline de datos PBC con VoronoiNN (`src/dataset.py`)
- [x] Descarga y extraccion CoREMOF 2019 (15,000 CIFs)

## Fase I: Fundamentos Fisico-Matematicos -- COMPLETADA
- [x] LogEuclideanExp (Autograd Function) con Taylor para singularidades
- [x] Metrica MIC en el Toroide T^3
- [x] Campo Vectorial Conservativo via Potencial Escalar Phi
- [x] Armonicos Esfericos (e3nn) para equivariancia SE(3)
- [x] Tensor de Esfuerzo de Virial para dinamicas de celda
- [x] Sinkhorn Loss con Annealing Termodinamico t^4
- [x] Teorema de Liouville via Estimador de Hutchinson
- [x] Sinkhorn per-graph O(N_i^2) y OpenMP tuning (8 hilos)

## Fase II: Entrenamiento Masivo -- COMPLETADA
- [x] Procesamiento CoREMOF: 9,784 train / 1,223 val
- [x] 500 epocas con 8 cores (mejor val loss: 0.6266)
- [x] 12 checkpoints guardados (cada 50 epocas + best + final)
- [x] Parche emergencia E150: lambda_coords 5.0, lambda_div 0.001
- [x] Clamping de divergencia [-10, 10] y gradient clipping global
- [x] Fallback de Bulk Modulus via heuristica |F/V|

## Fase III: Descubrimiento y Caracterizacion -- EN CURSO
- [x] Generacion de 20 candidatos MOF (RK4, 100 steps)
- [x] Script `run_porosity_analysis.py` con parsers para Zeo++
- [x] Motor GCMC Python puro integrado y validado
- [x] Relajacion LJ + screening GCMC de candidatos (batch)
- [ ] **BLOQUEANTE:** Instalar Zeo++ (network.exe) para porosidad (Di, Df, ASA, AV)
- [ ] Ranking final combinado (GCMC + porosidad) en summary_report.csv

## Fase IV: Validacion DFT -- PENDIENTE
- [ ] Instalar tblite / xtb-python
- [ ] Relajacion geometrica GFN2-xTB del TOP 10
- [ ] Calculo de energias de adsorcion E_ads(H2)
- [ ] Reporte final en `results/xtb_validation_report.csv`

## Fase V: Visualizacion Cinematica -- COMPLETADA
- [x] WyckoffGuidedRecorder integrado en `generate_flow.py`
- [x] `export_cinematic_xyz.py` con T^3 -> Cartesianas
- [x] `blender_animator.py` con driver de material basado en c_state

## Pipeline de Ejecucion

```powershell
python src/unzip_mofs.py                                    # Extraer CoREMOF
python src/dataset.py --action all                          # CIF -> grafos periodicos
python -u src/train_flow.py --config config.yaml            # Entrenamiento
python src/generate_final_batch.py --num_samples 20         # Generacion
python src/run_batch_gcmc.py                                # Relajacion + GCMC
python src/run_porosity_analysis.py                         # Porosidad (requiere Zeo++)
python src/validate_xtb.py                                  # DFT (requiere tblite)
```
