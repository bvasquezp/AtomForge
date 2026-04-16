# ROADMAP DE TAREAS (Deep-Material-v2)

* [x] **FASE 1: Refactorización Teórica.** Crear `PhysicsInformedLoss` en `train_flow.py` para integrar $\exp(\mathbf{A})$, Wasserstein $\mathcal{L}_{sym}$ y Divergencia.
* [ ] **FASE 2: Entrenamiento del Oráculo.** Ejecutar entrenamiento completo y guardar logs.
* [ ] **FASE 3: Validación Visual.** Completar e integrar `plot_crystallization.py`.
* [ ] **FASE 4: Inferencia Cinemática.** Completar e integrar `WyckoffGuidedRecorder` en el bucle de generación de Euler.
* [ ] **FASE 5: Pipeline de Cine.** Integrar exportador XYZ con PBC y autómata de Blender.
* [ ] **FASE 6: Oráculo Cuántico (DFT).** Relajación final con `gfn2-xtb` para validar matriz Hessiana.
