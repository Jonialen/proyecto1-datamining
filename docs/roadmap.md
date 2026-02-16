# Roadmap - Proyecto 1: Análisis Exploratorio INE Guatemala

## Lo que YA está completado (Entrega Fase 1 y 2)

### Infraestructura y Pipeline
- [x] Pipeline ETL funcional (.sav → Parquet → DuckDB).
- [x] Armonización de esquemas y etiquetas inteligentes SPSS.
- [x] API de consulta de alto nivel para Notebooks.

### Análisis Exploratorio (Notebook 01_eda.ipynb)
- [x] **Fase 1.1**: Documentación detallada de limpieza y armonización.
- [x] **Fase 1.2**: 5 Hipótesis validadas con gráficos y discusión técnica (COVID, Escolaridad, Mortalidad Infantil, Vía Pública, Brecha Rural).
- [x] **Fase 1.3**: Clustering con etiquetado dinámico basado en métricas reales (Edad/Asistencia).
- [x] **Auditoría**: Verificación de normalidad (Shapiro-Wilk) y calidad de agrupamiento (Silueta).

### Marco del Proyecto (docs/informe_fase1.md)
- [x] **Fase 2.1**: Situación problemática basada en hallazgos del EDA.
- [x] **Fase 2.2**: Problema científico formulado.
- [x] **Fase 2.3**: Objetivos medibles y alcanzables.

## Lo que falta (Próximos pasos)

### Fase 3: Cierre y Reflexión
- [x] Resumen de hallazgos y validación de congruencia (Ya integrado en el notebook).
- [ ] Revisión final del informe por parte del usuario.

### Fase 4: Entregables Finales
- [ ] Exportar notebook a HTML/PDF (limpio, sin errores).
- [ ] Generar PDF del informe formal.
- [ ] Realizar `git push` final al repositorio.

---
**Nota Final**: Los resultados han sido auditados y contrastados con estadísticas oficiales, mostrando una alta congruencia con la realidad demográfica de Guatemala.
