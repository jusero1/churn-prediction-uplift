# Model Card — churn-prediction-uplift

## Propósito

Soportar la retención en tres decisiones: **quién** se va (propensión),
**cuándo** (supervivencia) y **a quién sirve contactar** (uplift), cerrando con
una simulación de campaña.

## Modelos

| Componente | Modelo | Dato |
|---|---|---|
| Propensión de fuga | LightGBM (`scale_pos_weight`) | Cell2Cell (51.047, churn 28,8 %) |
| Supervivencia | Kaplan-Meier + Cox PH (`lifelines`) | Cell2Cell (`MonthsInService`) |
| Uplift | T-learner (dos LightGBM) | Hillstrom (Men's E-Mail vs. control) |

## Desempeño (test)

**Propensión** (n ≈ 15.300):

| AUC | Gini | KS |
|---|---|---|
| 0,675 | 0,349 | 0,255 |

Modesto pero realista para churn de telecom (variables de uso, sin buró ni
contrato). El ranking es suficiente para priorizar gestión.

**Supervivencia:** vida media (mediana) de **34 meses**. Cox — razones de riesgo
(HR > 1 aumenta el riesgo de fuga):

| Covariable | Hazard ratio | Dirección |
|---|---|---|
| `CustomerCareCalls` | 1,008 | aumenta churn |
| `DroppedCalls` | 1,005 | aumenta churn |
| `OverageMinutes` | 1,001 | aumenta churn |
| `MonthlyMinutes` | 0,9999 | reduce churn |
| `MonthlyRevenue` | 0,999 | reduce churn |

Lectura: más llamadas a servicio al cliente y más llamadas caídas (mala
experiencia) elevan el riesgo; más uso y consumo lo bajan.

**Uplift** (Hillstrom, Men's vs. control): efecto promedio (ATE) sobre visitas
**+7,7 pp**. Qini = 0,0017. Uplift observado del **decil 1 = +9,5 pp**, casi 2×
el del medio (~+5 pp) → el modelo concentra el efecto incremental arriba.

**Simulación de campaña** (escenario: costo 0,40, valor 5,0 por respuesta): el
valor neto se maximiza contactando el **top 30 %** por uplift; contactar a **toda
la base pierde dinero** (valor neto −224). El *sleeping dog* existe: hay clientes
con uplift bajo o negativo a los que no conviene contactar.

## Limitaciones conocidas

- **Uplift de señal débil:** el efecto heterogéneo de Hillstrom es sutil (Qini
  pequeño); los deciles no son perfectamente monótonos. Es propio del dataset.
- **Supervivencia aproximada:** corte transversal con antigüedad, no cohorte
  longitudinal (ver [data_sources.md](data_sources.md)).
- **Propensión modesta:** AUC 0,675; con contrato, buró o histórico transaccional
  mejoraría. La calibración no se ajustó (prioriza ranking).
- **Dos dominios:** telecom y retail; la metodología se transfiere, los datos no.

## Gobierno

- **Responsable:** jusero1. **Fecha de revisión:** 2026-08-10.
- **Reproducibilidad:** semillas fijas; `make data` + `make train` reconstruyen
  todos los artefactos.
