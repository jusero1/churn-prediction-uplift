# Diccionario de datos

## Cell2Cell (propensión + supervivencia)

51.047 clientes, 58 variables. Selección de campos clave (el resto son contadores
de uso: minutos, llamadas caídas/bloqueadas, recargas, etc.).

| Campo | Tipo | Definición |
|---|---|---|
| `Churn` → `churn` | objetivo | 1 si el cliente se fugó, 0 si sigue activo. |
| `MonthsInService` | numérica | Antigüedad en meses. Duración para el análisis de supervivencia. |
| `MonthlyRevenue` | numérica | Ingreso mensual promedio. |
| `MonthlyMinutes` | numérica | Minutos mensuales promedio. |
| `CustomerCareCalls` | numérica | Llamadas al servicio al cliente (driver de riesgo principal). |
| `DroppedCalls` | numérica | Llamadas caídas (calidad de servicio). |
| `OverageMinutes` | numérica | Minutos por encima del plan. |
| `RetentionCalls` | numérica | Llamadas al área de retención. |
| `CreditRating` | categórica | Calificación crediticia del cliente. |
| `CustomerID`, `ServiceArea` | — | Identificador y área geográfica; se descartan (alta cardinalidad). |

Limpieza: `Churn` Yes/No → 1/0; nulos numéricos imputados con la mediana;
categóricas con `Unknown` y convertidas a `category` para LightGBM.

## Hillstrom (uplift)

64.000 clientes, 12 variables (se usan las de Men's E-Mail vs. No E-Mail).

| Campo | Tipo | Definición |
|---|---|---|
| `segment` → `treatment` | tratamiento | 1 = recibió email de hombres; 0 = sin email (control). |
| `visit` → `outcome` | outcome | 1 si visitó el sitio en las 2 semanas siguientes. |
| `conversion` | outcome alt. | 1 si compró (raro, ~1 %). |
| `spend` | numérica | Gasto en las 2 semanas siguientes. |
| `recency` | numérica | Meses desde la última compra. |
| `history` | numérica | Gasto histórico ($). |
| `history_segment` | categórica | Tramo de gasto histórico. |
| `mens`, `womens` | binaria | Compró en la categoría de hombres / mujeres el año anterior. |
| `newbie` | binaria | Cliente nuevo (último año). |
| `zip_code` | categórica | Urbano / suburbano / rural. |
| `channel` | categórica | Canal de compra previo (phone / web / multichannel). |

Las categóricas se codifican con one-hot (`drop_first`) y los nombres se
normalizan a identificadores válidos para LightGBM.
