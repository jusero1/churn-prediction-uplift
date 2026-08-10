# Fuentes de datos

El proyecto tiene **dos componentes, cada uno sobre su propia fuente real y
pública**, porque ninguna fuente única cubre bien las tres técnicas
(propensión, supervivencia y uplift). El uplift, en particular, exige datos con
**tratamiento/control aleatorizado**, que un dataset de churn normal no tiene.

## Fuente 1 — Cell2Cell (propensión + supervivencia)

| Campo | Detalle |
|---|---|
| **Nombre** | Cell2Cell telecom churn |
| **Origen** | Teradata Center for CRM, Duke University. Datos reales anonimizados de una operadora de telecom. |
| **Espejo usado** | Kaggle: `jpacse/datasets-for-churn-telecom` (`cell2celltrain.csv`) |
| **Licencia** | Declarada como "Unknown" en Kaggle → **no se redistribuye**. Solo se descarga vía `download.py`; `data/` está en `.gitignore` y **no** se versiona muestra. |
| **Fecha de descarga** | 2026-08-10 |
| **Volumen** | 51.047 clientes × 58 variables |
| **Objetivo** | `Churn` (Yes/No) → `churn` (1 = fuga). Tasa de evento 28,8 %. |
| **Tiempo (supervivencia)** | `MonthsInService` (antigüedad) como duración; los no-fuga son observaciones **censuradas por la derecha**. |

## Fuente 2 — Hillstrom / MineThatData (uplift)

| Campo | Detalle |
|---|---|
| **Nombre** | Kevin Hillstrom — MineThatData E-Mail Analytics Challenge (2008) |
| **Origen** | Experimento **aleatorizado real** de un minorista: clientes asignados al azar a email de hombres, email de mujeres o **sin email** (control). |
| **URL** | http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv |
| **Licencia** | Publicado libremente por el autor para el reto. Descarga directa, sin credenciales. |
| **Fecha de descarga** | 2026-08-10 |
| **Volumen** | 64.000 clientes × 12 variables |
| **Tratamiento** | `segment`. Se usa **Men's E-Mail vs. No E-Mail** (contraste canónico de uplift, efecto más heterogéneo); se descartan las filas de Women's E-Mail. |
| **Outcome** | `visit` (visita en las 2 semanas siguientes, ~15 %). Más señal que `conversion` (~1 %). |

## Descarga reproducible

`src/churn_uplift/data/download.py` (objetivo `make data`) descarga Cell2Cell por
la **API de Kaggle** (requiere `~/.kaggle/kaggle.json`) y Hillstrom por **URL
directa**, y registra el SHA-256 de cada archivo en `data/checksums.txt`.

## Limitaciones declaradas

- **Supervivencia sobre un corte transversal:** Cell2Cell es una foto con
  antigüedad, no una cohorte longitudinal seguida en el tiempo. Usar
  `MonthsInService` como duración con `churn` como evento es un enfoque estándar
  pero **aproximado** (la censura y el sesgo de supervivencia se asumen, no se
  observan encadenados).
- **Uplift de señal sutil:** el efecto heterogéneo en Hillstrom es real pero
  débil; el Qini es pequeño (ver model card). Es un hecho conocido del dataset,
  no un defecto del método.
- **Dos dominios distintos:** telecom (churn) y retail (email). Se declara: la
  *metodología* es transferible; los datos no se mezclan.
