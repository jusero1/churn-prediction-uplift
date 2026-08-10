# Metodología

## Pregunta de negocio

*¿A quién estoy a punto de perder — y a quién sirve contactar?* El proyecto no se
queda en predecir la fuga: la separa en tres decisiones operativas.

## Definición del evento y la ventana (lo que casi todos omiten)

- **Churn (Cell2Cell):** el evento es la fuga contractual (`Churn = 'Yes'`)
  observada en el corte de datos. Para supervivencia, la duración es la
  antigüedad (`MonthsInService`) y los clientes activos son **censurados por la
  derecha** a su antigüedad actual.
- **Uplift (Hillstrom):** el tratamiento es recibir el email (de hombres) frente
  al control (sin email); el outcome es una **visita** al sitio en las 2 semanas
  posteriores al envío. Al ser un experimento aleatorizado, la diferencia
  tratamiento−control es un efecto causal, no una correlación.

## Componente 1 — Propensión de fuga

Clasificador **LightGBM** sobre las 55 variables de Cell2Cell (categóricas
nativas, desbalance con `scale_pos_weight`). Partición estratificada, semilla
fija. Métricas de discriminación con `riskkit` (KS, Gini, AUC) — la misma
implementación de todo el portafolio.

## Componente 2 — Supervivencia (tiempo hasta la fuga)

Responde *cuándo*, no solo *si*. **Kaplan-Meier** (curva de retención no
paramétrica, global y por segmento de uso) y **Cox** de riesgos proporcionales
(razones de riesgo por covariable: cuánto multiplica cada driver el riesgo
instantáneo de fuga). Vía `lifelines`.

## Componente 3 — Uplift (a quién sirve contactar)

El diferenciador. Un cliente que se va (o se queda) haga lo que haga la empresa
es **contacto desperdiciado**; solo los *persuadables* (uplift positivo) valen la
inversión. Modelo **T-learner** (dos modelos): se ajusta un modelo de outcome en
el grupo tratado y otro en el control, y el uplift estimado es la diferencia de
sus probabilidades. Requiere el tratamiento aleatorizado que aporta Hillstrom.

**Evaluación de uplift** (no sirve el accuracy): **curva y coeficiente de Qini**,
y **uplift observado por decil** (validación operativa: el decil 1 debe mostrar
el mayor uplift real tratado−control).

## Cierre de negocio — Simulación de campaña

Dado un costo por contacto y un valor por respuesta incremental (parámetros de
escenario, no cifras del dato), se contacta al top-*k* % por uplift predicho y se
estima la respuesta incremental de ese subgrupo desde el propio experimento. La
tabla muestra el punto donde **dejar de ampliar la campaña**: más allá, el costo
supera al valor incremental.

## Reproducibilidad

Semillas fijas y versiones en `pyproject.toml`. `make data` reconstruye los datos
crudos con checksum; `make train` reconstruye modelos, métricas y figuras. Los
tests corren sobre fixtures sintéticas con efecto conocido, sin descargar datos.
