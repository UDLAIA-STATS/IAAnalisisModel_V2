# SoccerAnalysis_AI
Motor de visión por computador y analítica deportiva para el análisis automatizado de partidos de fútbol.

Pueden acceder al stack completo mediante este enlace: <a href="https://github.com/UDLAIA-STATS/">Repositorio Principal</a>
Pueden acceder a una explicación más técnica del proyecto mediante este <a href="https://youtu.be/dXAw_b-Ry6I">video</a>.

El sistema permite detectar, rastrear y analizar:
* Jugadores.
* Balón.
* Arcos.
* Número de camiseta.
* Color de camiseta.

A partir de estos datos se generan métricas cinemáticas, información de posicionamiento y reportes para el análisis deportivo.
## Objetivo
Transformar datos de video en información estructurada sobre el partido mediante un pipeline automatizado que incluye:

* **Detección de objetos:** identificación de jugadores, balón y arcos.
* **Multi-Object Tracking:** seguimiento de los objetos detectados durante el partido.
* **Resolución de identidad:** reconocimiento del número y color de camiseta.
* **Análisis cinemático:** cálculo de velocidad, aceleración y distancia recorrida.
* **Transformación espacial:** conversión de coordenadas mediante homografía del campo.
* **Generación de reportes:** producción de datos, métricas y visualizaciones.

## Resultados

El procesamiento genera:

* Estados frame por frame de jugadores, balón y arcos.
* Posiciones y trayectorias.
* Velocidad, aceleración y distancia recorrida.
* Identificación mediante número de camiseta.
* Reportes estadísticos en formato CSV.
* Videos anotados con detecciones y tracking.
* Métricas de rendimiento y tiempos de ejecución.

## Arquitectura

El proyecto utiliza una arquitectura en capas que separa la presentación, orquestación, procesamiento y persistencia.

```text
Presentation
     │
     ▼
Orchestration
     │
     ▼
Vision / Processing
     │
     ▼
Persistence
```

El procesamiento es coordinado por `Orchestrator`, que ejecuta las diferentes etapas del análisis:

```text
VideoDownload
      ↓
ObjectDetection
      ↓
NumberAndColorRecognition
      ↓
ConversionCalculatorSteps
      ↓
ValidationProcess
      ↓
Report Generation
```

Esta arquitectura permite mantener independientes las etapas del pipeline y facilita el intercambio de implementaciones mediante `AnalysisStepHandler`.

## Estructura del proyecto
```text
SoccerAnalysis_AI/
├── main.py                  # Punto de entrada de FastAPI
├── pyproject.toml           # Dependencias y configuración
├── docker-compose.yaml      # Configuración de Docker
│
├── res/
│   ├── database/
│   ├── models/
│   │   ├── yolo/            # Modelos YOLO
│   │   ├── trocr/           # Modelos OCR
│   │   └── config/          # Configuración de trackers
│   │       └── bytetrack.yaml
│   └── outputs/
│       ├── videos/
│       ├── images/
│       ├── reports/
│       ├── annotated/
│       ├── metrics/
│       └── time_reports/
│
└── src/
    ├── config/              # Configuración y rutas
    ├── core/                # Lógica central y procesamiento
    ├── entities/            # Entidades e interfaces
    └── presentation/        # API y routers de FastAPI
```

La estructura del proyecto separa los recursos del sistema de la lógica de aplicación y de la capa de presentación.

## Requisitos
* Python `>= 3.8` y `< 3.14`.
* `uv` como gestor de paquetes.
* PostgreSQL.
* Pesos de los modelos requeridos.
* Configuración de ByteTrack mediante `bytetrack.yaml`.

Principales dependencias:
```text
FastAPI
Uvicorn
Ultralytics
Supervision
MediaPipe
SQLModel
Psycopg
Pandas
PySpark
Boto3
Dynaconf
Logfire
OpenCV
EasyOCR
Transformers
PyTorch
```

Las dependencias completas se encuentran definidas en `pyproject.toml`.
## Instalación

Instalar las dependencias mediante `uv`:
```bash
uv sync
```

Configurar las variables de entorno utilizando el prefijo `APP_`. Por ejemplo:

```text
APP_DATABASE_URL
```

Los modelos YOLO requeridos deben ubicarse en:
```text
./res/models/yolo/
```

La aplicación valida los modelos y prepara la estructura de recursos durante el inicio.

## Ejecución
Iniciar la aplicación mediante:

```bash
python main.py
```

La API se ejecutará en el puerto `6070`:

```text
http://localhost:6070/docs
```

Para verificar el servicio:

```bash
curl -X GET http://localhost:6070/docs
```

El router de análisis expone el endpoint utilizado para iniciar el procesamiento:

```http
POST /analyze/run
```

## Docker

El proyecto incluye `docker-compose.yaml`, que define el servicio `ia-model-v2`.

El servicio se construye a partir del `Dockerfile` local y utiliza la red externa compartida `shared-net`.

## Multi-Object Tracking

El sistema implementa seguimiento multiobjeto mediante trackers especializados:

```text
TrackerManager
├── PlayerTracker
├── BallTracker
└── GoalTracker
```

Los trackers son coordinados por `TrackerManager`, permitiendo procesar independientemente jugadores, balón y arcos.

## Patrones y técnicas implementadas

### Orchestrator
`Orchestrator` coordina las diferentes etapas del pipeline de análisis.

### Strategy
`AnalysisStepHandler` define un contrato común para los diferentes pasos del análisis, permitiendo intercambiar sus implementaciones.

### Repository
El acceso a datos se encapsula mediante repositorios como `TaskRepository` y `PlayerRepository`.

### Lifespan
FastAPI utiliza `asynccontextmanager` para gestionar el ciclo de vida de la aplicación, incluyendo la inicialización de base de datos, directorios, modelos y recursos de observabilidad.

### Procesamiento distribuido
PySpark se utiliza para el postprocesamiento de eventos y datos generados durante el análisis.

### Observabilidad
Logfire proporciona instrumentación y trazabilidad de la aplicación y del servicio FastAPI.

### Configuración
Dynaconf gestiona la configuración externa mediante variables de entorno y archivos de configuración.

## Repositorio
<a href="https://github.com/UDLAIA-STATS/SoccerAnalysis_AI">SoccerAnalysis_AI — GitHub</a>

## Referencias
* <a href="https://fastapi.tiangolo.com/">FastAPI</a>
* <a href="https://docs.ultralytics.com/">Ultralytics</a>
* <a href="https://supervision.roboflow.com/">Supervision</a>
* <a href="https://spark.apache.org/docs/latest/api/python/">PySpark</a>
* <a href="https://www.dynaconf.com/">Dynaconf</a>
* <a href="https://sqlmodel.tiangolo.com/">SQLModel</a>
* <a href="https://opencv.org/">OpenCV</a>
* <a href="https://github.com/JaidedAI/EasyOCR">EasyOCR</a>
