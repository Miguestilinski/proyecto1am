# ChatBot Hospital El Pino - Fase 2

Asistente de gestión y clasificación de camas del Hospital El Pino, utilizando LLM local con arquitectura RAG.

## Requisitos

- Python 3.10+
- Node.js 18+
- Ollama instalado y ejecutándose

## Estructura del Proyecto

```
ChatBoot/
├── backend/
│   ├── src/
│   │   ├── data_loader.py    # Carga y procesa datasets
│   │   ├── llm_setup.py      # Configuración RAG + LLM
│   │   └── main.py           # API FastAPI
│   ├── data/                 # ChromaDB (se crea automáticamente)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── layouts/
│   │   └── pages/
│   ├── astro.config.mjs
│   └── package.json
└── *.xlsx, *.csv             # Datasets clínicos
```

## Instalación y Ejecución

### 1. Ollama (LLM Local)

```bash
# Instalar Ollama (Windows/Mac/Linux)
# Descargar desde: https://ollama.com

# Descargar modelo Llama 3.2
ollama pull llama3.2:latest

# Iniciar servicio
ollama serve
```

### 2. Backend (Python + FastAPI)

```bash
# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
cd backend
pip install -r requirements.txt

# Ejecutar una vez para crear la base de datos vectorial
python -c "from src.data_loader import initialize_database; initialize_database()"

# Iniciar API
python -m src.main
# La API estará en http://localhost:8000
```

### 3. Frontend (Astro + Tailwind)

```bash
cd frontend
npm install
npm run dev
# El frontend estará en http://localhost:4321
```

## Endpoints de la API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Estado de la API |
| `/health` | GET | Verificación de salud |
| `/chat` | POST | Enviar mensaje al chatbot |
| `/context/{query}` | GET | Ver contexto recuperado |

## Ejemplo de Uso

```bash
# Consultar la API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Paciente de 45 años, diagnóstico J69.0"}'
```

## Preguntas de Prueba (Validación)

1. Tengo un paciente varón de 65 años con el código de diagnóstico J69.0. ¿Necesita una cama de hospitalización general o quirúrgica?

2. ¿Cuál es la probabilidad de que un paciente con el diagnóstico E11.9 necesite un quirófano?

3. ¿Puede clasificar a una paciente de 30 años con el código CIE-10 Z39.0?

4. Según la información del paciente, ¿cuál es el código DRG asignado?

5. Explique el motivo de clasificar a este paciente como de Clase 1 (Quirúrgico).

6. ¿Faltan variables demográficas o diagnósticas que sean necesarias para predecir con precisión la necesidad de cama para este paciente?

7. ¿Cuál es el procedimiento quirúrgico más frecuente asociado al código de diagnóstico I10 en nuestra base de datos?

8. «Si añado el código de procedimiento 96.71 (Ventilación mecánica continua) a este paciente, ¿cambia su clasificación?»

9. «¿Puede resumir los recursos clínicos previstos necesarios para un paciente clasificado como Clase 0?»

10. «Prueba de advertencia: ¿Esta entrada está por debajo del umbral de confianza del 40 % para la necesidad de cirugía?»

## Tecnologías Utilizadas

- **Backend**: FastAPI, LangChain, ChromaDB, Ollama (Llama 3.2)
- **Frontend**: Astro, Tailwind CSS
- **Datos**: ICD-10, ICD-9, IR-GRD V3.1, Registros Clínicos El Pino