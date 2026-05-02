# EvidenciaMed — Backend

FastAPI backend para análisis de papers científicos con IA.

## Variables de entorno (configurar en Render Dashboard)

| Variable | Descripción |
|---|---|
| `ANTHROPIC_API_KEY` | Tu key de Anthropic (sk-ant-...) |
| `EVIDENCIAMED_API_KEY` | Token secreto que usará el frontend (inventas uno, ej: `em-prod-abc123xyz`) |
| `ALLOWED_ORIGINS` | URL del frontend en Vercel, separadas por coma |

## Endpoints

```
GET  /health              → Status check
POST /analyze/doi         → Analiza por DOI (busca en CrossRef + Unpaywall)
POST /analyze/text        → Analiza texto libre
```

### Autenticación
Todos los endpoints POST requieren header:
```
X-API-Key: TU_EVIDENCIAMED_API_KEY
```

### Ejemplo DOI
```bash
curl -X POST https://tu-backend.onrender.com/analyze/doi \
  -H "Content-Type: application/json" \
  -H "X-API-Key: em-prod-abc123xyz" \
  -d '{"doi": "10.1056/NEJMoa1302413"}'
```

## Deploy en Render

1. Push este repo a GitHub
2. En Render → New Web Service → conecta el repo
3. Runtime: Python 3
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Agregar las 3 variables de entorno
7. Deploy

## Lógica de resolución de DOI

1. Consulta **CrossRef** → obtiene metadatos (título, autores, abstract)
2. Consulta **Unpaywall** → busca PDF open access
3. Si hay PDF → extrae texto completo con PyMuPDF → Claude analiza texto completo
4. Si no hay PDF → Claude analiza solo el abstract
5. Si no hay abstract → devuelve 422 (artículo de pago sin OA)
