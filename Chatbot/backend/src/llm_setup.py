import chromadb
import requests
from pathlib import Path
import re
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent
CSV_PATH = DATA_DIR / "dataset_elpino.csv"

df_clinical = pd.read_csv(CSV_PATH, sep=";", on_bad_lines="skip")

DIAG_MAIN_COL = "Diag 01 Principal (cod+des)"
GRD_COL = "GRD"
AGE_COL = "Edad en años"
SEX_COL = "Sexo (Desc)"

DIAG_COLS = [c for c in df_clinical.columns if c.startswith("Diag")]
PROC_COLS = [c for c in df_clinical.columns if c.startswith("Proced")]
PROC_MAIN_COLS = [c for c in PROC_COLS if "01 Principal" in c]

LOW_CONFIDENCE_THRESHOLD = 40.0
MIN_CASES_FOR_STRONG_EVIDENCE = 5

SYSTEM_PROMPT = """Eres el Especialista en Gestión Clínica del Hospital El Pino. Tu función es clasificar pacientes y entregar estadísticas de ocupación de camas basadas exclusivamente en los datos proporcionados.

### INSTRUCCIONES DE OPERACIÓN:
- PRIORIDAD DE DATOS: Usa siempre los números de 'ESTADÍSTICAS DEL HOSPITAL'. Son la verdad absoluta del recinto.
- NO CAMBIES LOS NÚMEROS: No recalcules, no inviertas y no contradigas conteos, probabilidades, confianza, advertencias ni GRD indicados en ESTADÍSTICAS DEL HOSPITAL.
- CLASIFICACIÓN DE PACIENTES: Clase 0 = Médica/Hospitalización. Clase 1 = Quirúrgica/Pabellón. Si la probabilidad quirúrgica es >50%, clasifica como Clase 1; si es <=50%, clasifica como Clase 0.
- DIAGNÓSTICO PRINCIPAL VS ASOCIADO: Para clasificar un paciente usa la base por diagnóstico principal. Para preguntas de asociación, por ejemplo procedimientos frecuentes asociados a un diagnóstico, puedes usar cualquier diagnóstico asociado.
- NO ARRASTRES CÓDIGOS: Si la consulta actual contiene un nuevo código CIE-10, ignora códigos de preguntas anteriores para el cálculo.
- CONTEXTO CONVERSACIONAL: Si la consulta actual no trae código CIE-10 pero pregunta por "este paciente", usa el último paciente válido del historial informado en ESTADÍSTICAS DEL HOSPITAL.
- RAG COMO APOYO: Usa REGISTROS DE REFERENCIA RAG solo como apoyo descriptivo. Si contradice las estadísticas, ignóralo. No menciones diagnósticos, GRD o procedimientos del RAG que no pertenezcan al código consultado.
- NO INVENTES DATOS: Si no hay registros históricos suficientes, dilo explícitamente.
- PROCEDIMIENTOS: Si aparece un código de procedimiento CIE-9, evalúa si cambia la clasificación respecto al diagnóstico solo.
- DRG: Si se solicita código DRG, usa el GRD histórico más frecuente informado en las estadísticas. Aclara que es una estimación histórica, no una asignación oficial por grouper.
- CONFIANZA: Respeta literalmente el estado de confianza. Si dice BAJA o si es menor a 40%, advierte que está bajo el umbral del 40%.
- EXPLICABILIDAD: Explica la clasificación usando conteo de casos, probabilidad quirúrgica, distribución Clase 0/Clase 1, procedimiento y GRD más frecuente cuando estén disponibles.
- CONSULTAS GENERALES: Si se pregunta por recursos de Clase 0 o Clase 1, responde usando la sección de recursos clínicos previstos.

### CONSULTA ACTUAL:
{question}

### ESTADÍSTICAS DEL HOSPITAL:
{stats_context}

### REGISTROS DE REFERENCIA RAG:
{context}

### INTERACCIÓN PREVIA:
{history}

Responde de forma técnica, directa y sin mencionar nombres de reglas internas."""


def _norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _contains_code(series: pd.Series, code: str) -> pd.Series:
    """Busca un código exacto dentro de columnas tipo 'J69.0 - descripción'."""
    pattern = rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])"
    return series.astype(str).str.contains(pattern, case=False, regex=True, na=False)


def _is_surgical(series: pd.Series) -> pd.Series:
    """En esta base, los GRD quirúrgicos se identifican por el token PH."""
    return series.astype(str).str.contains(r"\bPH\b", case=False, regex=True, na=False)


def _extract_patient_data(query: str) -> dict:
    q = query or ""
    q_upper = q.upper()
    q_lower = q.lower()

    cie10_codes = sorted(set(re.findall(r"\b[A-Z]\d{2}(?:\.\d+)?\b", q_upper)))
    procedure_codes = sorted(set(re.findall(r"\b\d{2}\.\d{1,2}\b", q_upper)))

    age = None
    age_patterns = [
        r"(\d{1,3})\s*años",
        r"edad\s*(?:de)?\s*(\d{1,3})",
        r"paciente\s*(?:de)?\s*(\d{1,3})",
    ]
    for pattern in age_patterns:
        match = re.search(pattern, q_lower)
        if match:
            candidate = int(match.group(1))
            if 0 <= candidate <= 120:
                age = candidate
                break

    sex = None
    if any(word in q_lower for word in ["varón", "hombre", "masculino", "sexo masculino"]):
        sex = "Hombre"
    elif any(word in q_lower for word in ["mujer", "femenino", "sexo femenino", "paciente femenina", "paciente mujer", "una paciente", "la paciente"]):
        sex = "Mujer"

    return {
        "cie10_codes": cie10_codes,
        "procedure_codes": procedure_codes,
        "age": age,
        "sex": sex,
        "raw_query": q,
    }


def _is_classification_query(query: str) -> bool:
    q = (query or "").lower()
    return any(word in q for word in [
        "clasificar",
        "clasifica",
        "clasificación",
        "necesita una cama",
        "cama de hospitalización",
        "hospitalización",
        "cirugía",
        "quirófano",
        "quirurg",
        "clase 0",
        "clase 1",
        "paciente",
        "cambia su clasificación",
        "cambia la clasificación",
    ])


def _is_association_query(query: str) -> bool:
    q = (query or "").lower()
    return any(word in q for word in [
        "asociado",
        "asociada",
        "más frecuente",
        "frecuente asociado",
        "procedimiento",
        "recursos clínicos",
        "recursos previstos",
        "cualquier diagnóstico",
        "todos los diagnósticos",
    ])


def _asks_for_drg(query: str) -> bool:
    q = (query or "").lower()
    return "drg" in q or "grd" in q


def _asks_for_resources(query: str) -> bool:
    q = (query or "").lower()
    return "recursos" in q or "clase 0" in q or "clase 1" in q


def _filter_by_diagnosis(code: str, use_all_diagnoses: bool = False) -> pd.DataFrame:
    cols = DIAG_COLS if use_all_diagnoses else [DIAG_MAIN_COL]
    mask = pd.Series(False, index=df_clinical.index)
    for col in cols:
        if col in df_clinical.columns:
            mask |= _contains_code(df_clinical[col], code)
    return df_clinical[mask].copy()


def _apply_demographic_filters(subset: pd.DataFrame, age=None, sex=None, age_window: int = 10) -> pd.DataFrame:
    filtered = subset.copy()
    if sex and SEX_COL in filtered.columns:
        filtered = filtered[filtered[SEX_COL].astype(str).str.lower().eq(sex.lower())]

    if age is not None and AGE_COL in filtered.columns:
        ages = pd.to_numeric(filtered[AGE_COL], errors="coerce")
        filtered = filtered[(ages >= age - age_window) & (ages <= age + age_window)]

    return filtered


def _apply_procedure_filters(subset: pd.DataFrame, procedure_codes: list[str]) -> pd.DataFrame:
    filtered = subset.copy()
    for proc_code in procedure_codes:
        proc_mask = pd.Series(False, index=filtered.index)
        for col in PROC_COLS:
            if col in filtered.columns:
                proc_mask |= _contains_code(filtered[col], proc_code)
        filtered = filtered[proc_mask]
    return filtered


def _safe_mode(series: pd.Series) -> str:
    clean = series.dropna().astype(str).map(str.strip)
    clean = clean[clean != ""]
    if clean.empty:
        return "No disponible"
    return clean.mode().iloc[0]


def _top_values_from_columns(subset: pd.DataFrame, cols: list[str], top_n: int = 5) -> list[tuple[str, int]]:
    values = []
    for col in cols:
        if col in subset.columns:
            values.extend(
                subset[col]
                .dropna()
                .astype(str)
                .map(str.strip)
                .loc[lambda s: s != ""]
                .tolist()
            )
    if not values:
        return []
    counts = pd.Series(values).value_counts().head(top_n)
    return [(idx, int(count)) for idx, count in counts.items()]


def _summarize_subset(label: str, subset: pd.DataFrame) -> dict:
    total = len(subset)
    if total == 0:
        return {
            "label": label,
            "total": 0,
            "surgical_count": 0,
            "medical_count": 0,
            "prob_surgical": None,
            "predicted_class": None,
            "confidence": None,
            "low_confidence": True,
            "most_common_drg": "No disponible",
            "low_sample": True,
        }

    surgical_mask = _is_surgical(subset[GRD_COL])
    surgical_count = int(surgical_mask.sum())
    medical_count = int(total - surgical_count)
    prob_surgical = round((surgical_count / total) * 100, 1)
    predicted_class = 1 if prob_surgical > 50 else 0
    confidence = round(abs(prob_surgical - 50) * 2, 1)

    return {
        "label": label,
        "total": total,
        "surgical_count": surgical_count,
        "medical_count": medical_count,
        "prob_surgical": prob_surgical,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
        "most_common_drg": _safe_mode(subset[GRD_COL]),
        "low_sample": total < MIN_CASES_FOR_STRONG_EVIDENCE,
    }


def _format_summary(summary: dict) -> str:
    if summary["total"] == 0:
        return f"{summary['label']}: 0 casos históricos. No permite calcular probabilidad confiable."

    clase = "Clase 1 (Quirúrgica/Pabellón)" if summary["predicted_class"] == 1 else "Clase 0 (Médica/Hospitalización)"
    confidence_status = "BAJA" if summary["low_confidence"] else "ACEPTABLE"
    warning = (
        f"ADVERTENCIA: está bajo el umbral de confianza del {LOW_CONFIDENCE_THRESHOLD:.0f}%."
        if summary["low_confidence"]
        else f"No está bajo el umbral de confianza del {LOW_CONFIDENCE_THRESHOLD:.0f}%."
    )
    sample_warning = (
        f" ADVERTENCIA: muestra pequeña (<{MIN_CASES_FOR_STRONG_EVIDENCE} casos), interpretar con cautela."
        if summary["low_sample"]
        else ""
    )
    return (
        f"{summary['label']}: Total {summary['total']} casos. "
        f"Distribución: {summary['surgical_count']} casos Clase 1 (Quirúrgica), "
        f"{summary['medical_count']} casos Clase 0 (Médica). "
        f"Probabilidad quirúrgica: {summary['prob_surgical']}%. "
        f"Clasificación sugerida: {clase}. "
        f"Confianza estadística aproximada: {summary['confidence']}%. "
        f"Estado de confianza: {confidence_status}. "
        f"{warning}{sample_warning} "
        f"GRD histórico más frecuente: {summary['most_common_drg']}."
    )


def _resources_context() -> str:
    return (
        "Recursos clínicos previstos para Clase 0 (Médica/Hospitalización): cama de hospitalización médica, "
        "evaluación médica, enfermería, monitorización básica, medicamentos, laboratorio e imagenología según diagnóstico. "
        "No se espera uso inicial de quirófano/pabellón.\n"
        "Recursos clínicos previstos para Clase 1 (Quirúrgica/Pabellón): cama quirúrgica, evaluación quirúrgica, "
        "pabellón/quirófano, anestesia, recuperación postoperatoria, insumos quirúrgicos y eventual cama crítica según severidad."
    )


def _missing_variables_text(patient: dict, query: str) -> str:
    missing = []
    if patient.get("age") is None:
        missing.append("edad")
    if patient.get("sex") is None:
        missing.append("sexo")
    q = (query or "").lower()
    if not patient.get("procedure_codes") and any(word in q for word in ["precisión", "precisa", "cambia", "quirófano", "quirurg", "pabellón"]):
        missing.append("procedimientos CIE-9")

    if missing:
        return "Variables que podrían faltar para mejorar precisión: " + ", ".join(missing) + "."
    return "Variables mínimas presentes para clasificación básica: edad, sexo y diagnóstico principal."


def _build_code_stats(code: str, patient: dict, query: str) -> str:
    output = []
    procedure_codes = patient.get("procedure_codes", [])
    age = patient.get("age")
    sex = patient.get("sex")
    is_classification = _is_classification_query(query)
    is_association = _is_association_query(query)

    primary_subset = _filter_by_diagnosis(code, use_all_diagnoses=False)
    all_diag_subset = _filter_by_diagnosis(code, use_all_diagnoses=True)

    output.append(f"Código CIE-10 detectado: {code}.")

    primary_summary = _summarize_subset(f"Base por diagnóstico principal {code}", primary_subset)
    output.append(_format_summary(primary_summary))

    # En clasificación de pacientes, el diagnóstico asociado es solo referencia, no base de decisión.
    if len(all_diag_subset) != len(primary_subset):
        associated_summary = _summarize_subset(f"Referencia por cualquier diagnóstico asociado a {code} — NO usar como clasificación principal", all_diag_subset)
        output.append(_format_summary(associated_summary))

    if is_classification and primary_subset.empty:
        if not all_diag_subset.empty:
            output.append(
                f"DECISIÓN BLOQUEADA: Para clasificar pacientes se prioriza el diagnóstico principal. "
                f"No hay registros históricos con {code} como diagnóstico principal. "
                f"El código aparece como diagnóstico asociado en {len(all_diag_subset)} registros, pero eso no permite clasificarlo de forma concluyente como Clase 0/Clase 1."
            )
        else:
            output.append(
                f"DECISIÓN BLOQUEADA: No hay registros históricos para {code}. No se recomienda entregar clasificación concluyente."
            )
        output.append(_missing_variables_text(patient, query))
        return "\n".join(output)

    most_specific_subset = primary_subset.copy()

    if sex or age is not None:
        demo_subset = _apply_demographic_filters(most_specific_subset, age=age, sex=sex)
        if not demo_subset.empty:
            most_specific_subset = demo_subset
            demo_label = f"Subconjunto demográfico para {code}"
            if sex:
                demo_label += f", sexo={sex}"
            if age is not None:
                demo_label += f", edad={age}±10 años"
            output.append(_format_summary(_summarize_subset(demo_label, most_specific_subset)))
        else:
            output.append(
                f"Subconjunto demográfico para {code}: 0 casos con sexo={sex or 'no especificado'} "
                f"y edad={str(age) + '±10 años' if age is not None else 'no especificada'}. "
                "Se mantiene la estadística por diagnóstico principal como respaldo."
            )

    if procedure_codes:
        proc_subset = _apply_procedure_filters(most_specific_subset, procedure_codes)
        if not proc_subset.empty:
            most_specific_subset = proc_subset
            output.append(_format_summary(_summarize_subset(f"Subconjunto diagnóstico principal + procedimiento(s) {', '.join(procedure_codes)}", proc_subset)))
        else:
            fallback_proc_subset = _apply_procedure_filters(primary_subset, procedure_codes)
            if not fallback_proc_subset.empty:
                most_specific_subset = fallback_proc_subset
                output.append(
                    "No hubo casos con diagnóstico + demografía + procedimiento; se usa diagnóstico principal + procedimiento sin filtro demográfico."
                )
                output.append(_format_summary(_summarize_subset(f"Subconjunto diagnóstico principal + procedimiento(s) {', '.join(procedure_codes)}", fallback_proc_subset)))
            else:
                output.append(
                    f"No hay registros históricos para {code} como diagnóstico principal combinados con procedimiento(s) {', '.join(procedure_codes)}."
                )

    # Procedimientos frecuentes: para asociación se permite cualquier diagnóstico; para clasificación se deja como referencia secundaria.
    proc_base = all_diag_subset if is_association else primary_subset
    surgical_proc_base = proc_base[_is_surgical(proc_base[GRD_COL])] if not proc_base.empty else proc_base
    proc_cols_for_top = PROC_MAIN_COLS if PROC_MAIN_COLS else PROC_COLS
    top_procs = _top_values_from_columns(surgical_proc_base, proc_cols_for_top, top_n=5)
    if top_procs:
        source_note = "cualquier diagnóstico asociado" if is_association else "diagnóstico principal"
        output.append(f"Procedimientos principales quirúrgicos más frecuentes asociados a {code} usando {source_note}:")
        for proc, count in top_procs:
            output.append(f"- {proc}: {count} casos")
    else:
        output.append(f"No hay procedimientos quirúrgicos principales frecuentes disponibles para {code}.")

    top_drgs = _top_values_from_columns(most_specific_subset, [GRD_COL], top_n=3)
    if top_drgs:
        output.append(f"GRD más frecuentes para el subconjunto más específico de {code}:")
        for grd, count in top_drgs:
            output.append(f"- {grd}: {count} casos")

    # Decisión final calculada en Python para evitar que el LLM la invente o invierta.
    final_summary = _summarize_subset("DECISIÓN FINAL basada en el subconjunto más específico disponible", most_specific_subset)
    if final_summary["total"] > 0:
        output.append(_format_summary(final_summary))
        if _asks_for_drg(query):
            output.append(
                "DRG/GRD estimado: "
                f"{final_summary['most_common_drg']}. Es una estimación histórica por frecuencia, no una asignación oficial por grouper."
            )

    output.append(_missing_variables_text(patient, query))
    return "\n".join(output)


def get_real_stats(patient_data_or_codes):
    """
    Calcula estadísticas reales desde dataset_elpino.csv.
    Acepta el formato nuevo dict o el formato antiguo: set/list de códigos.
    """
    if isinstance(patient_data_or_codes, dict):
        patient = patient_data_or_codes
    else:
        patient = {
            "cie10_codes": sorted(patient_data_or_codes) if patient_data_or_codes else [],
            "procedure_codes": [],
            "age": None,
            "sex": None,
            "raw_query": "",
        }

    codes = patient.get("cie10_codes", [])
    query = patient.get("raw_query", "")

    if not codes:
        output = ["No hay un código CIE-10 específico en la consulta actual para generar estadísticas de probabilidad."]
        if _asks_for_resources(query):
            output.append(_resources_context())
        else:
            output.append(_resources_context())
        return "\n".join(output)

    output = []
    for code in codes:
        output.append(_build_code_stats(code, patient, query))

    output.append(_resources_context())
    return "\n".join(output)


def get_retriever():
    client = chromadb.PersistentClient(path=str(DATA_DIR / "backend" / "data" / "chroma_db"))
    return client.get_collection("hospital_elpino")


def _filter_rag_documents(docs: list[str], codes: set[str], procedure_codes: set[str]) -> list[str]:
    """
    Reduce contaminación del RAG: si hay códigos exactos, conserva documentos que los contengan
    y evita que aparezcan registros clínicos semánticamente similares pero de otros diagnósticos.
    """
    if not codes and not procedure_codes:
        return docs

    wanted = {c.upper() for c in codes.union(procedure_codes)}
    filtered = []
    for doc in docs:
        upper_doc = str(doc).upper()
        if any(code in upper_doc for code in wanted):
            filtered.append(doc)
    return filtered if filtered else docs[:3]


def get_context_for_query(query, k=15):
    patient_data = _extract_patient_data(query)
    codes = set(patient_data["cie10_codes"])
    procedure_codes = set(patient_data["procedure_codes"])

    # Si hay código exacto, reducimos el RAG para evitar ruido. Si no hay código, mantenemos más contexto.
    effective_k = 5 if codes or procedure_codes else k

    collection = get_retriever()
    results = collection.query(query_texts=[query], n_results=effective_k)

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    print(f"DEBUG: Datos detectados en la pregunta: {patient_data}")

    stats_context = get_real_stats(patient_data)

    extra_docs = []
    for code in codes.union(procedure_codes):
        try:
            res = collection.get(where={"code": code})
            if res and res.get("documents"):
                extra_docs.extend(res["documents"])
        except Exception as e:
            print(f"DEBUG: No se pudo recuperar documento exacto para {code}: {e}")

    context_parts = []
    seen = set()
    for doc in (extra_docs + _filter_rag_documents(docs, codes, procedure_codes)):
        if doc not in seen:
            seen.add(doc)
            context_parts.append(doc)

    return "\n\n".join(context_parts[:effective_k]), stats_context, results


def call_ollama(prompt, model="llama3.2:latest"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "")
        return "Error: Ollama no disponible. ¿Tienes Ollama instalado y ejecutándose?"
    except Exception as e:
        return f"Error de conexión con Ollama: {str(e)}. Asegúrate de que Ollama esté ejecutándose en localhost:11434"


def _build_history_context(history: list | None) -> str:
    if not history:
        return "No hay historial previo."

    history_formatted = []
    for msg in history[-4:]:
        role_name = "USUARIO" if msg.get("role") == "user" else "ASISTENTE"
        history_formatted.append(f"{role_name}: {msg.get('content', '')}")
    return "\n".join(history_formatted)


def _resolve_search_query(question: str, history: list | None) -> str:
    """
    Usa historial solo cuando la pregunta actual no trae código CIE-10.
    Evita contaminar una consulta nueva, por ejemplo E11.9, con un paciente anterior J69.0.
    Para preguntas como "este paciente", privilegia mensajes de usuario con perfil clínico.
    Para pruebas de confianza, permite recuperar la última respuesta estructurada si contiene la entrada más reciente.
    """
    current_patient = _extract_patient_data(question)
    if current_patient["cie10_codes"] or not history:
        return question

    q_lower = (question or "").lower()
    refers_previous_patient = any(phrase in q_lower for phrase in [
        "este paciente",
        "el paciente",
        "según la información del paciente",
        "su clasificación",
        "su drg",
        "su grd",
        "esta entrada",
        "este caso",
        "la entrada",
        "añado",
        "agrego",
        "cambia",
    ])

    if not refers_previous_patient:
        return question

    is_procedure_update = any(phrase in q_lower for phrase in ["añado", "agrego", "procedimiento", "cambia"])
    is_confidence_check = _asks_for_confidence(question) if "_asks_for_confidence" in globals() else ("confianza" in q_lower or "umbral" in q_lower)

    def find_candidate(messages: list[str]) -> str | None:
        for previous in reversed(messages):
            previous_patient = _extract_patient_data(previous)
            if not previous_patient["cie10_codes"]:
                continue

            previous_lower = previous.lower()
            previous_was_association = _asks_most_frequent_procedure(previous) or _is_association_query(previous)
            previous_looks_like_patient = (
                "paciente" in previous_lower
                or previous_patient.get("age") is not None
                or previous_patient.get("sex") is not None
                or _is_classification_query(previous)
            )

            if previous_was_association and not previous_looks_like_patient:
                continue

            if is_procedure_update:
                candidate_code = previous_patient["cie10_codes"][0]
                previous_was_probability_only = "probabilidad" in previous_lower and not any(
                    key in previous_lower for key in ["clasificar", "necesita una cama", "cama de hospitalización", "cambia"]
                )
                previous_has_patient_profile = previous_patient.get("age") is not None or previous_patient.get("sex") is not None
                if _filter_by_diagnosis(candidate_code, use_all_diagnoses=False).empty:
                    continue
                if previous_was_probability_only:
                    continue
                if not (previous_looks_like_patient and previous_has_patient_profile):
                    continue

            return previous + " " + question
        return None

    user_messages = [m.get("content", "") for m in history if m.get("role") == "user" and m.get("content")]
    all_messages = [m.get("content", "") for m in history if m.get("content")]

    # Para confianza, se prioriza la entrada estructurada más reciente, que puede estar en la respuesta anterior.
    if is_confidence_check:
        candidate = find_candidate(all_messages)
        if candidate:
            return candidate

    candidate = find_candidate(user_messages)
    if candidate:
        return candidate

    candidate = find_candidate(all_messages)
    if candidate:
        return candidate

    return question


def _asks_for_confidence(query: str) -> bool:
    q = (query or "").lower()
    return "confianza" in q or "umbral" in q or "40" in q or "advertencia" in q


def _asks_missing_variables(query: str) -> bool:
    q = (query or "").lower()
    return "faltan variables" in q or "variables demográficas" in q or "variables diagnost" in q or "predecir con precisión" in q


def _asks_general_resources(query: str) -> bool:
    q = (query or "").lower()
    return "recursos" in q and not _extract_patient_data(query).get("cie10_codes")


def _asks_most_frequent_procedure(query: str) -> bool:
    q = (query or "").lower()
    return "procedimiento" in q and ("más frecuente" in q or "mas frecuente" in q or "frecuente" in q) and ("asociado" in q or "asociada" in q)


def _classification_base_summary(code: str, patient: dict) -> tuple[pd.DataFrame, dict, str]:
    """Devuelve el subconjunto decisional para clasificación: diagnóstico principal + filtros opcionales."""
    base = _filter_by_diagnosis(code, use_all_diagnoses=False)
    decision_note = "diagnóstico principal"

    if base.empty:
        return base, _summarize_subset(f"Base por diagnóstico principal {code}", base), decision_note

    demo = _apply_demographic_filters(base, age=patient.get("age"), sex=patient.get("sex"))
    if (patient.get("age") is not None or patient.get("sex")) and not demo.empty:
        base = demo
        decision_note += " + filtro demográfico disponible"

    if patient.get("procedure_codes"):
        proc = _apply_procedure_filters(base, patient.get("procedure_codes", []))
        if not proc.empty:
            base = proc
            decision_note += " + procedimiento"
        else:
            # Si la combinación con demografía queda vacía, intenta diagnóstico principal + procedimiento sin demografía.
            fallback = _apply_procedure_filters(_filter_by_diagnosis(code, use_all_diagnoses=False), patient.get("procedure_codes", []))
            if not fallback.empty:
                base = fallback
                decision_note = "diagnóstico principal + procedimiento (sin filtro demográfico por falta de casos)"

    return base, _summarize_subset(f"Base decisional por {decision_note} para {code}", base), decision_note


def _class_label(predicted_class) -> str:
    if predicted_class == 1:
        return "Clase 1 (Quirúrgica/Pabellón)"
    if predicted_class == 0:
        return "Clase 0 (Médica/Hospitalización)"
    return "Sin clasificación concluyente"


def _confidence_sentence(summary: dict) -> str:
    if summary.get("confidence") is None:
        return "No es posible calcular confianza estadística por falta de casos."
    if summary.get("low_confidence"):
        return f"Sí está bajo el umbral del {LOW_CONFIDENCE_THRESHOLD:.0f}%: confianza {summary['confidence']}%."
    return f"No está bajo el umbral del {LOW_CONFIDENCE_THRESHOLD:.0f}%: confianza {summary['confidence']}%."


def _direct_probability_answer(code: str, patient: dict) -> str:
    base = _filter_by_diagnosis(code, use_all_diagnoses=False)
    primary_summary = _summarize_subset(f"Diagnóstico principal {code}", base)
    if primary_summary["total"] == 0:
        associated = _summarize_subset(f"Cualquier diagnóstico asociado {code}", _filter_by_diagnosis(code, use_all_diagnoses=True))
        if associated["total"] == 0:
            return f"No hay registros históricos para el código {code}; no es posible calcular probabilidad quirúrgica."
        return (
            f"No hay casos con {code} como diagnóstico principal. Como referencia, aparece como diagnóstico asociado en "
            f"{associated['total']} registros, con probabilidad quirúrgica {associated['prob_surgical']}%, pero esa cifra no debe usarse para clasificar un paciente por diagnóstico principal."
        )
    return (
        f"Para {code} como diagnóstico principal hay {primary_summary['total']} casos: "
        f"{primary_summary['surgical_count']} Clase 1 y {primary_summary['medical_count']} Clase 0. "
        f"La probabilidad de requerir quirófano/pabellón es {primary_summary['prob_surgical']}%. "
        f"Clasificación sugerida por regla >50%: {_class_label(primary_summary['predicted_class'])}. "
        f"{_confidence_sentence(primary_summary)} "
        f"GRD más frecuente: {primary_summary['most_common_drg']}."
    )


def _direct_classification_answer(code: str, patient: dict, query: str) -> str:
    primary = _filter_by_diagnosis(code, use_all_diagnoses=False)
    associated = _filter_by_diagnosis(code, use_all_diagnoses=True)

    if primary.empty:
        if not associated.empty:
            associated_summary = _summarize_subset(f"Referencia asociada {code}", associated)
            return (
                f"No entregaría una clasificación Clase 0/Clase 1 concluyente para este paciente usando {code}, porque no hay registros históricos con ese código como diagnóstico principal. "
                f"El código sí aparece como diagnóstico asociado en {associated_summary['total']} registros, pero esa evidencia no debe reemplazar el diagnóstico principal para clasificar necesidad de cama. "
                f"Como referencia no decisional, la probabilidad quirúrgica asociada es {associated_summary['prob_surgical']}%. "
                f"Se recomienda solicitar/confirmar diagnóstico principal, procedimientos y antecedentes clínicos antes de clasificar."
            )
        return f"No hay registros históricos para {code}; no se recomienda entregar clasificación concluyente."

    # Si hay procedimiento, compara diagnóstico solo vs diagnóstico + procedimiento.
    only_patient = dict(patient)
    only_patient["procedure_codes"] = []
    _, base_summary, base_note = _classification_base_summary(code, only_patient)
    _, final_summary, final_note = _classification_base_summary(code, patient)

    changed = ""
    if patient.get("procedure_codes") and base_summary.get("predicted_class") is not None and final_summary.get("predicted_class") is not None:
        if base_summary["predicted_class"] != final_summary["predicted_class"]:
            changed = f" Al añadir el procedimiento, la clasificación cambia de {_class_label(base_summary['predicted_class'])} a {_class_label(final_summary['predicted_class'])}."
        else:
            changed = f" Al añadir el procedimiento, la clasificación no cambia: se mantiene en {_class_label(final_summary['predicted_class'])}."

    proc_text = ""
    if patient.get("procedure_codes"):
        proc_text = f" con procedimiento(s) {', '.join(patient.get('procedure_codes', []))}"

    return (
        f"El paciente con código {code}{proc_text} se clasifica como {_class_label(final_summary['predicted_class'])}. "
        f"Base usada: {final_note}. Total {final_summary['total']} casos; "
        f"{final_summary['surgical_count']} Clase 1 y {final_summary['medical_count']} Clase 0. "
        f"Probabilidad quirúrgica: {final_summary['prob_surgical']}%. "
        f"{_confidence_sentence(final_summary)} "
        f"GRD histórico más frecuente: {final_summary['most_common_drg']}." + changed + " " + _missing_variables_text(patient, query)
    )


def _direct_drg_answer(code: str, patient: dict) -> str:
    subset, summary, note = _classification_base_summary(code, patient)
    if summary["total"] == 0:
        associated = _summarize_subset(f"Cualquier diagnóstico asociado {code}", _filter_by_diagnosis(code, use_all_diagnoses=True))
        if associated["total"] > 0:
            return (
                f"No hay GRD estimable con {code} como diagnóstico principal. Como referencia no decisional, el GRD más frecuente cuando {code} aparece como diagnóstico asociado es {associated['most_common_drg']}. "
                "Esto no equivale a una asignación oficial de DRG para el paciente."
            )
        return f"No hay registros suficientes para estimar un DRG asociado a {code}."
    return (
        f"El GRD histórico más frecuente para el subconjunto disponible de {code} es: {summary['most_common_drg']}. "
        f"Base usada: {note}, con {summary['total']} casos. Esta es una estimación histórica por frecuencia, no una asignación oficial por grouper DRG."
    )


def _direct_explanation_answer(code: str, patient: dict, query: str) -> str:
    _, summary, note = _classification_base_summary(code, patient)
    requested_class_1 = "clase 1" in (query or "").lower() or "quirúrgico" in (query or "").lower() or "quirurgico" in (query or "").lower()

    if summary["total"] == 0:
        return f"No es posible explicar una clasificación concluyente para {code}, porque no hay registros suficientes como diagnóstico principal."

    actual = _class_label(summary["predicted_class"])
    if requested_class_1 and summary["predicted_class"] != 1:
        return (
            f"No corresponde justificarlo como Clase 1, porque la estadística disponible clasifica el caso como {actual}. "
            f"La probabilidad quirúrgica es {summary['prob_surgical']}%, con {summary['surgical_count']} casos Clase 1 y {summary['medical_count']} casos Clase 0 sobre {summary['total']} casos. "
            f"El GRD más frecuente es {summary['most_common_drg']}."
        )
    return (
        f"La clasificación como {actual} se explica porque, usando {note}, existen {summary['total']} casos históricos: "
        f"{summary['surgical_count']} Clase 1 y {summary['medical_count']} Clase 0. "
        f"La probabilidad quirúrgica calculada es {summary['prob_surgical']}%, por lo que se aplica la regla de decisión >50% para Clase 1 y <=50% para Clase 0. "
        f"GRD más frecuente: {summary['most_common_drg']}. {_confidence_sentence(summary)}"
    )


def _direct_missing_variables_answer(code: str | None, patient: dict, query: str) -> str:
    base = _missing_variables_text(patient, query)
    extras = (
        " Para aumentar precisión clínica también sería recomendable incorporar diagnósticos secundarios, procedimientos CIE-9, comorbilidades, severidad, uso de ventilación mecánica, UCI, tipo de ingreso y condición de egreso."
    )
    if code:
        _, summary, _ = _classification_base_summary(code, patient)
        if summary["total"] > 0:
            return f"{base} Para {code}, la predicción básica tiene {summary['total']} casos históricos y confianza {summary['confidence']}%." + extras
    return base + extras


def _direct_most_frequent_procedure_answer(code: str) -> str:
    all_diag_subset = _filter_by_diagnosis(code, use_all_diagnoses=True)
    if all_diag_subset.empty:
        return f"No hay registros donde {code} aparezca como diagnóstico principal o asociado; no se puede identificar procedimiento frecuente."
    surgical_subset = all_diag_subset[_is_surgical(all_diag_subset[GRD_COL])]
    proc_cols_for_top = PROC_MAIN_COLS if PROC_MAIN_COLS else PROC_COLS
    top = _top_values_from_columns(surgical_subset, proc_cols_for_top, top_n=5)
    if not top:
        return f"No se encontraron procedimientos quirúrgicos principales asociados a {code}."
    first_proc, first_count = top[0]
    detail = "; ".join([f"{proc} ({count} casos)" for proc, count in top[:5]])
    return (
        f"El procedimiento quirúrgico principal más frecuente asociado a {code}, considerando el código en cualquier diagnóstico y filtrando casos Clase 1, es {first_proc}, con {first_count} casos. "
        f"Top procedimientos: {detail}."
    )


def _direct_confidence_answer(code: str | None, patient: dict) -> str:
    if not code:
        return "No puedo evaluar el umbral de confianza porque la entrada actual no contiene ni recupera un código CIE-10 válido del paciente."
    _, summary, note = _classification_base_summary(code, patient)
    if summary["total"] == 0:
        return f"Sí debe considerarse una entrada no confiable para {code}, porque no hay casos históricos suficientes como diagnóstico principal."
    return (
        f"Para {code}, usando {note}, la confianza estadística aproximada es {summary['confidence']}%. "
        f"{_confidence_sentence(summary)} Probabilidad quirúrgica: {summary['prob_surgical']}%, con {summary['total']} casos históricos."
    )


def _direct_answer_if_validation(question: str, search_query: str) -> str | None:
    """Respuestas determinísticas para el set de validación. Evita que el LLM contradiga cálculos."""
    patient = _extract_patient_data(search_query)
    codes = patient.get("cie10_codes", [])
    code = codes[0] if codes else None
    q = (question or "").lower()

    if _asks_general_resources(question):
        if "clase 1" in q:
            return "Para un paciente Clase 1 (Quirúrgica/Pabellón), se prevén cama quirúrgica, evaluación por equipo quirúrgico, pabellón/quirófano, anestesia, recuperación postoperatoria, insumos quirúrgicos y eventual cama crítica según severidad."
        return "Para un paciente Clase 0 (Médica/Hospitalización), se prevén cama de hospitalización médica, evaluación médica, enfermería, monitorización básica, medicamentos, laboratorio e imagenología según diagnóstico. No se espera uso inicial de quirófano/pabellón."

    if not code:
        return None

    if _asks_most_frequent_procedure(question):
        return _direct_most_frequent_procedure_answer(code)

    if _asks_for_confidence(question):
        return _direct_confidence_answer(code, patient)

    if _asks_missing_variables(question):
        return _direct_missing_variables_answer(code, patient, search_query)

    if _asks_for_drg(question):
        return _direct_drg_answer(code, patient)

    if "explique" in q or "motivo" in q or "por qué" in q or "porque" in q:
        return _direct_explanation_answer(code, patient, question)

    if "probabilidad" in q or "necesite un quirófano" in q or "necesite un quirofano" in q:
        return _direct_probability_answer(code, patient)

    if _is_classification_query(question) or patient.get("procedure_codes"):
        return _direct_classification_answer(code, patient, search_query)

    return None


def answer_question(question: str, history: list = None):
    search_query = _resolve_search_query(question, history)
    history_str = _build_history_context(history)

    context, stats_context, raw_results = get_context_for_query(search_query)

    # Para preguntas de validación, la respuesta se genera de forma determinística
    # desde Python para evitar contradicciones del LLM en conteos, porcentajes o umbrales.
    direct_answer = _direct_answer_if_validation(question, search_query)
    if direct_answer:
        return {
            "answer": direct_answer,
            "context_used": f"STATS: {stats_context} | RAG: {context[:1000]}",
            "sources": raw_results.get("metadatas", [[]])[0][:5],
        }

    full_prompt = SYSTEM_PROMPT.format(
        stats_context=stats_context,
        context=context,
        question=question,
        history=history_str,
    )
    print(f"--- DEBUG PROMPT ENVIADO ---\n{full_prompt}\n---------------------------")

    response = call_ollama(full_prompt)
    return {
        "answer": response,
        "context_used": f"STATS: {stats_context} | RAG: {context[:1000]}",
        "sources": raw_results.get("metadatas", [[]])[0][:5],
    }


if __name__ == "__main__":
    tests = [
        "Tengo un paciente varón de 65 años con el código de diagnóstico J69.0. ¿Necesita una cama de hospitalización o de cirugía?",
        "¿Cuál es la probabilidad de que un paciente con el diagnóstico E11.9 necesite un quirófano?",
        "¿Puede clasificar a una paciente de 30 años con el código CIE-10 Z39.0?",
        "Si añado el código de procedimiento 96.71 a este paciente con J69.0, ¿cambia su clasificación?",
        "¿Cuál es el procedimiento quirúrgico más frecuente asociado al código de diagnóstico I10 en nuestra base de datos?",
    ]
    for test_query in tests:
        print("\n=== CONSULTA ===")
        print(test_query)
        patient = _extract_patient_data(test_query)
        print(get_real_stats(patient))
