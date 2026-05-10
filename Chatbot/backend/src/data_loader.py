import pandas as pd
import chromadb
from chromadb.config import Settings
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DB_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

def load_datasets():
    cie10 = pd.read_excel(PROJECT_ROOT / "CIE-10.xlsx")
    cie9 = pd.read_excel(PROJECT_ROOT / "CIE-9.xlsx")
    grd = pd.read_excel(PROJECT_ROOT / "IR-GRD V3.1 CON PRECIOS FONASA 2016.xlsx")
    elpino = pd.read_csv(PROJECT_ROOT / "dataset_elpino.csv", sep=';', on_bad_lines='skip')
    return cie10, cie9, grd, elpino

def extract_codes_from_diagnosis(diag_str):
    if pd.isna(diag_str):
        return []
    codes = []
    parts = str(diag_str).split(';')
    for part in parts:
        code = part.strip().split()[0] if part.strip() else ''
        if code and (code.replace('.', '').replace('-', '').isalnum()):
            codes.append(code)
    return codes

def create_rag_documents(cie10, cie9, grd, elpino):
    documents = []
    ids = []
    metadatas = []

    for idx, row in cie10.iterrows():
        doc = f"Código ICD-10: {row.get('Código', '')}. Descripción: {row.get('Descripción', '')}. Categoría: {row.get('Categoría', '')}. Sección: {row.get('Sección', '')}."
        documents.append(doc)
        ids.append(f"cie10_{idx}")
        metadatas.append({"type": "cie10", "code": str(row.get('Código', ''))})

    for idx, row in cie9.iterrows():
        doc = f"Código ICD-9: {row.get('Código', '')}. Descripción: {row.get('Descripción', '')}."
        documents.append(doc)
        ids.append(f"cie9_{idx}")
        metadatas.append({"type": "cie9", "code": str(row.get('Código', ''))})

    for idx, row in grd.iterrows():
        doc = f"GRD: {row.get('IR-GRD CÓDIGO', '')}. Nombre: {row.get('NOMBRE DEL GRUPO GRD', '')}. Peso: {row.get('Peso v31', '')}. Categoría: {row.get('CATEGORÍA DIAGNÓSTICA MAYOR CDM', '')}. Tipo: {row.get('TIPO GRD', '')}. Precio FONASA: ${row.get('Precio FONASA 2016', 0):,.0f} CLP."
        documents.append(doc)
        ids.append(f"grd_{idx}")
        metadatas.append({
            "type": "grd",
            "code": str(row.get('IR-GRD CÓDIGO', '')),
            "category": str(row.get('CATEGORÍA DIAGNÓSTICA MAYOR CDM', '')),
            "is_surgical": "PH" in str(row.get('TIPO GRD', ''))
        })

    for idx, row in elpino.iterrows():
        diag_principal = row.get('Diag 01 Principal (cod+des)', '')
        codes = extract_codes_from_diagnosis(diag_principal)
        edad = row.get('Edad en años', 'ND')
        sexo = row.get('Sexo (Desc)', 'ND')
        grd_val = row.get('GRD', '')
        
        is_surgical = 1 if 'PH' in str(grd_val) else 0
        
        doc = f"Registro clínico - Edad: {edad} años. Sexo: {sexo}. Diagnóstico principal: {diag_principal}. GRD asignado: {grd_val}. Clase: {'Quirúrgica/Pabellón' if is_surgical else 'Médica'}."
        
        documents.append(doc)
        ids.append(f"elpino_{idx}")
        metadatas.append({
            "type": "clinical_record",
            "age": str(edad),
            "sex": str(sexo),
            "grd": str(grd_val),
            "class": is_surgical
        })

    return documents, ids, metadatas

def create_vector_store(documents, ids, metadatas):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    try:
        client.delete_collection("hospital_elpino")
    except:
        pass
    
    collection = client.create_collection("hospital_elpino")
    
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metadatas)
    
    return collection

def initialize_database():
    print("Cargando datasets...")
    cie10, cie9, grd, elpino = load_datasets()
    print(f"CIE-10: {len(cie10)}, CIE-9: {len(cie9)}, GRD: {len(grd)}, El Pino: {len(elpino)}")
    
    print("Creando documentos RAG...")
    documents, ids, metadatas = create_rag_documents(cie10, cie9, grd, elpino)
    print(f"Total documentos: {len(documents)}")
    
    print("Indexando en ChromaDB...")
    collection = create_vector_store(documents, ids, metadatas)
    print("Base de datos vectorial creada exitosamente!")
    
    return collection

if __name__ == "__main__":
    initialize_database()