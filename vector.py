from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import os
import shutil

# Eliminar base de datos
# shutil.rmtree("chroma_agenda_db")

# Cargar tareas estructuradas
with open("agenda_estructurada.json", "r", encoding="utf-8") as f:
    tareas = json.load(f)

# Cargar modelo de embeddings
# embeddings = OllamaEmbeddings(model="mxbai-embed-large")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Ruta a la base de datos de vectores
db_location = "./chroma_agenda_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    for i, tarea in enumerate(tareas):
        contenido = f"{tarea['texto']}"
        document = Document(
            page_content=contenido,
            metadata={  # puedes dejar solo lo necesario o ninguno
                "fecha": tarea["fecha"],
                "tipo": tarea["tipo"],
                "urgencia": tarea["urgencia"],
                "categoria": tarea["categoría"],
                "estado": tarea["estado"],
                "etiquetas": ", ".join(tarea["etiquetas"]) if isinstance(tarea["etiquetas"], list) else str(tarea["etiquetas"]),
            },
            id=str(i)
        )

        ids.append(str(i))
        documents.append(document)

# Crear base de datos vectorial
vector_store = Chroma(
    collection_name="tareas_agenda",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Crear retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
