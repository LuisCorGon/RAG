from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Asegúrate de tener el retriever bien importado

def detectar_filtros(pregunta: str) -> dict:
    pregunta = pregunta.lower()
    filtros = {}

    if any(palabra in pregunta for palabra in ["urgente", "urgencia", "inmediato"]):
        filtros["urgencia"] = "urgente"
    elif "pendiente" in pregunta:
        filtros["estado"] = "pendiente"

    if any(p in pregunta for p in ["compras", "comprar", "tienda", "regalo"]):
        filtros["categoria"] = "compras"

    if any(p in pregunta for p in ["familia", "hermano", "hermana", "madre", "padre", "tío", "abuelo"]):
        filtros["etiquetas"] = {"$contains": "familia"}

    if any(p in pregunta for p in ["salud", "médico", "pastillas", "aspirina", "revisión médica"]):
        filtros["categoria"] = "salud"

    if any(p in pregunta for p in ["trabajo", "gestión", "documentos", "informe"]):
        filtros["categoria"] = "trabajo"

    if any(p in pregunta for p in ["vehículo", "vehiculo", "coche", "gasolina", "itv"]):
        filtros["etiquetas"] = {"$contains": "vehiculo"}

    if "reunión" in pregunta or "reuniones" in pregunta:
        filtros["tipo"] = "reunión"

    return filtros



# Cargar modelo LLaMA 3 local (debes tenerlo activo con Ollama)
model = OllamaLLM(model="llama3.2")

# Prompt adaptado al contexto de tareas
template = """
Eres un asistente experto en organización personal. Ayudas a una persona a gestionar sus tareas.

Historial de conversación:
{historial}

Estas son algunas tareas relevantes de su agenda:
{tareas}

Tu respuesta debe ser clara y visualmente organizada. Usa listas con viñetas o numeración. Si hay tareas urgentes, indícalo con el emoji ⚠️. Si tienen fecha, usa 📅. Para tareas relacionadas con salud, usa 🏥, y con vehículos 🚗.

No inventes nada. Usa solo la información de la lista anterior.

Pregunta: {question}
"""

def buscar_tareas(question, filtros=None, k=20):
    """
    Recupera tareas relevantes usando búsqueda vectorial.
    Si se pasan filtros (urgencia, categoría, etc.), los aplica como filtros exactos por metadatos.
    """
    if filtros:
        # print(f"🎯 Filtro aplicado: {filtros}")
        docs = retriever.vectorstore.similarity_search(
            query=question,
            k=k,
            filter=filtros
        )
    else:
        docs = retriever.invoke(question)

    return docs


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

historial = []

# Bucle de interacción
while True:
    print("\n\n-------------------------------")
    question = input("Pregunta ('q' para salir): ")
    print("\n\n")
    if question.lower() == "q":
        break

    filtros = detectar_filtros(question)
    # print(f"\nFiltros aplicados: {filtros}")
    # Usar retriever con filtros dinámicos si hay alguno
    if filtros:
        docs = buscar_tareas(question, filtros=filtros, k=30)
    else:
        docs = retriever.invoke(question)
    
    # Extraer el contenido textual de los documentos
    tareas_relevantes = "\n".join([doc.page_content for doc in docs])

    texto_historial = "\n".join(historial[-5:])

    # print("\nTareas recuperadas:\n", tareas_relevantes)

    result = chain.invoke({
        "tareas": tareas_relevantes,
        "question": question,
        "historial": texto_historial
    })

    print(result)

    historial.append(f"Usuario: {question}")
    historial.append(f"Asistente: {result}")
