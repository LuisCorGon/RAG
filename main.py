from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = OllamaLLM(model="llama3.2")

template = """
Eres un asistente experto en organización personal. Ayudas a una persona a gestionar sus tareas.

Historial de conversación:
{historial}

Estas son algunas tareas relevantes de su agenda:
{tareas}

Tu respuesta debe ser clara y visualmente organizada. Usa listas con viñetas o numeración. Si hay tareas urgentes, indícalo con ⚠️. Si tienen fecha, usa 📅. Para salud 🏥, vehículo 🚗, etc.

Pregunta: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

historial = []

def detectar_filtros(pregunta: str) -> dict:
    p = pregunta.lower()
    f = {}
    if "urgente" in p: f["urgencia"] = "urgente"
    if "comprar" in p or "compra" in p: f["categoria"] = "compras"
    if "familia" in p or "hermana" in p or "madre" in p: f["etiquetas"] = {"$contains": "familia"}
    if "salud" in p: f["categoria"] = "salud"
    if "trabajo" in p: f["categoria"] = "trabajo"
    if "vehículo" in p or "coche" in p: f["etiquetas"] = {"$contains": "vehiculo"}
    if "reunión" in p: f["tipo"] = "reunión"
    return f

def buscar_tareas(question, filtros=None, k=30):
    if filtros:
        return retriever.vectorstore.similarity_search(question, k=k, filter=filtros)
    else:
        return retriever.invoke(question)

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "history": historial})

@app.post("/", response_class=HTMLResponse)
async def post_chat(request: Request, question: str = Form(...)):
    filtros = detectar_filtros(question)
    docs = buscar_tareas(question, filtros=filtros)
    tareas_relevantes = "\n".join([doc.page_content for doc in docs])
    historial_txt = "\n".join(historial[-6:])

    result = chain.invoke({
        "tareas": tareas_relevantes,
        "question": question,
        "historial": historial_txt
    })

    historial.append(f"<b>Usuario:</b> {question}")
    respuesta_html = result.replace('\n', '<br>')
    historial.append(f"<b>Asistente:</b> {respuesta_html}")


    return templates.TemplateResponse("index.html", {"request": request, "history": historial})
