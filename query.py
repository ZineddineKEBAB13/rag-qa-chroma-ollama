from sentence_transformers import SentenceTransformer
import chromadb
import requests , json
import tkinter as tk
from tkinter import scrolledtext



#query = "Why are energy storage technologies like batteries and hydrogen important for renewable energy?"
model = SentenceTransformer("sentence-transformers/msmarco-bert-base-dot-v5")
chroma_client = chromadb.PersistentClient(path="./db_ai_search")
collection = chroma_client.get_or_create_collection(name = "AI_and_Climate_docs")



def generate_answer():
    
    query = query_entry.get("1.0",tk.END)
    query_embedding = model.encode([query])[0]
    
    results = collection.query(query_embeddings = [query_embedding.tolist()],n_results = 3 , include = ["documents","metadatas","distances"])
    
    
    context = "\n\n".join(results["documents"][0])
    
    prompt = f"""
    You are a concise AI assistant.
    Use only the information provided in the "Context" section to answer.
    If the information is insufficient or unrelated to the question, reply exactly:
    i don't know, sorry, i don't have enough information to answer this question
    
    Do NOT add any introduction like "The answer is", "According to", or anything else.
    Give ONLY the answer text itself — no extra words.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer (direct, short, no extra words):
    """

    
    url = "http://localhost:11434/api/generate"
    data = {"model" :"gemma:2b" , "prompt" : prompt , "stream":False}
    response  = requests.post(url,json = data)
    result = json.loads(response.text)
    print(result["response"])
    
    answer_box.insert("1.0",result["response"])

def reset_fields():
    query_entry.delete("1.0",tk.END)
    answer_box.delete("1.0",tk.END)
    


######   le graphique 
root = tk.Tk()
root.title(" 🧠 🧠 RAG ASSISTANT 🧠 🧠")

tk.Label(root,text = "enter your question :",font = ("Arial", 12)).pack(pady = 5)
query_entry = scrolledtext.ScrolledText(root, width=70, height=5, font=("Arial", 11))
query_entry.pack(pady=5)
tk.Button(root,text = "Generate Answer" ,command = generate_answer, bg = "#4CAF50" , fg  = "white").pack(pady = 5)
tk.Button(root,text = "reset" , command = reset_fields, bg="#f44336",fg = "white").pack(pady = 5)
tk.Label(root, text = "Answer : ", font=("Arial", 12)).pack(pady = 5)
answer_box = scrolledtext.ScrolledText(root, width=70, height=12, font=("Arial", 11))
answer_box.pack(pady=5)

root.mainloop()






















