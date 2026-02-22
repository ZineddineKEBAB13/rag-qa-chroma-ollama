# RAG Document QA (Chroma + Ollama)

Système de question-réponse sur documents basé sur RAG (Retrieval-Augmented Generation).
Le projet découpe des documents en segments, crée des embeddings, indexe les segments dans une base vectorielle Chroma, puis répond aux questions en récupérant les passages les plus pertinents avant de générer une réponse via un LLM local (Ollama).

## Fonctionnalités
- Découpage de documents en chunks avec gestion des sources
- Création d'embeddings et indexation persistante dans Chroma
- Recherche top-k des passages pertinents pour une requête
- Génération d'une réponse avec un LLM local via Ollama
- Prompt qui contraint l'utilisation du contexte récupéré pour limiter les hallucinations

## Structure du projet
- `create_chunks.py` : préparation du corpus et découpage en chunks
- `create_embeddings.py` : création des embeddings et indexation dans Chroma
- `query.py` : requêtes utilisateur, retrieval top-k et génération de réponse
- `doc1.txt`, `doc2.txt` : exemples de documents
- `chroma.sqlite3` et fichiers `*.bin` : stockage persistant de la base Chroma

## Prérequis
- Python 3.10+
- Ollama installé et lancé en local
- Un modèle disponible dans Ollama (ex: Mistral)