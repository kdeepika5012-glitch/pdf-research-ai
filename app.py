import os
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pypdf import PdfReader

import chromadb
from google import genai


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please add it to your .env file."
    )


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

UPLOAD_FOLDER = Path("uploads")
CHROMA_FOLDER = Path("chroma_db")

UPLOAD_FOLDER.mkdir(exist_ok=True)
CHROMA_FOLDER.mkdir(exist_ok=True)


# =========================================================
# GEMINI
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

GENERATION_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-2"


# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_FOLDER)
)

collection = chroma_client.get_or_create_collection(
    name="pdf_rag_collection"
)


# =========================================================
# CURRENT PDF
# =========================================================

current_document_id = None
current_pdf_name = None


# =========================================================
# CREATE EMBEDDING
# =========================================================

def create_embedding(text):

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return result.embeddings[0].values


# =========================================================
# EXTRACT PDF TEXT
# =========================================================

def extract_pdf_text(pdf_path):

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()

        if text:

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(
    pages,
    chunk_size=1200,
    overlap=200
):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# =========================================================
# CLEAR PREVIOUS DOCUMENT
# =========================================================

def clear_previous_document():

    global current_document_id
    global current_pdf_name

    try:

        existing = collection.get()

        if existing and existing.get("ids"):

            collection.delete(
                ids=existing["ids"]
            )

    except Exception as e:

        print(
            "Chroma clear error:",
            e
        )

    # Delete previous PDFs

    for file in UPLOAD_FOLDER.iterdir():

        if file.is_file():

            try:
                file.unlink()
            except Exception:
                pass

    current_document_id = None
    current_pdf_name = None


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD PDF
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    global current_document_id
    global current_pdf_name

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No PDF file was uploaded."
            }), 400

        file = request.files["file"]

        if not file.filename:

            return jsonify({
                "success": False,
                "error": "Please select a PDF file."
            }), 400

        if not file.filename.lower().endswith(".pdf"):

            return jsonify({
                "success": False,
                "error": "Only PDF files are allowed."
            }), 400

        # Clear previous PDF

        clear_previous_document()

        # Create document ID

        document_id = str(
            uuid.uuid4()
        )

        filename = (
            f"{document_id}.pdf"
        )

        pdf_path = (
            UPLOAD_FOLDER /
            filename
        )

        file.save(pdf_path)

        # Extract PDF

        pages = extract_pdf_text(
            pdf_path
        )

        if not pages:

            pdf_path.unlink(
                missing_ok=True
            )

            return jsonify({
                "success": False,
                "error": (
                    "Could not extract text from this PDF. "
                    "If it is a scanned PDF, OCR may be required."
                )
            }), 400

        # Create chunks

        chunks = create_chunks(
            pages
        )

        if not chunks:

            pdf_path.unlink(
                missing_ok=True
            )

            return jsonify({
                "success": False,
                "error": (
                    "No readable content was found "
                    "in the PDF."
                )
            }), 400

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        # Create embeddings

        for index, chunk in enumerate(
            chunks
        ):

            print(
                f"Embedding "
                f"{index + 1}/{len(chunks)}"
            )

            chunk_id = (
                f"{document_id}_{index}"
            )

            embedding = create_embedding(
                chunk["text"]
            )

            ids.append(
                chunk_id
            )

            embeddings.append(
                embedding
            )

            documents.append(
                chunk["text"]
            )

            metadatas.append({
                "document_id": document_id,
                "source": file.filename,
                "page": chunk["page"],
                "chunk": index
            })

        # Store in Chroma

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        current_document_id = (
            document_id
        )

        current_pdf_name = (
            file.filename
        )

        return jsonify({

            "success": True,

            "message":
                "PDF uploaded and indexed successfully.",

            "filename":
                file.filename,

            "pages":
                len(pages),

            "chunks":
                len(chunks)
        })

    except Exception as e:

        print(
            "UPLOAD ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# CHAT
# =========================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    global current_document_id

    try:

        # Check PDF

        if not current_document_id:

            return jsonify({
                "success": False,
                "reply": (
                    "Please upload a PDF first. "
                    "I can answer questions only "
                    "from the uploaded PDF."
                )
            }), 400

        # Get question

        data = (
            request.get_json(
                silent=True
            ) or {}
        )

        question = (
            data
            .get("message", "")
            .strip()
        )

        if not question:

            return jsonify({
                "success": False,
                "reply":
                    "Please enter a question."
            }), 400

        # Embed question

        question_embedding = (
            create_embedding(
                question
            )
        )

        # Search Chroma

        results = collection.query(

            query_embeddings=[
                question_embedding
            ],

            n_results=5,

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = (
            results
            .get("documents", [[]])[0]
        )

        metadatas = (
            results
            .get("metadatas", [[]])[0]
        )

        distances = (
            results
            .get("distances", [[]])[0]
        )

        if not documents:

            return jsonify({
                "success": True,
                "reply": (
                    "I don't know. This information "
                    "is not available in the uploaded PDF."
                )
            })

        # Relevant chunks

        context_parts = []

        relevant_chunks = []

        for doc, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            if distance <= 0.75:

                relevant_chunks.append(
                    (
                        doc,
                        metadata,
                        distance
                    )
                )

                context_parts.append(
                    f"[Page {metadata.get('page', 'unknown')}]\n"
                    f"{doc}"
                )

        # Nothing relevant

        if not relevant_chunks:

            return jsonify({
                "success": True,
                "reply": (
                    "I don't know. This information "
                    "is not available in the uploaded PDF."
                )
            })

        context = "\n\n".join(
            context_parts
        )

        # Strict RAG prompt

        prompt = f"""
You are a strict PDF-based RAG assistant.

Your ONLY source of truth is the CONTEXT extracted
from the user's uploaded PDF.

IMPORTANT RULES:

1. Answer ONLY using information explicitly
   supported by the CONTEXT.

2. Do NOT use your general knowledge.

3. Do NOT guess.

4. Do NOT invent facts.

5. Do NOT answer questions unrelated to the PDF.

6. If the answer cannot be found or reasonably
   supported by the CONTEXT, reply exactly:

I don't know. This information is not available in the uploaded PDF.

7. Keep the answer clear and concise.

8. If useful, mention the PDF page number.

USER QUESTION:
{question}

CONTEXT FROM UPLOADED PDF:

{context}
"""

        # Gemini

        response = client.models.generate_content(

            model=GENERATION_MODEL,

            contents=prompt
        )

        answer = (
            response.text.strip()
            if response.text
            else ""
        )

        if not answer:

            answer = (
                "I don't know. This information "
                "is not available in the uploaded PDF."
            )

        return jsonify({

            "success": True,

            "reply": answer
        })

    except Exception as e:

        print(
            "CHAT ERROR:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "reply":
                "Something went wrong while "
                "processing your question.",

            "error":
                str(e)
        }), 500


# =========================================================
# RESET
# =========================================================

@app.route(
    "/reset",
    methods=["POST"]
)
def reset():

    try:

        clear_previous_document()

        return jsonify({
            "success": True,
            "message":
                "PDF session cleared."
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error":
            "File is too large. Maximum size is 20 MB."
    }), 413


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )