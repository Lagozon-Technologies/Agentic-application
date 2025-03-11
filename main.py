import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI
from base import *
from dotenv import load_dotenv
from state import session_state
load_dotenv()  # Load environment variables from .env file
from typing import Optional
import logging
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException,Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import JSONResponse
from typing import List
import os, time
import nest_asyncio
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext

from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import json
nest_asyncio.apply()
# Setup environment variables
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TITLE = os.getenv("TITLE")
# ROLE = os.getenv("ROLE").split(',')
# PAGE = os.getenv("PAGE").split(",")
ASK_QUESTION = os.getenv("ASK_QUESTION")
ASK = os.getenv("ASK")
UPLOAD_DOC = os.getenv("UPLOAD_DOC")
E_QUESTION = os.getenv("E_QUESTION")
SECTION = os.getenv("SECTION").split(",")
DOCSTORE = os.getenv("DOCSTORE").split(",")
COLLECTION = os.getenv("COLLECTION").split(",")
DATABASE = os.getenv("DATABASE").split(",")
P_QUESTION = os.getenv("P_QUESTION")
INSERT_DOCUMENT = os.getenv("INSERT_DOCUMENT")
ADD_DOC = os.getenv("ADD_DOC")
DOC_ADDED = os.getenv("DOC_ADDED")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DELETE_DOC = os.getenv("DELETE_DOC")
C_DELETE = os.getenv("C_DELETE")
api_key = os.getenv("UNSTRUCTURED_API_KEY")
api_url = os.getenv("UNSTRUCTURED_API_URL")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DOC_DELETED = os.getenv("DOC_DELETED")
N_DOC = os.getenv("N_DOC")
image = os.getenv("image")
imagess = os.getenv("imagess")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
QA_PROMPT_STR = os.getenv("QA_PROMPT_STR")
LLM_INSTRUCTION = os.getenv("LLM_INSTRUCTION")
NO_METADATA = os.getenv("NO_METADATA")
METADATA_INSTRUCTION = os.getenv("METADATA_INSTRUCTION").split(",")
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_API_KEY
Settings.llm = OpenAI(model=LLM_MODEL)
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

openai_ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL)
CSV_FILE_PATH = "record_results.csv"

app = FastAPI()

# Set up static files and templates
app.mount("/stats", StaticFiles(directory="stats"), name="stats")
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI API key and model
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
models = os.getenv('models').split(',')
databases = os.getenv('databases').split(',')
subject_areas1 = os.getenv('subject_areas1').split(',')
subject_areas2 = os.getenv('subject_areas2').split(',')
question_dropdown = os.getenv('Question_dropdown')
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)  # Adjust model as necessary
from table_details import get_table_details  # Importing the function

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):

    # Extract table names dynamically
    tables = []

    # Pass dynamically populated dropdown options to the template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models,
        "databases": databases,  # Dynamically populated database dropdown
        "section": subject_areas2,
        "tables": tables,        # Table dropdown based on database selection
        "question_dropdown": question_dropdown.split(','),  # Static questions from env
    })
    


@app.get("/authentication", response_class=HTMLResponse)
async def user_page(request: Request):
    return templates.TemplateResponse("authentication.html", {"request": request})
    

# @app.get("/user_more", response_class=HTMLResponse)
# async def user_more(request: Request):
#     return templates.TemplateResponse("user_more.html", {"request": request})
@app.get("/user_more", response_class=HTMLResponse)
async def user_more(request: Request):
    # return templates.TemplateResponse("user_more.html", {"request": request})
    tables = []

    # Pass dynamically populated dropdown options to the template
    return templates.TemplateResponse("user.html", {
        "request": request,
        "models": models,
        "databases": databases,  # Dynamically populated database dropdown
        "section": subject_areas2,
        "tables": tables,        # Table dropdown based on database selection
        "question_dropdown": question_dropdown.split(','),  # Static questions from env
    })
    


@app.get("/get_questions/")
async def get_questions(subject: str):
    """Fetch questions from the selected subject's CSV file."""
    csv_file = f"{subject}_questions.csv"
    if not os.path.exists(csv_file):
        return JSONResponse(
            content={"error": f"The file `{csv_file}` does not exist."}, status_code=404
        )

    try:
        # Read the questions from the CSV
        questions_df = pd.read_csv(csv_file)
        if "question" in questions_df.columns:
            questions = questions_df["question"].tolist()
        else:
            questions = questions_df.iloc[:, 0].tolist()
        return {"questions": questions}
    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred while reading the file: {str(e)}"}, status_code=500
        )
@app.get("/get-tables/")
async def get_tables(selected_section: str):
    # Fetch table details for the selected section
    print("now starting...")
    table_details = get_table_details(selected_section)
    # Extract table names dynamically
    print("till table details")
    tables = [line.split("Table Name:")[1].strip() for line in table_details.split("\n") if "Table Name:" in line]
    # Return tables as JSON
    return {"tables": tables}

# Table data display endpoint
# def display_table_with_styles(data, table_name, page_number, records_per_page):
#     start_index = (page_number - 1) * records_per_page
#     end_index = start_index + records_per_page
#     page_data = data.iloc[start_index:end_index]
#     styled_table = page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"') \
#         .set_table_styles(
#             [{
#                 'selector': 'th',
#                 'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]
#             },
#             {
#                 'selector': 'td',
#                 'props': [('border', '2px solid black'), ('padding', '5px')]
#             }]
#         ).to_html(escape=False)
#     return styled_table

def display_table_with_styles(data, table_name):
    
    page_data = data # Get whole table data

    styled_table = page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"') \
        .set_table_styles(
            [{
                'selector': 'th',
                'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]
            },
            {
                'selector': 'td',
                'props': [('border', '2px solid black'), ('padding', '5px')]
            }]
        ).to_html(escape=False)
    print(styled_table)
    return styled_table


# Invocation Function
def invoke_chain(question, messages, selected_model, selected_subject):
    try:
        history = ChatMessageHistory()
        for message in messages:
            if message["role"] == "user":
                history.add_user_message(message["content"])
            else:
                history.add_ai_message(message["content"])
        runner = graph.compile()
        result = runner.invoke({
            'question': question,
            'messages': history.messages,
            'selected_model': selected_model,
            'selected_subject': selected_subject
        })
        print(f"Result from runner.invoke:", result)
        print("resultttttt",result.get("messages")[-1].name)

        # result.get("intent") = classify_intent(intent)
        if result.get("SQL_Statement"):
            print(f"Result from runner.invoke:", result)
            # Return SQL-related results
            return {
                "intent": "db_query",
                "SQL_Statement": result.get("SQL_Statement"),
                "chosen_tables": result.get("chosen_tables"),
                "tables_data": result.get("tables_data"),
                "db": result.get("db")
            }



        elif result.get("messages")[-1].name == "researcher":
            print("Intent Classification: researcher (inside elif)")
            # Return search results
            return {
                "intent": "researcher",
                "search_results": result.get("messages")[-1].content  # Get the last message (search results)
            }

        elif result.get("messages")[-1].name == "intellidoc":   # Check if the last message is intellidoc
            print("Intent Classification: intellidoc (inside elif)")
            # Return document retrieval results
            return {
                "intent": "intellidoc",
                "search_results": result.get("messages")[-1].content  # Get the last message (document retrieval results)
            }
        else:
            print("Intent Classification: Unknown (inside else)")
            # Handle other intents (e.g., intellidoc)
            return {
                "intent": result.get("messages")[-1].name,
                "message": "This intent is not yet implemented."
            }

        #return result['SQL_Statement'], result['chosen_tables'], result['tables_data'], result.get('db')

    except Exception as e:
        print("Error:", e)
        custom_message = "Insufficient information to generate SQL Query."
        return custom_message, [], {}, None


@app.post("/submit")
async def submit_query(
    section: str = Form(...),
    example_question: str = Form(...),
    user_query: str = Form(...),
    page: Optional[int] = Query(1),
    records_per_page: Optional[int] = Query(5),
):
    selected_subject = section
    session_state['user_query'] = user_query

    prompt = user_query if user_query else example_question
    if 'messages' not in session_state:
        session_state['messages'] = []

    session_state['messages'].append({"role": "user", "content": prompt})

    try:
        result = invoke_chain(
            prompt, session_state['messages'], "gpt-4o-mini", selected_subject
        )
        print(f"submit_query received: {result}")

        if result["intent"] == "db_query":
            session_state['generated_query'] = result.get("SQL_Statement", "")
            session_state['chosen_tables'] = result.get("chosen_tables", [])
            session_state['tables_data'] = result.get("tables_data", {})

            tables_html = []
            for table_name, data in session_state['tables_data'].items():
                

                html_table = display_table_with_styles(data, table_name)

                tables_html.append({
                    "table_name": table_name,
                    "table_html": html_table,
                   
                })

            response_data={
                "user_query": session_state['user_query'],
                "query": session_state['generated_query'],
                "tables": tables_html
            }

   



        elif result["intent"] == "researcher":
            # If the intent is researcher, return search results
            session_state['generated_query'] = result["search_results"]

            response_data = {
                "user_query": session_state['user_query'],
                "search_results": result.get("search_results", "No results found."),
            }



        # elif result["intent"] == "intellidoc":
        #     # # Call intellidoc_tool to get the results
        #     print("result is: ",result)

        #     # Construct the response data dictionary
        #     response_data = {
        #         "user_query": session_state['user_query'],
        #         "search_results": result.get("search_results", "No results found."),

        #     }
        
        elif result["intent"] == "intellidoc":
            print("result is: ", result)

            # Extract relevant parts
            search_results = result.get("search_results", "No results found.")

            # Splitting the response into different sections
            parts = search_results.split("\n\n")

            # Identify sections: Answer, Source, FAQ
            answer = parts[0] if len(parts) > 0 else ""
            source = next((part for part in parts if part.startswith("Source:")), "")
            

            # Construct the formatted HTML response
            formatted_response = f"""
                <p>{answer}</p>
                <p><strong>{source}</strong></p>
               
            """

            # Construct the response data dictionary
            response_data = {
                "user_query": session_state['user_query'],
                "search_results": formatted_response.strip()
            }


        else:
            # Handle other intents (e.g., intellidoc)
            response_data = {
                "user_query": session_state['user_query'],
                "message": result.get("message", "This intent is not yet implemented."),
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the prompt: {str(e)}")

@app.get("/get_table_data/")
async def get_table_data(
    table_name: str = Query(...),
    page_number: int = Query(1),
    records_per_page: int = Query(10),
):
    """Fetch paginated and styled table data."""
    try:
        # Check if the requested table exists in session state
        if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
            raise HTTPException(status_code=404, detail=f"Table {table_name} data not found.")

        # Retrieve the data for the specified table
        data = session_state["tables_data"][table_name]
        total_records = len(data)
        total_pages = (total_records + records_per_page - 1) // records_per_page

        # Ensure valid page number
        if page_number < 1 or page_number > total_pages:
            raise HTTPException(status_code=400, detail="Invalid page number.")

        # Slice data for the requested page
        start_index = (page_number - 1) * records_per_page
        end_index = start_index + records_per_page
        page_data = data.iloc[start_index:end_index]

        # Style the table as HTML
        styled_table = (
            page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"')
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]},
                {'selector': 'td', 'props': [('border', '2px solid black'), ('padding', '5px')]},
            ])
            .to_html(escape=False)  # Render as HTML
        )

        return {
            "table_html": styled_table,
            "page_number": page_number,
            "total_pages": total_pages,
            "total_records": total_records,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating table data: {str(e)}")


class Session:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    def pop(self, key, default=None):
        return self.data.pop(key, default)

    def __contains__(self, item):
        return item in self.data

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def __iter__(self):
        return iter(self.data)
session = Session()

def hybrid_retrieve(query, docstore, vector_index, bm25_retriever, alpha=0.5):
    """Perform hybrid retrieval using BM25 and vector-based retrieval."""
    # Get results from BM25
    try:
        bm25_results = bm25_retriever.retrieve(query)
        # Get results from the vector store
        vector_results = vector_index.as_retriever(similarity_top_k=2).retrieve(query)
    except Exception as e:
        logging.error(e)
        return JSONResponse("Error with retriever")
    # Combine results with weighting
    combined_results = {}
    # Weight BM25 results
    for result in bm25_results:
        combined_results[result.id_] = combined_results.get(result.id_, 0) + (1 - alpha)
    # Weight vector results
    for result in vector_results:
        combined_results[result.id_] = combined_results.get(result.id_, 0) + alpha

    # Sort results based on the combined score
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    # Return the top N results
    return [docstore.get_document(doc_id) for doc_id, _ in sorted_results[:4]]

@app.route("/admin", methods=["GET", "POST"])
async def admin_page(request: Request):
    """Admin page to manage documents."""
    try:
        if request.method == "POST":
            form_data = await request.form()  # Parse the form data
            selected_section = form_data.get("section")

            # Ensure selected_section is valid
            if selected_section not in SECTION:
                raise ValueError("Invalid section selected.")

            collection_name = COLLECTION[SECTION.index(selected_section)]
            db_path = DATABASE[SECTION.index(selected_section)]

            logging.info(f"Selected section: {selected_section}, Collection: {collection_name}, DB Path: {db_path}")

            return templates.TemplateResponse(
                'admin.html',
                {
                    "request": request,
                    "section": selected_section,
                    "collection": collection_name,
                    "db_path": db_path
                }
            )

        logging.info('Rendering admin page')
        return templates.TemplateResponse(
            'admin.html',
            {
                "request": request,
                "sections": SECTION
            }
        )
    except Exception as e:
        logging.error(f"Error rendering admin page: {e}")
        return JSONResponse(
            {"status": "error", "message": f"Error rendering admin page: {str(e)}"},
            status_code=500
        )
@app.post("/upload")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    collection: str = Form(...),
    db_path: str = Form(...)
):
    """Handle file uploads for documents."""
    try:
        if files:
            logging.info(f"Handling upload for collection: {collection}, DB Path: {db_path}")
            for file in files:
                file_content = await file.read()
                file_name = file.filename

                try:
                    # Parse the uploaded file using LlamaParse
                    parsed_text = use_llamaparse(file_content, file_name)

                    # Split the parsed document into chunks
                    base_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
                    nodes = base_splitter.get_nodes_from_documents([Document(text=parsed_text)])

                    # Initialize storage context (defaults to in-memory)
                    storage_context = StorageContext.from_defaults()

                    # Prepare for storing document chunks
                    base_file_name = os.path.basename(file_name)
                    chunk_ids = []
                    metadatas = []

                    for i, node in enumerate(nodes):
                        chunk_id = f"{base_file_name}_{i + 1}"
                        chunk_ids.append(chunk_id)

                        metadata = {"type": base_file_name, "source": file_name}
                        metadatas.append(metadata)

                        document = Document(text=node.text, metadata=metadata, id_=chunk_id)
                        storage_context.docstore.add_documents([document])

                    # Load existing documents from the .json file if it exists
                    for i in range(len(DOCSTORE)):
                        if collection in DOCSTORE[i]:
                            coll = DOCSTORE[i]
                            break
                    existing_documents = {}
                    if os.path.exists(coll):
                        with open(coll, "r") as f:
                            existing_documents = json.load(f)

                        # Persist the storage context (if necessary)
                        storage_context.docstore.persist(coll)

                        # Load new data from the same file (or another source)
                        with open(coll, "r") as f:
                            st_data = json.load(f)

                        # Update existing_documents with st_data
                        for key, value in st_data.items():
                            if key in existing_documents:
                                # Ensure the existing value is a list before extending
                                if isinstance(existing_documents[key], list):
                                    existing_documents[key].extend(
                                        value
                                    )  # Merge lists if key exists
                                else:
                                    # If it's not a list, you can choose to replace it or handle it differently
                                    existing_documents[key] = (
                                        [existing_documents[key]] + value
                                        if isinstance(value, list)
                                        else [existing_documents[key], value]
                                    )
                            else:
                                existing_documents[key] = value  # Add new key-value pair

                        merged_dict = {}
                        for d in existing_documents["docstore/data"]:
                            merged_dict.update(d)
                        final_dict = {}
                        final_dict["docstore/data"] = merged_dict

                        # Write the updated documents back to the JSON file
                        with open(coll, "w") as f:
                            json.dump(final_dict, f, indent=4)

                    else:
                        # Persist the storage context if the file does not exist
                        storage_context.docstore.persist(coll)

                    # Initialize vector store index and add document chunks to collection
                    collection_instance = init_chroma_collection(db_path, collection)

                    embed_model = OpenAIEmbedding()
                    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
                    batch_size = 500
                    for i in range(0, len(nodes), batch_size):
                        batch_nodes = nodes[i : i + batch_size]
                        try:
                            collection_instance.add(
                                documents=[node.text for node in batch_nodes],
                                metadatas=metadatas[i : i + batch_size],
                                ids=chunk_ids[i : i + batch_size],
                            )
                            time.sleep(5)  # Add a retry with a delay
                            logging.info(f"Files uploaded and processed successfully for collection: {collection}")
                            return JSONResponse({"status": "success", "message": "Documents uploaded successfully."})


                        except:
                            # Handle rate limit by adding a delay or retry mechanism
                            print("Rate limit error has occurred at this moment")
                            return JSONResponse({"status": "error", "message": f"Error processing file {file_name}."})



                except Exception as e:
                    logging.error(f"Error processing file {file_name}: {e}")
                    return JSONResponse({"status": "error", "message": f"Error processing file {file_name}."})

        logging.warning("No files uploaded.")
        return JSONResponse({"status": "error", "message": "No files uploaded."})
    except Exception as e:
        logging.error(f"Error in upload_files: {e}")
        return JSONResponse({"status": "error", "message": "Error during file upload."})



def use_llamaparse(file_content, file_name):
    try:
        with open(file_name, "wb") as f:
            f.write(file_content)

        # Ensure the result_type is 'text', 'markdown', or 'json'
        parser = LlamaParse(result_type='text', verbose=True, language="en", num_workers=2)
        documents = parser.load_data([file_name])

        os.remove(file_name)

        res = ''
        for i in documents:
            res += i.text + " "
        return res
    except Exception as e:
        logging.error(f"Error parsing file: {e}")
        raise

def init_chroma_collection(db_path, collection_name):
    try:
        db = chromadb.PersistentClient(path=db_path)
        collection = db.get_or_create_collection(collection_name, embedding_function=openai_ef)
        logging.info(f"Initialized Chroma collection: {collection_name} at {db_path}")
        return collection
    except Exception as e:
        logging.error(f"Error initializing Chroma collection: {e}")
        raise


@app.get("/show_documents")
async def show_documents(request: Request,
                          collection_name: str = Form(...),
                          db_path: str = Form(...)  ):
    """Show available documents."""
    try:

        if not collection_name or not db_path:
            raise ValueError("Missing 'collection' or 'db_path' query parameters.")

        # Initialize the collection
        collection = init_chroma_collection(db_path, collection_name)

        # Retrieve metadata and IDs from the collection
        docs = collection.get()['metadatas']
        ids = collection.get()['ids']

        # Create a dictionary mapping document names to IDs
        doc_name_to_id = {}
        for doc_id, meta in zip(ids, docs):
            if 'source' in meta:
                doc_name = meta['source'].split('\\')[-1]
                if doc_name not in doc_name_to_id:
                    doc_name_to_id[doc_name] = []
                doc_name_to_id[doc_name].append(doc_id)

        # Get the unique document names
        doc_list = list(doc_name_to_id.keys())

        # Logging the successful retrieval
        logging.info(f"Documents retrieved successfully for collection: {collection_name}")

        # Render the template with the document list
        return templates.TemplateResponse(
            'admin.html',
            {
                "request": request,
                "section": collection_name,
                "documents": doc_list,
                "collection": collection_name,
                "db_path": db_path,
                "sections": SECTION,
            }
        )

    except Exception as e:
        logging.error(f"Error showing documents: {e}")
        return JSONResponse({"status": "error", "message": "Error showing documents."})

@app.post("/delete_document")
async def delete_document(request: Request,
                          collection_name: str = Form(...),
                          doc_name: str = Form(...),
                          db_path: str = Form(...)  ):
    """Handle document deletion."""
    try:
        # Initialize the collection
        collection = init_chroma_collection(db_path, collection_name)
        print("document to be deleted",doc_name)
        # Retrieve metadata and IDs from the collection
        docs = collection.get()['metadatas']
        ids = collection.get()['ids']

        # Create a dictionary mapping document names to IDs
        doc_name_to_id = {}
        for doc_id, meta in zip(ids, docs):
            if 'source' in meta:
                name = meta['source'].split('\\')[-1]
                if name not in doc_name_to_id:
                    doc_name_to_id[name] = []
                doc_name_to_id[name].append(doc_id)

        # Get the unique document names
        ids_to_delete = doc_name_to_id.get(doc_name, [])

        print("Document name: ", doc_name)
        print("IDs to delete: ", ids_to_delete)

        if ids_to_delete:
            # Attempt deletion
            collection.delete(ids=ids_to_delete)

             # Step 1: Read the JSON file
            for i in range(len(DOCSTORE)):
                if collection_name in DOCSTORE[i]:
                    coll = DOCSTORE[i]
                    break
            with open(coll, 'r') as file:
                data = json.load(file)["docstore/data"]

            for i in ids_to_delete:
                del data[i]

            final_dict = {}
            final_dict["docstore/data"] = data


            with open(coll, 'w') as file:
                json.dump(final_dict, file, indent=4)

            logging.info(f"Document '{doc_name}' deleted successfully.")
            return JSONResponse({"status": "success", "message": f"Document '{doc_name}' deleted successfully."})
        else:
            logging.warning(f"Document '{doc_name}' not found for deletion.")
            return JSONResponse({"status": "error", "message": "Document not found."})
    except Exception as e:
        logging.error(f"Error deleting document '{doc_name}': {e}")
        print(f"Error deleting document: {e}")  # Print exception for debugging
        return JSONResponse({"status": "error", "message": "Error deleting document."})






