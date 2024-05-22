import os
import sys
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "database"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    if not os.path.exists(CHROMA_PATH):
        create_database()
    
    try:
        query = sys.argv[1]
        response = prompt_query(query)
        print(response)
    except IndexError:
        print('Execute like "python3 main.py \"your question here\""')
        return
    except Exception as e:
        print(e)
        return


def create_database():
    documents = load_documents(DATA_PATH)
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents(path: str) -> list[Document]:
    loader = DirectoryLoader(path, glob="*.txt", show_progress=True)
    return loader.load()    # documents


def split_text(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document]):

    # Create a new DB from the documents.
    Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def prompt_query(query: str) -> str:
    # Query vector db for relevant results
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    results = db.similarity_search_with_relevance_scores(query, k=100)

    # Make a prompt template for context to the model
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # prompt the model about the query and its context
    model = ChatOpenAI()
    response_text = model.invoke(prompt)
    return response_text.content
    

if __name__ == "__main__":
    main()
