from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# loading the data 
loader = PyPDFLoader("med.pdf")
data = loader.load()  # entire PDF is loaded as a single Document

# splitting the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = text_splitter.split_documents(data)


# Load SentenceTransformer model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Chroma vector store
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

llm = ChatGroq(groq_api_key="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", 
               model_name="llama3-8b-8192", temperature=0.5)


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke({"input": "what sholud we do we have have cough and fever?"})
print("final anwer \n")
print(response["answer"])
