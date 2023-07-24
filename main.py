import numpy as np
from langchain.document_loaders import UnstructuredPDFLoader
import chromadb
import openai
#The openai key has been set via environment variables
from langchain.indexes import VectorstoreIndexCreator


loader = UnstructuredPDFLoader(
        "/Users/apple/Downloads/Nature.pdf", mode="elements", strategy="fast",
    )

#index = VectorstoreIndexCreator().from_loaders([loader])

data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# Store 
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=all_splits,embedding=OpenAIEmbeddings())

question = "What are influences the risk of illness after ingestion?"
docs = vectorstore.similarity_search(question)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
from langchain.chains import RetrievalQA
#qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())

# Build prompt
from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "Thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain

from langchain.chains import RetrievalQA
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectorstore.as_retriever(),
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                                       return_source_documents=True)

result = qa_chain({"query": question})
print(result["result"])
#print(qa_chain({"query": question}))

print("hello")
#question = "What was the first capital of India?"
#result=index.query(question)
#print(result)




