from langchain_community.vectorstores import Chroma
from langchain.storage import LocalFileStore
import dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.storage._lc_store import create_kv_docstore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import tiktoken
from langchain.chains import LLMChain

st.set_page_config(page_title="AIDL RAG Chatbot", page_icon="🦜")
st.title("AIDL RAG Chatbot")



old='''
__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')'''



with st.sidebar:
     
    st.write("API keys")

    openai_api_key=st.text_input("Open AI key",placeholder="Enter Open AI Key",value="Replace with a valid open ai key")

    st.markdown("""---""")
    

    with st.expander("Advanced Options", expanded=False):
            prompt_options = st.selectbox(
                'Prompt:',
            ('Langchain Default', 'Latest Optimized Prompt','Custom Prompt'))
        
            if prompt_options=="Custom Prompt": 
                st.write("Please include {context} and {input} in your custom prompt or it will not be accepted")
                custom_prompt_input=st.text_area("Custom Prompt",placeholder="Enter the custom prompt",value="Thoroughly examine all the provided context to craft the most accurate response to the following question: \n\n Context: {context} \n Question: {input}")



llm= ChatOpenAI(temperature=0,model="gpt-3.5-turbo-0125",openai_api_key=openai_api_key)  

embeddings = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=openai_api_key)

# Chroma Db

index_path="./extras/chroma_db"
parent_store="./extras/store_location"
    

old_2='''
'''

prompt = PromptTemplate.from_template('''
You are a chatbot office hours assistant. You provide help for students regarding their course material. Make sure to respect the order of the lessons when a student is asking about a certain topic. Meaning that, if a student is studying lesson 4 don't give him answers with required knowledge of lesson 5. But you can clarify that if needed.
week1:
Lesson 1: Overview of applied AI and Programming
Lesson 2: Overview of the Python Language
Lesson 3: Integrated Development Environments (IDEs)
Lesson 4: Variables
Lesson 5: Numbers
Lesson 6: Strings
week2:
Lesson 7: Input
Lesson 8: Boolean Data Type
Lesson 9: Conditionals
week3:
Lesson 10: Introduction to Loops
Lesson 11: Lists
Lesson 12: Iterators
Graded Assessment: Weeks 1-3 (20%)
week4:
Lesson 14: A Use Case (Introducing Our First Dataset)
Lesson 15: Advanced Data Structure
Lesson 16: Functions and Modular Programming
Lesson 17: File Handling in Python
week5:
Lesson 18: Classes and Object-Oriented Programming in Python
Lesson 19: Modules, Packages, and Libraries
Extended Content: Inheritance
week6:
Lesson 21: Introduction to NumPy
Lesson 22: Introduction to Pandas DataFrame
Graded Assessment: Project - Phase 1 (30%)
week7:
Lesson 24: Data Visualization
Lesson 25: Data Cleaning & Preprocessing
week8:
Lesson 26: Introduction to Machine Learning
Lesson 27: Scikit-Learn & Logistic Regression
Submission of Phase 2
Extra Lesson: Linear Regression Using Scikit-Learn
Thoroughly examine all the instructions above and the following course content to format the final answer:

Course Content: {context} 
Question: {input}

'''                                     
)




# Vector Store + Retriever
vector_db = Chroma(collection_name="split_parents",persist_directory=index_path, embedding_function=embeddings)
fs= LocalFileStore(parent_store)
store = create_kv_docstore(fs)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=vector_db,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k":3}
)





combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
old='''# Building retrieving chain
    retrieval_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever)'''


if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")


def print_first_three_lines_of_list(string_list):
    final_array=[]
    for input_string in string_list:
        lines = str(input_string)[0:100]
        final_array.append(lines)

    return final_array

def return_docs():
    total=num_tokens_from_string(str(prompt),"cl100k_base")
    final_list=[]
    for i in doc_list:
        if total + num_tokens_from_string(str(i), "cl100k_base") < 16000:
            total+=num_tokens_from_string(str(i), "cl100k_base")
            final_list.append(str(i))

    return final_list    

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    #st.write(user_query)
    doc_list=retriever.get_relevant_documents(user_query)
    #st.write(print_first_three_lines_of_list(doc_list))
    #st.write(doc_list)
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    #token_total=num_tokens_from_string(str(doc_list[0]), "cl100k_base")
   # st.write(token_total)
    #doc_list_length=len(doc_list)
    



    final_docs=return_docs()
    #st.write(print_first_three_lines_of_list(final_docs))
    #st.write(final_docs)
    
    chain = LLMChain(llm=llm, prompt=prompt)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),max_thought_containers=0)
        response=chain.invoke({"context":final_docs,"input":user_query})["text"]
        #st.write(response)
        #response=retrieval_chain.invoke({"input": user_query})["answer"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


    
            
            
            


            




 

    








