from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

client = OpenAI(
    
)

## vectoer emmebding

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


vector_db = QdrantVectorStore.from_existing_collection(

    url="http://localhost:6333",
    collection_name="lernnig_rag",
    embedding=embedding_model
)

## Take user input

user_input = input["Ask somthing : "]

## relevent chunks form the db 
search_results = vector_db.similarity_search(query=user_input)

context = "\n\n\n".join([
    f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
    for result in search_results
])

SYSTEM_PROMPT = f"""

You are helfull ai assistant who answers the user query based on the avilable context
retrived form the PDF file along with   page_content and page_number

You should only ans the user based on the following context to navigate the user 
to  open the right page number to know more.

Context:
{context}

"""


response = client.chat.completions.create(

    model="gpt-5",
    messages =[

        {"role": "system" ,"content ":SYSTEM_PROMPT },
        {"role": "user" ,"content ":user_input }
    ]
)


print(f"ans : {response.choices[0].message.content}")





