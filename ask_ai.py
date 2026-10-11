from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import google.genai as genai
from google.genai import types          #gemini install (for generations)
import os
import sys
import subprocess



system_instruction = """
CONTEXT: You are a clinical doctoral assistant. You have access to a database of all conditions, supplied by the NHS,
so all advice you may give is backed by them. I want you to be concise giving answers to the client, being open and honest
where needed, but not scaring them, and giving them the best advice.

USE:
When you respond, I want you to be inquisitive, obviously medical conditions are extremely specialised, and possibly unique
to each person who recieves them, so if in doubt about anything ask questions to gather more information about the client.
When you respond with a possible diagnosis, I want you to give, your thoughts of what it could be, ways to lower the sympotoms
that the user may be able to do immediately, and I want you to give them helpful advice about what to do.
When you ask questions, try to possibly do them one by one, so the user doesnt get flooded with questions, as they are more likely
to give you more context

FORMATTING:
Write in a single, continuous narrative flow, like a paragraph in a book. 
Do not use lists. Do not use bullet points. Do not use headers. 
Use full sentences and standard punctuation (commas and periods) only.

IMPORTANT:
The symbols * and ** are strictly forbidden. 
If you want to list items, use words like "firstly," "secondly," "additionally," or "finally" within a standard paragraph. 
Failure to follow this will result in a system error. 
Output ONLY plain sentences.

Try to ask questions before you diagnose as you dont want to flood the user with data. Be honest, if you conclude that the users
condition is one which doesnt need to be solved immediately you can say things like "If it doesnt get better in 4 days then call the doctor"
etc.


ALSO IMPORTANT:
You can either generate text or function_calls, I know you have the capability to do both simultaneously but doing so will result in a FATAL
system error
"""


# 1. Load the existing "brain" (Don't rebuild it!)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./medical_db", embedding_function=embeddings)

# 2. The Loop
print("💬 Medical AI is ready. Type 'exit' to stop.")

def check_database(x, query):
    # Search the database for the 3 most relevant paragraphs
    docs = vector_db.similarity_search(query, k=x)
    cumulative_source = ""
    if docs:
        for i, doc in enumerate(docs):
            cumulative_source += str(f"\n[Source {i+1}]: {doc.page_content}")
        print(cumulative_source)
        return cumulative_source
    else:
        print("\n❌ No matching information found in your files.")
        return "No relevant information found in database."


query_db = {
    "name": "check_database",
    "description":"""This is your research function. Whenever you are querying the NHS database about any conditional information
you will use this tool to search for it. Use it as much as you like,
and dont be sparing. What will be returned is all context you ask for about the condition you query for
        """,
    "parameters": {
        "type":"object",
        "properties":{
            "query":{
                "type": "string",
                "description":"The search query that you are supplying to the database, the more detailed you can make it, the more detailed your response will be",
            },
            "x":{
                "type": "integer",
                "description":""" This is the number of items in the database you are going to query, a higher number will give you more context,
but as the number gets larger, the later pieces of data will become potentially more irrelevent, I would keep this between 5 and 20.
"""
            },
        },
        "required":["query","x"]
    },
}


def handle_function_calls(response):
    """Recursively handle function calls until we get a text response"""
    # Check if this response contains a function call
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call

        
        # Execute the function
        if function_call.name == 'check_database':
            data = check_database(**function_call.args)
            # Send the function result back to the model
            new_response = chat.send_message(data)
            # Recursively handle if there are more function calls
            return handle_function_calls(new_response)
    
    # If no function call, return the response
    return response


with open('api_keys.txt', 'r', encoding='utf-8') as file:
    line = file.readline()
api_key_gemini = line #api key for gem (secret)
client = genai.Client(api_key=api_key_gemini)
tools = types.Tool(function_declarations=[query_db])
config = types.GenerateContentConfig(tools=[tools], system_instruction=system_instruction)
chat = client.chats.create(model="gemini-2.5-flash", config=config)
    

if __name__ == "__main__":
    print("\nNHSGPT How can I help you?")
    while True:
        try:
            prompt = str(input("\n--> "))
            
            if prompt.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye! Take care of your health.")
                break

            # Send the message
            response = chat.send_message(prompt)
            
            # Handle any function calls and get final response
            final_response = handle_function_calls(response)
            
            # Print the final text response
            print(f"\n{final_response.text}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye! Take care of your health.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
