# Loading the packages
import json
from elasticsearch import Elasticsearch
import creds
from tqdm.auto import tqdm
from openai import OpenAI

# Loading the json file containing course documents and preparing it for indexing
with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)

# Flattening the nested structure into a single list of documents, adding course info to each
documents = []
for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

# Defining the elastic search client
es_client = Elasticsearch('http://localhost:9200')
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
index_name = 'course-questions'
es_client.indices.create(index=index_name, body=index_settings)

for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)

def elastic_search(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    response = es_client.search(index=index_name, body=search_query)    
    result_docs = []    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])  
    return result_docs

# Function to format the top results into a readable context string
def create_context(results):
    context = ""
    for doc in results:
        context += f"section: {doc['section']}\nquestion: {doc['question']}\nanswer : {doc['text']}\n\n"
    return context

# Function to construct a prompt for the AI using the query and the search context
def create_prompt(query, context):
    prompt_template = '''
    You are a course teaching assistant. Answer the QUESTION based on the CONTEXT. 
    Use only the facts from the CONTEXT when answering the QUESTION.
    If the CONTEXT does not contain the answer, output: I am afraid I do not know the answer to your question.

    QUESTION: {question}

    CONTEXT: {context}
    '''
    # Formatting the template with the actual query and context
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

# Function to send the prompt to the LLM API and return the generated answer
def response_generator(prompt): 
    # Defining the OpenAI-compatible client for NVIDIA's hosted model
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=creds.nvidia_api_key
    )
    # Making a chat completion request to the language model
    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1-0528",
        messages=[{'role': 'user', 'content': prompt}]
    )
    # Extracting the content from the first choice in the response
    answer = response.choices[0].message.content
    return answer

# Main logic: Get user input, run search, generate context, prompt, and return answer
query = input('Ask your question:\n')
results = elastic_search(query=query)
context = create_context(results)
prompt = create_prompt(query=query, context=context)
answer = response_generator(prompt=prompt)

# Displaying the final answer
print('\nGenerating answer...\n')
print(f"Answer: {answer}")