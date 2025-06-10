# Loading required packages
import json
import minsearch
import creds
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

# Function to perform a search on the documents using a simple index
def search(query):
    # Creating the search index with fields to search on
    index = minsearch.Index(text_fields=['question', 'section', 'text'],
                            keyword_fields=['course'])
    # Fitting the index with the prepared documents
    index.fit(documents)

    # Setting boost values to prioritize certain fields more than others
    boost = {'question': 5.0, 'section': 0.5}

    # Performing the search and returning the top 5 results
    results = index.search(query=query, boost_dict=boost, num_results=5)
    return results

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
results = search(query=query)
context = create_context(results)
prompt = create_prompt(query=query, context=context)
answer = response_generator(prompt=prompt)

# Displaying the final answer
print('\nGenerating answer...\n')
print(f"Answer: {answer}")