# Loading required packages
import json
import minsearch
import creds
from openai import OpenAI

# Loading the json and prepare it for indexing
with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)
documents = []
for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)


# Indexing the json
index = minsearch.Index(text_fields=['question', 'section', 'text'],
                        keyword_fields=['course'])
index.fit(documents)

boost = {'question': 5.0, 'section': 0.5}
# filter_dict = {'course': "data-engineering-zoomcamp"}
query = 'When will the course start?'

results = index.search(query= query, boost_dict= boost, num_results=5)

prompt_template = '''
You are a course teaching assistant. Answer the QUESTION based on the CONTEXT. 
Use only the facts from the CONTEXT when answering the QUESTION.
If the CONTEXT does not contain the answer, output: I am afraid I do not know the answer to your question.

QUESTION: {question}

CONTEXT: {context}
'''

context = ""
for doc in results:
    context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer : {doc['text']}\n\n"

prompt = prompt_template.format(question = query, context = context).strip()

# Defining the Open AI client
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = creds.nvidia_api_key
)

# Generating the responses
response = client.chat.completions.create(model = "deepseek-ai/deepseek-r1-0528", messages = [
                                          {'role': 'user', 'content': prompt}])

print(response.choices[0].message.content)
