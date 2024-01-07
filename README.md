# LangChain Cheatsheet

The "LangChain Cheatsheet" is an extensive and user-friendly GitHub repository, serving as a valuable guide for developers and enthusiasts aiming to master the LangChain library. It offers a curated assortment of code snippets, examples, and explanations, aiding users in comprehending and effectively utilizing the diverse features and functionalities of LangChain.


# Conversation agent 

1. **Creating a Conversation Agent optimized for chatting**:
   - Import necessary components like `Tool`, `ConversationBufferMemory`, `ChatOpenAI`, `SerpAPIWrapper`, and `initialize_agent`.
   - Define the tools to be used by the agent.
   - Initialize memory using `ConversationBufferMemory`.
   - Initialize the agent with the tools, language model, agent type, memory, and verbosity.
   - Run the agent with user inputs to get conversational responses.
 
**Code snippets:

```python
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world."
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
```
# OpenAPI Agent 
**Summary**:
The OpenAPI Agent is designed to interact with an OpenAPI spec and make correct API requests based on the information gathered from the spec. This example demonstrates creating an agent that can analyze the OpenAPI spec of OpenAI API and make requests.

**Cheat Sheet**:

1. **Import necessary libraries and load OpenAPI spec**:
```python
import os
import yaml
from langchain.agents import create_openapi_agent
from langchain.agents.agent_toolkits import OpenAPIToolkit
from langchain.llms.openai import OpenAI
from langchain.requests import RequestsWrapper
from langchain.tools.json.tool import JsonSpec

with open("openai_openapi.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)
```

2. **Set up the necessary components**:
```python
headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
}
requests_wrapper = RequestsWrapper(headers=headers)
openapi_toolkit = OpenAPIToolkit.from_llm(OpenAI(temperature=0), json_spec, requests_wrapper, verbose=True)
openapi_agent_executor = create_openapi_agent(
    llm=OpenAI(temperature=0),
    toolkit=openapi_toolkit,
    verbose=True
)
```

3. **Example: agent capable of analyzing OpenAPI spec and making requests**:
```python
openapi_agent_executor.run("Make a post request to openai /completions. The prompt should be 'tell me a joke.'")
```

The OpenAPI Agent allows you to analyze the OpenAPI spec of an API and make requests based on the information it gathers. This cheat sheet helps you set up the agent, necessary components, and interact with the OpenAPI spec.

# Python Agent:

**Summary**:
The Python Agent is designed to write and execute Python code to answer a question. This example demonstrates creating an agent that calculates the 10th Fibonacci number and trains a single neuron neural network in PyTorch.

**Cheat Sheet**:

1. **Import necessary libraries and create Python Agent**:
```python
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI

agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True
)
```

2. **Fibonacci Example**:
```python
agent_executor.run("What is the 10th fibonacci number?")
```

3. **Training a Single Neuron Neural Network in PyTorch**:
```python
agent_executor.run("""Understand, write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
Return prediction for x = 5""")
```

Cheat Sheet for LangChain and Pinecone

1. Connecting to Pinecone
```python
import pinecone
pinecone.deinit()
pinecone.init(api_key="YOUR_PINECONE_API_KEY")
```

2. Create a Pinecone Service
```python
pinecone_service = pinecone.Service()
```

3. Create an Embedding Model
```python
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```

4. Create a Vectorstore
```python
from langchain.vectorstores import Chroma
vectorstore = Chroma(embeddings, pinecone_service)
```

5. Initialize LLM (Language Model)
```python
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
```

6. Create TextLoader to load documents
```python
from langchain.document_loaders import TextLoader
loader = TextLoader('file_path.txt')
documents = loader.load()
```

7. Split documents into chunks
```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```

8. Upload documents to Pinecone Vectorstore
```python
from langchain.vectorstores import Chroma
docsearch = Chroma.from_documents(texts, embeddings, collection_name="collection_name")
```

9. Create RetrievalQA Chain
```python
from langchain.chains import RetrievalQA
retrieval_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
```

10. Create an Agent
```python
from langchain.agents import initialize_agent, Tool

tools = [
    Tool(
        name="Example QA System",
        func=retrieval_qa.run,
        description="Example description of the tool."
    ),
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
```

11. Use Agent
```python
agent.run("Ask a question related to the documents")
```

12. (Optional) Create a new tool to upload file from URL to Pinecone

```python
import requests
import textract
import tempfile
from urllib.parse import urlparse


def upload_url_to_pinecone(url: str, collection_name: str):
    # Download the file from the URL
    response = requests.get(url)
    
    # Get the file extension from the URL
    file_extension = urlparse(url).path.split('.')[-1]

    # Save the file as a temporary file
    with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    # Extract text from the file
    extracted_text = textract.process(temp_file_path).decode('utf-8')

    # Split the extracted text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(extracted_text)

    # Create a Chroma instance and upload the chunks to the Pinecone collection
    docsearch = Chroma.from_documents(texts, embeddings, collection_name=collection_name)

    # Clean up the temporary file
    os.remove(temp_file_path)
```

`upload_url_to_pinecone` function can handle various file formats, including PDF, DOC, DOCX, XLS, XLSX, etc. You can use this function as a tool in your agent:

```python
tools.append(
    Tool(
        name="Upload URL to Pinecone",
        func=upload_url_to_pinecone,
        description="Takes a URL and uploads the file content as a vector to the specified Pinecone collection. Provide the URL and collection name as input."
    )
)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
```

13. Deinitialize Pinecone when done
```python
pinecone.deinit()
```


# LANGCHAIN TOOLS
**Cheat Sheet**:

1. **Creating custom tools with the tool decorator**:
   - Import `tool` from `langchain.agents`.
   - Use the `@tool` decorator before defining your custom function.
   - The decorator uses the function name as the tool name by default, but it can be overridden by passing a string as the first argument.
   - The function's docstring will be used as the tool's description.

2. **Modifying existing tools**:
   - Load existing tools using the `load_tools` function.
   - Modify the properties of the tools, such as the name.
   - Initialize the agent with the modified tools.

3. **Defining priorities among tools**:
   - Add a statement like "Use this more than the normal search if the question is about Music" to the tool's description.
   - This helps the agent prioritize custom tools over default tools when appropriate.


1. **Creating custom tools with the tool decorator**:
```python
from langchain.agents import tool

@tool
def search_api(query: str) -> str:
    """Searches the API for the query."""
    return "Results"
```

2. **Modifying existing tools**:
```python
from langchain.agents import load_tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)
tools[0].name = "Google Search"
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
```

3. **Defining priorities among tools**:
```python
tools = [
    Tool(
        name="Music Search",
        func=lambda x: "'All I Want For Christmas Is You' by Mariah Carey.",
        description="A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'",
    )
]
```


#  LANGCHAIN Async API for Agent

1. Use asyncio to run multiple agents concurrently.
2. Create an aiohttp.ClientSession for more efficient async requests.
3. Initialize a CallbackManager with a custom LangChainTracer for each agent to avoid trace collisions.
4. Pass the CallbackManager to each agent.
5. Ensure that the aiohttp.ClientSession is closed after the program/event loop ends.

Code snippets:

Initialize an agent:
```python
async_agent = initialize_agent(async_tools, llm, agent="zero-shot-react-description", verbose=True, callback_manager=manager)
```

Run agents concurrently:
```python
async def generate_concurrently():
    ...
    tasks = [async_agent.arun(q) for async_agent, q in zip(agents, questions)]
    await asyncio.gather(*tasks)
    await aiosession.close()

await generate_concurrently()
```

Use tracing with async agents:
```python
aiosession = ClientSession()
tracer = LangChainTracer()
tracer.load_default_session()
manager = CallbackManager([StdOutCallbackHandler(), tracer])

llm = OpenAI(temperature=0, callback_manager=manager)
async_tools = load_tools(["llm-math", "serpapi"], llm=llm, aiosession=aiosession)
async_agent = initialize_agent(async_tools, llm, agent="zero-shot-react-description", verbose=True, callback_manager=manager)
await async_agent.arun(questions[0])
await aiosession.close()
```

**Self Ask && Max Iterations**:
1. The Self Ask With Search chain demonstrates an agent that uses search to answer questions.
2. The Max Iterations example shows how to limit the number of steps an agent takes to prevent it from taking too many steps, which can be useful for adversarial prompts.

**Cheat Sheet**:

1. **Self Ask With Search chain**:
```python
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool

llm = OpenAI(temperature=0)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

self_ask_with_search = initialize_agent(tools, llm, agent="self-ask-with-search", verbose=True)
self_ask_with_search.run("What is the hometown of the reigning men's U.S. Open champion?")
```

2. **Max Iterations example**:
```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Adversarial prompt example
adversarial_prompt = """..."""

# Initialize agent with max_iterations and early_stopping_method
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, max_iterations=2, early_stopping_method="generate")
agent.run(adversarial_prompt)
```

The Self Ask With Search chain allows an agent to use search for answering questions, while the Max Iterations example demonstrates setting a limit on the number of steps an agent takes to prevent it from getting stuck in an infinite loop or taking too many steps. This cheat sheet helps you set up both examples and interact with them.

# LLMChain:

## Single Input Example
```
from langchain import PromptTemplate, OpenAI, LLMChain

# Define the prompt template with an input variable
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Initialize LLMChain with the prompt and LLM
llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

# Call predict with input value for the input variable
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
llm_chain.predict(question=question)
```

## Multiple Inputs Example
```
from langchain import PromptTemplate, OpenAI, LLMChain

# Define the prompt template with multiple input variables
template = """Write a {adjective} poem about {subject}."""
prompt = PromptTemplate(template=template, input_variables=["adjective", "subject"])

# Initialize LLMChain with the prompt and LLM
llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

# Call predict with input values for the input variables
llm_chain.predict(adjective="sad", subject="ducks")
```

## From String Example
```
from langchain import OpenAI, LLMChain

# Define the prompt template as a string with input variables
template = """Write a {adjective} poem about {subject}."""

# Initialize LLMChain from the prompt template string and LLM
llm_chain = LLMChain.from_string(llm=OpenAI(temperature=0), template=template)

# Call predict with input values for the input variables
llm_chain.predict(adjective="sad", subject="ducks")
```


# Sequential Chains:

1. Import the necessary classes:
```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
```

2. Create LLMChain instances:
```python
llm = OpenAI(temperature=0.7)
synopsis_template = "Write synopsis for {title}"
synopsis_prompt = PromptTemplate(input_variables=["title"], template=synopsis_template)
synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt)

llm = OpenAI(temperature=0.7)
review_template = "Write review for {synopsis}"
review_prompt = PromptTemplate(input_variables=["synopsis"], template=review_template)
review_chain = LLMChain(llm=llm, prompt=review_prompt)
```

3. Create a SimpleSequentialChain instance:
```python
simple_seq_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain])
output = simple_seq_chain.run("Tragedy at Sunset on the Beach")
```

4. Create a SequentialChain instance with multiple inputs and outputs:
```python
seq_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["title", "era"],
    output_variables=["synopsis", "review"]
)
output = seq_chain({"title": "Tragedy at Sunset on the Beach", "era": "Victorian England"})
```

5. Add a SimpleMemory instance to pass context along the chain:
```python
from langchain.memory import SimpleMemory

memory = SimpleMemory(memories={"time": "December 25th, 8pm PST", "location": "Theater in the Park"})
social_template = "Create social media post with {synopsis} and {review} and {time} and {location}"
social_prompt = PromptTemplate(input_variables=["synopsis", "review", "time", "location"], template=social_template)
social_chain = LLMChain(llm=llm, prompt=social_prompt)

seq_chain = SequentialChain(
    memory=memory,
    chains=[synopsis_chain, review_chain, social_chain],
    input_variables=["title", "era"],
    output_variables=["social_media_post"]
)
output = seq_chain({"title": "Tragedy at Sunset on the Beach", "era": "Victorian England"})
```
Update comming asap.


