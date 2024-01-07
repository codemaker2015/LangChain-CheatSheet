**Conversation agent Cheat Sheet**:

1. Import the necessary components, define the tools to be used, initialize the memory, initialize the agent, and run the agent with user inputs to get conversational responses.

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

**OpenAPI Agent Cheat Sheet**:

1. Import the necessary libraries and load OpenAPI spec.
2. Set up the necessary components.
3. Initialize an agent capable of analyzing OpenAPI spec and making requests.

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

openapi_agent_executor.run("Make a post request to openai /completions. The prompt should be 'tell me a joke.'")
```

**Python Agent Cheat Sheet**:

1. Import the necessary libraries and create the Python agent.
2. Use the agent to calculate the 10th Fibonacci number or train a single neuron neural network in PyTorch.

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

agent_executor.run("What is the 10th fibonacci number?")

agent_executor.run("""Understand, write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
Return prediction for x = 5""")
```

# Pinecone Cheat Sheet

1. **Connecting to Pinecone**:
```python
import pinecone
pinecone.deinit()
pinecone.init(api_key="YOUR_PINECONE_API_KEY")
```

2. **Creating a Pinecone Service**:
```python
pinecone_service = pinecone.Service()
```

3. **Creating an Embedding Model**:
```python
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```

4. **Creating a Vectorstore**:
```python
from langchain.vectorstores import Chroma
vectorstore = Chroma(embeddings, pinecone_service)
```

5. **Initializing LLM (Language Model)**:
```python
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
```

6. **Creating TextLoader to Load Documents**:
```python
from langchain.document_loaders import TextLoader
loader = TextLoader('file_path.txt')
documents = loader.load()
```

7. **Splitting Documents into Chunks**:
```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```

8. **Uploading Documents to Pinecone Vectorstore**:
```python
from langchain.vectorstores import Chroma
docsearch = Chroma.from_documents(texts, embeddings, collection_name="collection_name")
```

9. **Creating RetrievalQA Chain**:
```python
from langchain.chains import RetrievalQA
retrieval_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
```

10. **Creating an Agent**:
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

11. **Using the Agent**:
```python
agent.run("Ask a question related to the documents")
```

12. **(Optional) Creating a New Tool to Upload a File from URL to Pinecone**:
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
