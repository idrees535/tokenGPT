# TokenGPT

TokenGPT is an AI-powered chatbot designed to assist with token engineering. It leverages the ChatGPT API, Retrieval-Augmented Generation (RAG), and function calling capabilities to provide comprehensive insights and analysis for decentralized economies.

## Setup Instructions

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/idrees535/tokenGPT.git
cd tokenGPT
```

### Step 2: Create a Virtual Environment

Create a virtual environment to manage dependencies:

```bash
python -m venv tgpt_venv
```

### Step 3: Activate the Virtual Environment

Activate the virtual environment:

- On Windows:
```bash
    tgpt_venv\Scripts\activate
```

- On macOS and Linux:
 ```bash
    source tgpt_venv/bin/activate
```

### Step 4: Install Dependencies

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```
### Set Up Environment Variables
TokenGPT requires an OpenAI API key, a base path for saving files, and a Dune API key to function. You need to provide your own API keys and base path.

Create a .env file in the root directory of the project.
Add the following environment variables to the .env file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
BASE_PATH=your_base_path_here
DUNE_API_KEY=your_dune_api_key_here
```
### Step 5: Run the Chatbot

Run the `chatbot.py` script to start the TokenGPT application:

```bash
python chatbot.py
```

### Step 6: Access the Gradio App

After running the script, a port will be opened. Open the URL of this port in your browser to interact with the Gradio app of TokenGPT. The URL will be displayed in the terminal, typically something like `http://127.0.0.1:7860`.

## Features

- **Data-Driven Insights**
- **Economic and Behavioral Analysis**
- **Risk Assessment**
- **Python-Based Modeling**
- **Precision and Conciseness**

## Components

- ChatGPT API
- RAG for Knowledge Base
- Function Calling Capability
- Integration with Data Extraction Tools
- Analysis Tools
- Graphs and Visualization Tools
- Gradio User Interface

## Future Enhancements

- Track governance activity across multiple protocols and analyze voting patterns
- Integrate smart contract audit tools / frameworks (e.g., Slither) to TokenGPT
- Integrate tools to perform formal verification of smart contracts
- Integration with more data tools
- Enhanced modeling and simulation capabilities
- Continuous improvement of the knowledge base
- Integrate smart contract development and deployment tools (e.g., Truffle/Hardhat)

---

Feel free to explore and contribute to TokenGPT! For any issues or contributions, please open an issue or submit a pull request.