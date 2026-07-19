# LLM Prompting & Agentic Workflows with LangChain

## 📌 Overview

This project explores LangChain's prompting and agent capabilities in progressively more complex stages — from a single structured prompt, to parameterized prompt templates, to autonomous ReAct agents equipped with web search and Python execution tools. The final stage applies these agentic tools to a real business forecasting scenario, comparing how two different open-source LLMs (Llama 3.1 8B and Qwen 2.5 7B) handle multi-step reasoning tasks that combine numerical calculation with external research.

## 🎯 Problem Statement

Getting useful, structured output from an LLM requires more than a single well-worded prompt — real applications need reusable, parameterized prompts, and often need the model to actively gather information or perform calculations rather than rely solely on its training data. This project investigates the practical gap between "asking an LLM a question" and "giving an LLM tools to research and compute an answer," including where autonomous agents succeed, where they falter, and how model choice affects agentic tool use.

## 🎯 Goal

- Call an LLM through LangChain with a single, well-scoped, cost-conscious prompt (short output, low temperature for factual accuracy).
- Build a reusable, parameterized prompt template to generate structured explanations across multiple topics without rewriting the prompt each time.
- Configure a ReAct agent with a web search tool to autonomously retrieve and summarize recent information (academic publications).
- Build a more capable business-analyst agent that combines a Python execution tool (for trend calculation) with web search (for qualitative context like weather and demand), and evaluate how prompt structure and model choice affect the reliability of its output.

## 🔍 Approach

### 1. Basic Prompting
- Called `meta-llama/Llama-3.1-8B-Instruct` (via `HuggingFaceEndpoint` + `ChatHuggingFace`) with a single prompt requesting a structured explanation of quantum computing (definition, advantages, current research), capped at 200 characters to control token cost.
- Set `temperature=0.01` deliberately, favoring factual precision over creative variation — an explicit, justified choice given the technical/scientific nature of the queries.

### 2. Parameterized Prompt Templates
- Built a reusable `ChatPromptTemplate` combining a system message (defining the assistant's role and output format) with a human message template accepting a `{topic}` variable.
- Chained the prompt directly to the model (`prompt | chat`) and ran it across three different topics (Bayesian methods in ML, Transformers, Explainable AI) without any prompt rewriting — demonstrating the reusability benefit of templating over ad hoc string formatting.

### 3. ReAct Agent for Information Retrieval
- Configured a ReAct-style agent (`create_react_agent`) using the standard LangChain hub prompt (`hwchase17/react`) and a free `DuckDuckGoSearchRun` tool (no API key required).
- Tasked the agent with autonomously finding and summarizing five recent academic publications on artificial intelligence, including title, authors, and description for each.
- **Observation:** the agent returned some results, but several lacked authors or abstracts and weren't genuinely the most recent publications — flagged as a limitation of the free search tool for this use case, worth revisiting with a more specialized academic search API.

### 4. Multi-Tool Business Analyst Agent
- Extended the agent with a second tool — `PythonREPL` — enabling it to perform its own numerical calculations (e.g., trend extrapolation) in addition to web search.
- Tested the agent on a realistic forecasting task: estimating 2026 orange export volumes from Brazil given 4 years of historical export data, instructed to factor in weather conditions and global demand.
- Compared two prompting strategies on the same task:
  - An **open-ended prompt** simply describing the task and asking the agent to only commit to an answer once confident.
  - A **explicitly staged, step-by-step prompt** (calculate baseline trend → search weather → search demand → adjust and conclude) forcing the agent through a fixed reasoning sequence before answering.
- Repeated both prompting strategies on a second model (`Qwen/Qwen2.5-7B-Instruct`) to compare model-level differences in agentic tool use.

## 📊 Results

| Model | Prompt style | Estimate (tonnes) | Notes |
|---|---|---:|---|
| Llama 3.1 8B | Open-ended | 225 | Did not clearly incorporate macroeconomic search findings |
| Llama 3.1 8B | Staged, step-by-step | 225.6 | Successfully incorporated search-based context, per the explicit steps |
| Qwen 2.5 7B | Open-ended | 227 | Did not invoke the search tool at all |
| Qwen 2.5 7B | Staged, step-by-step | 226.67 | Also skipped search despite explicit instructions to use it |

**Key findings:**
- The two prompting strategies produced similar final numbers for Llama, but only the **explicitly staged prompt** reliably forced the model to actually use its search tool and incorporate qualitative context (weather, demand) rather than just extrapolating from the numbers.
- **Llama 3.1 8B showed inconsistent reliability** across repeated runs — early iterations sometimes produced no result or clearly unrealistic estimates (e.g., over 300 tonnes), motivating a second-model comparison.
- **Qwen 2.5 7B produced plausible-looking numbers but never actually invoked the search tool** in either prompting condition — a reminder that "an agent got a reasonable-looking answer" doesn't guarantee it followed the intended process, and that agent tool-use reliability varies meaningfully by underlying model, not just by prompt design.
- **API key handling:** credentials were loaded from a local `creds.json` file rather than hardcoded in the notebook, following secure credential-handling practice.

## 🛠️ Tech Stack

- **Language:** Python
- **LLM orchestration:** LangChain (`langchain_huggingface`, `langchain_core.prompts`, `langchain_classic.agents`, `langchain_classic.chains`, `langchain_classic.hub`)
- **Models:** Hugging Face Inference Endpoints — `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
- **Agent tools:** `DuckDuckGoSearchRun` (web search, no API key required), `PythonREPL` (`langchain_experimental`, for in-agent calculation)
- **Agent framework:** ReAct-style agents (`create_react_agent`, `AgentExecutor`)

## 📁 Repository Structure

├── langchain_prompts_and_agents.ipynb   # Full prompting and agent experimentation notebook

├── creds.json                           # API credentials (not committed — see setup below)

└── README.md

## 🚀 How to Run

```bash
pip install langchain langchain-huggingface langchain-classic langchain-community langchain-experimental duckduckgo-search
jupyter notebook langchain_prompts_and_agents.ipynb
```

Create a `creds.json` file in the project root with your Hugging Face API token:
```json
{"HUGGINGFACEHUB_API_TOKEN": "your-token-here"}
```

> Note: `creds.json` should never be committed to version control — add it to `.gitignore`. A Hugging Face account with API access is required to run the LLM calls.

## 📈 Next Steps

- Replace the general-purpose web search tool with a dedicated academic search API (e.g., Semantic Scholar, arXiv) for the publication-retrieval agent, to get genuinely recent, properly attributed results.
- Add explicit tool-use verification/logging so it's possible to confirm an agent actually called its tools rather than just producing a plausible-sounding answer (as seen with Qwen skipping search entirely).
- Test additional models and larger sample sizes (multiple runs per prompt/model combination) to get a statistically meaningful picture of agentic reliability, rather than relying on single-run comparisons.
