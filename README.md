# Long Document QA
## Overview
Various LLM-based systems to develop and test a more robust long document QA systems. Main system types include RAG-based applications and long context LLMs. 

## Project Statement
Much information is contained in various documents that we use everyday -- such as news, stories, research papers, manuals, emails, recipies, etc. We often need to manually read/check many documents for the knowledge we need. Can we automate this knowledge acquisition process by automatically reading a document and answering questions? Given a document (e.g., a research paper), the model will read it and answer a question based on the information in the document in natural language.

## Setup
This setup guide will walk through using the attention_viewer/ directory. To set up the local LLM testing suite in local_llm, look at the README within the directory. To use dbqa.ipynb, simply make sure to run the first installation cell.  
  
attention_viewer/ is setup using a uv virtual environment.  

...

## Codebase Structure
```
evan-hutchins-long-document-qa/
    - attention_viewer/
        -- attention_visual/
            --- light_attention_viewer.html
        -- test_docs
            --- PaperQA_single.pdf
            --- PaperQA.pdf
        -- doc_loader.py
        -- main.py
    - docs/
        -- first_task.md
    - local_llm/
        -- test_docs/
            --- Chapter06.pdf
            --- sample_test.pdf
        -- long_context.py
        -- requirements.txt
        -- text_experiments.py
    - dbqa.ipynb
```

### Repository Contents
- dbqa.ipynb - First iteration through Python notebook. 
    - RAG (Retrieval Augmented Generation) system using LangChain that processes a PDF file into text, chunks the document, and creates an embedding in a Chroma vector database. Chunks are retrieved by user query through MMR vector search and routed to Gemini 2.5 Flash to generate a response.
    - Simple LCLM (Long Context Language Model) that provides full document and user query to Gemini 2.5 Flash (1M token context length) to generate a response. Experimented on both systems with a 20 page paper.
- local_llm/ - Second iteration through Python script on local machine.
    - Similar structure to the first LCLM, but built on top of Ollama to streamline process of running open-source LLMs locally rather than through an API to avoid rate limits. 
    - Tested with Gemma 3 and Llama 3.1 (both 128k token context window). Experimented with a 100 page paper.  
- attention_viewer/ - Third iteration through Python script on local machine.
    - Prototyped with PyTorch and the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library to visualize which paragraphs in a document were given the most attention. This is done by extracting the attention patterns of all tokens per attention head and per layer and taking the mean attention over heads for each paragraph. An HTML file is created that highlights the top N paragraphs in the document that received the most attention.
    - Tested with Llama-3.2-1B-Instruct. Due to hardware constraints, only documents up to 1 page in length were tested.
- docs/ - Project reports.

## Demo Video
...

## Algorithmic Design
...

## Issues and Future Work
...

## Change Log
...


