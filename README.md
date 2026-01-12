# Background

The project specifically examines the use of geospatial foundation models (GFMs) for a variety of tasks including water/flood segmentation, crop classification and change detection. The main tasks include:

- Reviewing GFMs applications
- Examining GFMs data requirements
- Fine-tuning and evaluating GFMs on specific tasks
- Agentic workflow orchestration

# Project Details

The focus of this project has been to use [TerraTorch](https://github.com/terrastackai/terratorch), a platform for hosting and working with GFMs. It allows the fine-tuning of specific models on downstream tasks.

There are many pre-trained GFMs backbone that could be called from TerraTorch, but this project has targeted a version of IBM's [TerraMind](https://github.com/IBM/terramind) model which is small and can run locally. This version of the TerraMind model is not limited by the requirement for huge compute resources-- It can run on a 16GB RAM, as was done in this project.

This project quickly expanded to involve provisioning a language model with tools to perform tasks. In this project, a multi-agent workflow was demonstrated using [LangChain](https://www.langchain.com/). The first involves extracting Sentinel-2 and Sentinel-1 satellite images from an API. In this case, the [Microsoft Planetary Computer STAC API](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/) was used. The second tool allows the language model to provide the satellite images as input to a fine-tuned GFM at a checkpoint location so it can produce output predictions based on the designated task.

# Contributing to the Project

Clone the project and install the poetry-managed environment using `poetry install`. Download the datasets for fine-tuning the GFMs and adapt predefined paths in the code to point to dataset location. Adapt existing directories in the code as appropriate.
