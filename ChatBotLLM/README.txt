***SET-UP PRE-REQUISITES***

1) You must have LM Studio installed on your machine 
2) Download a model of your choice in LM Studio
3) Enable the "Developer" tab in LM studio 

*** PACKAGE INSTALLATION ***
Depending on your IDE, you can either hit:

pip install -e

from the root of the project file or manually specify the path to the package in your path manager (applicable to Spyder)

***How to configure your model ***
For the package to run, you need to ensure that your LM studio has an active model. Follow these steps:
1) In LM Studio, go to Developer
2) Select "Load Model". For the package to run relatively quickly on any machine, select models with a reasonable number of parameters (4-8bn)
Optionally, you can also specify the size of the context window - this will help you to track your usage and check if the chat overflows its context window.
Also, beware that the longer the chat, the more time it will take for the model to generate the response (conversation is passed recursively) 

3) Make sure that the model status is "Running"
4) Copy the local host address from "Reachable at" and paste it in model_config.txt of ChatBotLLM. 
NB 1: the current version supports three types of LLMs, for each of which you need to fill in a separate txt file in the directory:
(1) Unimodal  - model_config.txt
(2) Embedding - embedding_model_config.txt
(3) Multimodal (supporting images) - multimodal_model_config.txt 
The default version of the package contains the models tested by the author. 
NB 2: keep /v1 at the end of the local host address for endpoints to work correctly 
5) When not needed, click "Eject" in LM studio 
