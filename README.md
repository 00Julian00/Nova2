# NOVA: Next-Generation Open-Source Virtual Assistant
### A Python framework for building AI assistants with minimal code. Integrates LLMs, Text-to-Speech, voice recognition, and long-term memory into a cohesive, easy-to-use system.

Version 1.0.0

## Table of contents:
1. [Introduction](#introduction)
2. [Getting started](#getting-started)
3. [Features](#features)
4. [Technologies used](#technologies-used)
5. [Project structure](#project-structure)
6. [Roadmap](#roadmap)

## Introduction
Nova2 started out as a rewrite to my [original Nova project](https://github.com/00Julian00/Nova), which is no longer being maintained.

What is Nova?  
  
Nova is an AI assistant building framework that aims to combine several technologies into one cohesive, uniform and easy to use interface, allowing developers to develop a fully working AI assistant in just a few lines of code.  
Nova is modular and easily extendable, allowing you to easily modify it to fit your needs. The complexities of LLMs, TTS, voice transcription, database management, retrieval-augumented-generation and more are abstracted away but still allow for fine control over them for experienced developers. The struggles of changing your code to fit a new system or AI model are not present as the interface always stays the same, allowing you to rapidly experiment with different AI models and systems without needing to change your code. Nova can be used by researchers or AI enthusiasts, hobbyists and in general everyone who wants something that "just works" without having to dig into documentation for every little change in the pipeline.

## Getting started
For this project you need an nvidia gpu with Cuda installed, as well as [cudnn 9.1.0](https://developer.nvidia.com/cudnn-9-1-0-download-archive).  
You also need python 3.11.X.

1. Clone the repo and navigate to the "Nova2" folder.
2. Run ```pip install -r requirements.txt```
3. LlamaCPP must be installed seperatly: ```pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124```

If you are on a linux system, you are using conda and encounter an issue like this:  
```OSError: cannot load library 'libportaudio.so.2':OSError: cannot load library 'libportaudio.so.2'```  
Try updating libstc++:  
```conda install -c conda-forge libstdcxx-ng```

Nova is now installed and ready to be used. I recommend to take a look at the examples in ```./Nova2/examples``` as they give you an overview about how Nova works and how to use it.  
Below is a simple script that shows you how to quickly set up a chat with an LLM using the Nova framework. Create a new script outside the Nova2 folder and copy this code:
```py
from nova import *

# The Nova class is the API
nova = Nova()

# The inference engine dictates how inference is run in the background
# For this example we are using LlamaCPP
inference_engine = InferenceEngineLlamaCPP()

# The conditioning contains our settings for the LLM
conditioning = LLMConditioning(
    model="bartowski/Qwen2.5-7B-Instruct-1M-GGUF",
    file="*Q8_0.gguf"
)

nova.configure_llm(inference_engine, conditioning)
nova.apply_config_llm()

conversation = Conversation()

# This is our primary loop for the back and fourth conversation between user and LLM.
while True:
    inp = input("User: ")

    msg = Message(author="user", content=inp)
    conversation.add_message(msg)

    response = nova.run_llm(conversation)

    print(f"AI: {response.message}")

    resp_message = Message(author="assistant", content=response.message)

    conversation.add_message(resp_message)
```
Remember to check out the examples to get more information about how to interact with the LLM system and other systems.

## Features
Nova aims to provide every building block you need to build an AI assistant pipeline:
- LLM inference: You can use different "inference engines" to run inference on Large-Language models. Inference engines are essentially wrappers for existing systems and APIs that run LLM inference.
- TTS inference: Like with LLMs, Nova provides inference engines for Text-to-Speech systems to turn text into spoken speech. Nova also includes an audio player that can play the resulting audio data.
- Transcriptor: Nova provides a transcriptor to continously transcribe spoken speech into text using OpenAIs Whisper model. The transcriptor also computes voice embeddings that are stored in a database that allow you to differentiate between different speakers and recognize returning speakers.
- Long term memory: Nova has a built in retrieval-augumented-generation pipeline that allows the LLM to form long-term memories of important information.
- Context system: The context system is responsible for short-term LLM memory. It organizes and saves data from various sources for the LLM to use.
- Tool system: A modular tool system that allows the LLM to perform any action you give it access to. You can also create and add new tools very easily.

## Technologies used
Nova combines a bunch of different AI models and technologies. Below is an overview over the most important models and technologies used:

### LLM system:
The LLM system comes with 2 inference engines that both handle LLM inference differently:
- The first inference engine utilizes [LlamaCPP](https://github.com/ggml-org/llama.cpp) (specifically [these](https://github.com/abetlen/llama-cpp-python) python bindings). A very fast inference engine written in C++.
- The second inference engines uses the [Groq](https://groq.com/) API. Their custom made chips allow for fast inference on large LLM models. They also offer a free API tier.

### TTS system:
The TTS system also comes with 2 inference engines:
- The first inference engine uses the [Zonos TTS model](https://github.com/Zyphra/Zonos) developed by Zyphra.
- The second inferene engine uses the [Elevenlabs API](https://elevenlabs.io/). They also offer a free API tier.

### Databases:
Nova uses 2 different database libraries:
- [Qdrant](https://qdrant.tech/) is a fast vector database framework. It is used to store long term memories as text embeddings, as well as voice embeddings.
- [Sqlalchemy](https://www.sqlalchemy.org/) provides a wrapper for SQL commands. It is used to store secrets, like API keys.

### Transcriptor:
The transcriptor combines several AI models and frameworks into its audio-preprocessing and transcription pipeline:
- [Whisper](https://openai.com/index/whisper/) developed by OpenAI is a Speech-to-Text model that can turn spoken language into text. The transcriptor uses a special approach for transcribing with whisper to allow for streaming transcriptions inspired by [this](https://www.youtube.com/watch?v=_spinzpEeFM) video.
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) is an inference engine for whisper that greatly increases inference speed over OpenAIs implementation by utilizing [CTranslate2](https://github.com/OpenNMT/CTranslate2/).
- [Denoiser](https://github.com/facebookresearch/denoiser) developed by Meta is a library that attempts to remove noise in the given audio data. The transcriptor uses it to boost the volume of speech compared to other noises in the given audio data.
- [Silero VAD](https://github.com/snakers4/silero-vad) is a voice activity detection library. It is used to detect speech in the audio data to filter out audio chunks.
- [Speechbrain](https://github.com/speechbrain/speechbrain?tab=readme-ov-file) is a library that is used to compute the speaker embeddings.

## Project structure
This section is dedicated to giving you an overview about how the project is structured:  
  
You can find almost all of the scripts in ./Nova2/app. Here, most systems are separated into a "manager" and a "data" script. The data script holds all custom datastructures the manager needs, while the manager does the actual work. The app folder also contains the "zonos" folder, which holds all code of the Zonos TTS. app also contains the "inference_engines" folder. This is where the inference engines for LLM and TTS are located. They are in the folders "inference_llm" and "inference_tts" respectivly. These folders also contain the scripts that contain the base classes the inference engine classes inherit from.  
  
In ./Nova2/data you can find the "libraries" folder that not only contains a list of which tools are considered to be "internal", but also the "prompt_library.json" file which holds all built-in prompts the system uses.  
  
./Nova2/db holds all databases of the project, which are separated into the "db_memory_embeddings", "db_secrets" and "db_voice_embeddings" folders.  
  
./Nova2/examples hold a bunch of Jupyter notebooks teching you the basics of how to user Nova.  
  
./Nova2/tool_api holds the "tool_api.py" script which provides tools with their API, as well as their base class they need to inherit from.  
  
./Nova2/tools holds all the tools Nova has access to.

## Roadmap
Below are a couple of ideas I want to implement into Nova in future releases. These are NOT guaranteed to be implemented in the future. They are just a couple of ideas I am currently playing around with.
- Add support for Vision-Language models.
- Allow for new secrets to be created via the API.
- Add support for multiple different contexts.
- Add a tool testing suite for user created tools.
- Add a "voice lock" functionality where certain data can only be accessed if a whitelisted voice is speaking.
- Add support for thinking models.
- Add more inference engines and internal tools.

## Support the Project
If you find Nova useful for your work or projects, consider [buying me a coffee](https://buymeacoffee.com/00julian00). Your support helps maintain and improve this open-source project.

## License
Nova is released under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).
