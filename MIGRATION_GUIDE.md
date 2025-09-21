# Migration Guide
### Nova2 Version 1.x.x to 2.0.0

## Overview
Version 2.0.0 brings significant improvements to Nova2, including a new tool system, inference engine improvements, and several API changes. This guide covers all breaking changes and how to update your code.

## 1. Inference Engine Changes

### 1.1 Model Property Replacement
The `get_current_model()` method has been completely removed from all inference engines and replaced with the `model` property.

**Before (1.x.x):**
```python
# This will no longer work
current_model = inference_engine.get_current_model()
```

**After (2.0.0):**
```python
# Use the model property instead
current_model = inference_engine.model
```

### 1.2 Inference Engine Configuration in Conditioning Objects
Inference engines are now specified directly in the conditioning objects rather than being passed separately.

**Before (1.x.x):**
```python
# LLM
inference_engine = InferenceEngineLlamaCPP()
conditioning = LLMConditioning(
    model="bartowski/Qwen2.5-7B-Instruct-1M-GGUF",
    file="*Q8_0.gguf"
)
nova.configure_llm(inference_engine=inference_engine, conditioning=conditioning)

# TTS
inference_engine = InferenceEngineZonos()
conditioning = TTSConditioning(
    model="Zyphra/Zonos-v0.1-transformer",
    voice="Laura",
    expressivness=100,
    stability=2.0
)
nova.configure_tts(inference_engine=inference_engine, conditioning=conditioning)
```

**After (2.0.0):**
```python
# LLM
conditioning = LLMConditioning(
    model="bartowski/Qwen2.5-7B-Instruct-1M-GGUF",
    inference_engine="inference_llamacpp",  # Specify engine in conditioning
    file="*Q8_0.gguf"
)
nova.configure_llm(conditioning=conditioning)

# TTS
conditioning = TTSConditioning(
    model="Zyphra/Zonos-v0.1-transformer",
    inference_engine="inference_zonos",  # Specify engine in conditioning
    voice="Laura",
    expressivness=100,
    stability=2.0
)
nova.configure_tts(conditioning=conditioning)
```

### 1.3 New Configure and Apply Methods
New convenience methods have been added that combine configuration and application in one step.

**New in 2.0.0:**
```python
# Configure and apply in one step
nova.configure_llm_and_apply(conditioning=conditioning)
nova.configure_tts_and_apply(conditioning=conditioning)
nova.configure_stt_and_apply(conditioning=conditioning)
```

## 2. Speech-to-Text (STT) Changes

### 2.1 Transcriptor Renamed to STT
The transcriptor has been renamed to STT (Speech-to-Text) throughout the API.

**Before (1.x.x):**
```python
# Old transcriptor methods
nova.configure_transcriptor(conditioning=conditioning)
nova.apply_config_transcriptor()
```

**After (2.0.0):**
```python
# New STT methods
nova.configure_stt(conditioning=conditioning)
nova.apply_config_stt()
```

### 2.2 STT Inference Engine System
The STT system now uses the same inference engine pattern as LLM and TTS.

**New in 2.0.0:**
```python
conditioning = STTConditioning(
    model="large-v3",
    inference_engine="inference_fasterwhisper",  # Specify engine in conditioning
    language="en",  # Language now set in conditioning
    device="cuda"
)
nova.configure_stt_and_apply(conditioning=conditioning)
```

### 2.3 Language Configuration Moved
The preferred language for transcription is now set in the conditioning object instead of separately.

**Before (1.x.x):**
```python
# Language was set in the conditioning
conditioning = TranscriptorConditioning(
    model="large-v3",
    inference_engine="inference_fasterwhisper",
    device="cuda"
)
```

**After (2.0.0):**
```python
# Language is now part of conditioning
conditioning = STTConditioning(
    model="large-v3",
    inference_engine="inference_fasterwhisper",
    language="en"  # Set language here
)
```

## 3. Text-to-Speech (TTS) Changes

### 3.1 AudioData Object Return
TTS inference engines now return an `AudioData` object instead of raw audio data.

**Before (1.x.x):**
```python
# Returned raw bytes
audio_bytes = nova.run_tts("Hello world")
# You had to handle raw audio data yourself
```

**After (2.0.0):**
```python
# Returns AudioData object
audio_data = nova.run_tts("Hello world")
# AudioData object can be directly used with the audio player
nova.play_audio(audio_data=audio_data)
```

## 4. Tool System Overhaul

### 4.1 New Tool Structure
Tools have been completely reworked and now use dependency injection and a new class structure.

**Before (1.x.x):**
```python
# Old tool structure
class Tool(ToolBaseClass):
    def on_startup(self) -> None:
        # Setup code
        pass
    
    def on_call(self, **kwargs) -> None:
        # Tool logic with kwargs
        parameter1 = kwargs.get("parameter1")
        # Tool implementation
```

**After (2.0.0):**
```python
# New tool structure with dependency injection
from Nova2.tool_api import ToolBaseClass
from Nova2.app import api_tools

class Main(ToolBaseClass):  # Class must be named "Main"
    def on_startup(self):
        # Dependency injection - access to API
        self._api = api_tools.ToolAPI()
    
    @ToolBaseClass.tool
    def my_tool_function(self, parameter1: str) -> None:
        """
        Tool description for the LLM.
        
        Arguments:
            parameter1 (str): Description of parameter.
        """
        # Tool implementation
        # Add results to context
        self._api.add_to_context(
            "Tool Name", 
            "Result message", 
            tool_call_id=self._tool_call_id
        )
```

### 4.2 Tool Method Decoration
Tool methods now need to be decorated with `@ToolBaseClass.tool`.

**New in 2.0.0:**
```python
class Main(ToolBaseClass):
    @ToolBaseClass.tool
    def calculate_sum(self, a: int, b: int) -> None:
        """
        Calculates the sum of two numbers.
        
        Arguments:
            a (int): First number.
            b (int): Second number.
        """
        result = a + b
        self._api.add_to_context(
            "Calculator", 
            f"The sum of {a} and {b} is {result}", 
            tool_call_id=self._tool_call_id
        )
```

The docstring of the method, as well as the type hints of the parameters will be used as the tool description for the LLM. 

## 5. Context Management Changes

### 5.1 Multiple Context Files Support
Nova2 now supports multiple context files, allowing you to organize different conversations or contexts separately.

**New in 2.0.0:**
```python
# Set active context file
nova.set_active_context_file("project_discussion")

# Get all available context files
context_files = nova.get_all_context_files()

# Get current active context file
active_file = nova.get_active_context_file()

# Rename a context file
nova.rename_context_file("old_name", "new_name")
```

## 6. LLM Changes

### 6.1 Reasoning Process Filtering
Reasoning models can now have their reasoning process filtered out from responses.

**New in 2.0.0:**
```python
conditioning = LLMConditioning(
    model="your-reasoning-model",
    inference_engine="your_engine",
    filter_thinking_process=True  # Filter out reasoning steps
)
```

## 7. API Changes

### 7.1 Stay Alive Function
A new `stay_alive` function has been added to keep the program running.

**New in 2.0.0:**
```python
# Keep the program running
nova.stay_alive()

# Or with a custom condition
running = True
nova.stay_alive(condition=running)
```

### 7.2 Secrets Management
Secrets are now stored in a `.env` file instead of an encrypted database.

## 8. Performance Improvements

### 8.1 Lazy Imports
Inference engines are now lazily imported, removing the need to install dependencies for unused engines.

This means you only need to install dependencies for the inference engines you actually use. For example, if you only use Groq, you don't need to install LlamaCPP dependencies.

### 8.2 Reduced Speaker Embedding Generation
Speaker embedding generation frequency has been reduced to once per sentence instead of more frequently, improving performance.

## 9. Recovery Manager

### 9.1 Experimental Recovery Manager
An experimental recovery manager has been added to prevent crashes of individual components.