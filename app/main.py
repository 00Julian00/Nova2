"""
Description: This script controls the execution of Nova
"""

#* Placeholder code for first integrated tests

import time
import threading

import torch

from .context_manager import ContextManager
from .llm_manager import LLMManager
from .tts_manager import TTSManager
from .transcriptor import VoiceAnalysis, VoiceProcessingHelpers
from .llm_data import Conversation, Message, LLMResponse

class Nova:
    def __init__(self):
        pass

    def debug(self, voice_analysis):
        for sentence in voice_analysis.start():
            sentence_string = VoiceProcessingHelpers.word_array_to_string(sentence)
            print(sentence_string)

    def start(self):
        # Check for CUDA availability
        if (not torch.cuda.is_available):
            raise Exception("CUDA is required to run Nova.")
        
        transcriptor = VoiceAnalysis(
            microphone_index=11,
            speculative=False,
            whisper_model="deepdml/faster-whisper-large-v3-turbo-ct2",
            device="cuda",
            voice_boost=10.0,
            language="de",
            verbose=True
            )

        thread = threading.Thread(target=self.debug, args=(transcriptor,))
        #thread.start()
        #thread.join()

        contextManager = ContextManager(voice_analysis=transcriptor)
        llmManager = LLMManager()
        ttsManager = TTSManager()

        contextManager.start()
        
        #exit()

        print("Listening...")

        while True:
            conversation = contextManager.get_context()
            message = conversation.get_newest("user")

            if not message:
                time.sleep(0.1)
                continue

            #* Placeholder hotword system
            if "Nova" in message.content:
                print("Hotword detected")
                reponse = llmManager.prompt_llm(
                    conversation=conversation,
                    tools=None,
                    model="llama-3.2-90b-vision-preview",
                    perform_rag=False
                    )
                
                ttsManager.interrupt()
                ttsManager.speak(reponse.message)

            time.sleep(0.1)