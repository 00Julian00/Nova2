"""
Description: This script collects all data provided from the voice analysis and stores them in long and short term context memory.
"""
import threading

import torch

from .context_data_manager import ContextDataManager
from .context_data import Listener
from .database_manager import VoiceDatabaseManager
from .llm_data import Conversation, Message
from .transcriptor_data import Word

class ContextManager:
    def __init__(self) -> None:
        """
        Prepares context data provided by a listener and stores them in the context file.
        """
        self._context_data_manager = ContextDataManager()
        self._voice_database_manager = VoiceDatabaseManager()

    def record_data(self, listener: Listener) -> None:
        """
        Begins to listen to the listener and record the data.
        """
        self._context_recorder = threading.Thread(target=self._record_context, args=(listener,))

    def _record_context(self, listener: Listener) -> None:
        """
        Prepares the data from 'transcriptor.py' to be stored in 'context.json'.
        """
        last_sentence = ""
        has_sentence_been_finished = False

        for sentence in listener.data():
            sentence_string = self._word_array_to_string(sentence)

            if sentence_string != last_sentence:
                has_sentence_been_finished = False
                last_sentence = sentence_string
                continue

            # Sentence is complete and can be processed.
            if (("." in sentence_string or "!" in sentence_string or "?" in sentence_string) and not has_sentence_been_finished):
                embedding_list = [word.speaker_embedding for word in sentence]
                average_embedding = self._take_average_embedding(embedding_list)
                
                self._voice_database_manager.open()

                voice = self._voice_database_manager.get_voice_name_from_embedding(average_embedding)

                if voice is not None:
                    if voice[1] < 0.8:  # If the confidence score is too low
                        voice_name = self._voice_database_manager.create_unknown_voice(average_embedding)
                    else:
                        voice_name = voice[0]
                else:
                    voice_name = self._voice_database_manager.create_unknown_voice(average_embedding)

                self._voice_database_manager.close()

                self._context_data_manager.add_to_context(source={"voice": voice_name}, content=sentence_string)
                has_sentence_been_finished = True

    def _word_array_to_string(self, word_array: list[Word]) -> str:
        text = ""
        for word in word_array:
            text += word.text
        return text
    
    def _take_average_embedding(embeddings: list[torch.FloatTensor]) -> torch.FloatTensor:
        return torch.mean(torch.stack(embeddings), dim=0)

    #* Placeholder code to connect all systems together for first integrated tests.
    def get_context(self) -> Conversation:
        """
        Load the context data from context.json and build a conversation from it.

        Returns:
            Conversation: The conversation built from the context data.
        """
        context = self._context_data_manager.get_context_data()

        conversation = Conversation()

        current_speaker = None # Keep track of the current voice to inform the LLM if the speaker has changed.

        for entry in context:
            if entry["source"] == {"assistant"}:
                author = "assistant"
            else:
                author = "user"
            
            if "voice" in entry["source"]:
                if entry["source"]["voice"] != current_speaker:
                    current_speaker = entry["source"]["voice"]

                    conversation.add_message(message=Message(
                        author="system",
                        content=f"The speaker has changed and is now: {current_speaker}."
                    ))

            conversation.add_message(message=Message(
                author=author,
                content=entry["content"]
            ))

        return conversation

    def start(self) -> None:
        """
        Begin the data processing.
        """
        self._context_recorder.start()

    def close(self) -> None:
        """
        End the data processing.
        """
        if self._context_recorder.is_alive():
            self._context_recorder.join()