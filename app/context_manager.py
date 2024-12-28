"""
Description: This script collects all data provided from the voice analysis and stores them in long and short term context memory.
"""

#TODO: Objectify context data

import threading

from .transcriptor import VoiceAnalysis, VoiceProcessingHelpers
from .context_data_manager import ContextDataManager
from .database_manager import VoiceDatabaseManager
from .llm_data import Conversation, Message

class ContextManager:
    def __init__(
                self,
                voice_analysis: VoiceAnalysis = None,
                ) -> None:
        """
        This class continously gathers the data generated by 'transcriptor.py'.
        It prepares the data to be stored in 'context.json'.
        """
        self._context_data_manager = ContextDataManager()
        self._voice_database_manager = VoiceDatabaseManager()
        self._managing_pipeline = threading.Thread(target=self._manage_context, args=(voice_analysis,))

    def _manage_context(self, voice_analysis: VoiceAnalysis) -> None:
        """
        Prepares the data from 'transcriptor.py' to be stored in 'context.json'.
        """
        last_sentence = ""
        has_sentence_been_finished = False

        for sentence in voice_analysis.start():
            sentence_string = VoiceProcessingHelpers.word_array_to_string(sentence)

            if sentence_string != last_sentence:
                has_sentence_been_finished = False
                last_sentence = sentence_string
                continue

            # Sentence is complete and can be processed.
            if (("." in sentence_string or "!" in sentence_string or "?" in sentence_string) 
                and not has_sentence_been_finished):
                
                embedding_list = [word.speaker_embedding for word in sentence]
                average_embedding = VoiceProcessingHelpers.take_average_embedding(embedding_list)
                
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

    #* Placeholder code to connect all systems together for first integrated tests.
    def get_context(self) -> Conversation:
        """
        Load the context data from context.json and build a conversation from it.

        Returns:
            Conversation: The conversation built from the context data.
        """
        context = ContextDataManager().get_context_data()

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
        self._managing_pipeline.start()

    def close(self) -> None:
        """
        End the data processing.
        """
        if self._managing_pipeline.is_alive():
            self._managing_pipeline.join()