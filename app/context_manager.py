"""
Description: This script collects all data provided to the system and stores them in long and short term context memory.
"""

import threading

from .transcriptor import VoiceAnalysis, VoiceProcessingHelpers
from .context_data_manager import ContextDataManager
from .database_manager import VoiceDatabaseManager

class ContextManager:
    def __init__(
            self,
            voice_analysis: VoiceAnalysis = None,
            ) -> None:

        self._voice_analysis = voice_analysis
        self._context_data_manager = ContextDataManager()
        self._voice_database_manager = VoiceDatabaseManager()
        self._managing_pipeline = threading.Thread(target=self._manage_context)

    def _manage_context(self) -> None:
        last_sentence = ""
        has_sentence_been_finished = False

        for sentence in self._voice_analysis:
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
                
                voice = self._voice_database_manager.get_voice_from_tensor(average_embedding)

                if voice is not None:
                    if voice[1] < 0.8:  # If the confidence score is too low
                        voice_name = self._voice_database_manager.create_unknown_voice(average_embedding)
                    else:
                        voice_name = voice[0]
                else:
                    voice_name = self._voice_database_manager.create_unknown_voice(average_embedding)

                self._context_data_manager.add_spoken_sentence_to_context(sentence_string, voice_name)
                has_sentence_been_finished = True

    def start(self) -> None:
        self._managing_pipeline.start()

    def close(self) -> None:
        if self._managing_pipeline.is_alive():
            self._managing_pipeline.join()

    def __del__(self) -> None:
        self.close()