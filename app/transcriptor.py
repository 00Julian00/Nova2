"""
Description: This script uses fasterwhisper to continously transcribe audio data from the microphone. It also creates voice embeddings.
"""

import os
import queue
import threading
import time
from typing import Generator, List
import multiprocessing

from . import helpers

import langcodes
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from denoiser import pretrained
from denoiser.dsp import convert_audio
from faster_whisper import WhisperModel
with helpers.suppress_output():
    from speechbrain.inference.speaker import EncoderClassifier
import silero_vad

from .transcriptor_data import Word, TranscriptorConditioning
from .database_manager import VoiceDatabaseManager
from .context_data import *

SAMPLE_RATE = 16000

class VoiceAnalysis:
    def __init__(self) -> None:
        """
        A pipeline for live voice analysis.
        """
        self._conditioning = None

        self._conditioning_dirty = None

        if os.name == "nt":
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # Only necessary for windows

        self._voice_database_manager = VoiceDatabaseManager()

    def configure(self, conditioning: TranscriptorConditioning):
        self._conditioning_dirty = conditioning

    def apply_config(self) -> None:
        if self._conditioning_dirty is None:
            raise Exception("Failed to initialize TTS. No TTS conditioning provided.")

        self._conditioning = self._conditioning_dirty

        self._verbose = False

        self._device = self._conditioning.device
        if self._conditioning.device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
        self._log(f"Using device '{self._device}'.")
        torch.set_default_dtype(torch.float32)

        with helpers.suppress_output():
            self._model = WhisperModel(
                model_size_or_path=self._conditioning.model,
                device=self._device,
                compute_type="float32",
                cpu_threads=multiprocessing.cpu_count()
                )
        self._log(f"Whisper model of size '{self._conditioning.model}' loaded.")
        # Check if the language is valid
        if self._conditioning.language is not None:
            try:
                langcodes.Language.get(self._conditioning.language)
            except:
                raise ValueError(f"{self._conditioning.language} is not a valid language code.")
            self._log(f"Language set to '{self._conditioning.language}'.")
        else:
            self._log("Language set to auto.")
        self._language = self._conditioning.language
        
        if self._conditioning.voice_boost != 0.0:
            self._denoise_model = pretrained.dns64()
            self._denoise_model.eval()
            self._denoise_model = self._denoise_model.to(self._device)
            self._log("Denoiser model loaded.")
        else:
            self._log("voice_boost is set to 0, skipping denoiser model loading.")
        
        self._vad_model = silero_vad.load_silero_vad()
        self._log("VAD model loaded.")

        with helpers.suppress_output():
            self._speaker_embedding_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={"device": self._device})
        self._log("Speaker embedding model loaded.")

        self._microphone_index = self._conditioning.microphone_index
        self._max_silence_chunks = 3
        self._current_sentence = []
        self._locked_words = 0
        self._audio_queue = queue.Queue()
        self._is_recording = True
        self._voice_boost = self._conditioning.voice_boost
        self._speculative = False
        self._recording_thread = threading.Thread(target=self._record_audio)

        self._log("Initialization complete.\n")

    def _record_audio(self) -> None:
        audio_buffer = np.array([], dtype=np.float32)  
        silence_start = None
        last_transcription_time = time.time()

        def callback(indata, frames, time_info, status):
            nonlocal audio_buffer, silence_start, last_transcription_time
            audio = np.frombuffer(indata, dtype=np.float32)  
            audio_buffer = np.concatenate((audio_buffer, audio))

            if time.time() - last_transcription_time >= 1:
                last_transcription_time = time.time()
                if len(audio_buffer) > 0:
                        self._audio_queue.put(audio_buffer)
                        audio_buffer = np.array([], dtype=np.float32)  

        with sd.InputStream(callback=callback, dtype=np.float32, channels=1, samplerate=SAMPLE_RATE, device=self._microphone_index):
            while self._is_recording:
                time.sleep(0.1)

        if len(audio_buffer) > 0:
            self._audio_queue.put(audio_buffer)

    def _boost_speech(self, audio_data: torch.FloatTensor) -> torch.FloatTensor:
        if self._voice_boost == 0.0:
            return audio_data

        audio_tensor = torch.from_numpy(audio_data).float().to(self._device)
        audio_tensor = audio_tensor.unsqueeze(0)
        audio = convert_audio(audio_tensor, SAMPLE_RATE, self._denoise_model.sample_rate, self._denoise_model.chin)

        with torch.no_grad():
            denoised = self._denoise_model(audio)[0]

        denoised = denoised * self._voice_boost
        audio_tensor = audio_tensor + denoised
        audio_tensor = audio_tensor / audio_tensor.abs().max()
        audio_tensor = audio_tensor.squeeze(0)
        audio_tensor = audio_tensor.cpu()

        return audio_tensor.numpy()

    def _detect_voice_activity(self, audio_chunk: torch.FloatTensor) -> bool:        
        timestamps = silero_vad.get_speech_timestamps(
            model=self._vad_model,
            audio=audio_chunk,
            threshold=self._conditioning.vad_threshold,
            sampling_rate=SAMPLE_RATE
        )

        return len(timestamps) > 0

    def _transcribe(self, audio_tensor: torch.FloatTensor) -> List[Word]:
        if self._language is not None:
            segments, info = self._model.transcribe(audio_tensor.cpu().numpy(), beam_size=5, language=self._language, condition_on_previous_text=False, word_timestamps=True)
        else:
            segments, info = self._model.transcribe(audio_tensor.cpu().numpy(), beam_size=5, condition_on_previous_text=False, word_timestamps=True) # Leave the language undefined so whisper autodetects it
        transcription = []
        for segment in segments:
            for word in segment.words:
                transcription.append(Word(text=word.word, start=word.start, end=word.end))
        return transcription

    def _update_transcription(self, words: List[Word]) -> tuple[List[Word], int]:
        new_locked_words = self._locked_words
        for i in range(len(words)):
            if i < self._locked_words:
                continue
            if i < len(self._current_sentence) and words[i].text == self._current_sentence[i].text:
                new_locked_words += 1
            else:
                break
        
        self._current_sentence = words[:new_locked_words] + words[new_locked_words:]
        self._locked_words = new_locked_words
        
        return self._current_sentence, self._locked_words
    
    def _generate_speaker_embedding(self, audio_data: torch.FloatTensor, start: float, end: float) -> torch.FloatTensor:
        if audio_data.ndim == 1:
            audio_data = audio_data.unsqueeze(0)

        audio_data = self._split_audio_by_timestamps(audio_data=audio_data, start=start, end=end)

        # Ensure the audio segment is long enough (at least 1 second)
        min_length = SAMPLE_RATE  # 1 second at 16000 Hz
        if audio_data.shape[1] < min_length:
            # Pad the audio data if it's too short
            padding = torch.zeros(1, min_length - audio_data.shape[1], device=audio_data.device)
            audio_data = torch.cat([audio_data, padding], dim=1)

        try:
            return self._speaker_embedding_model.encode_batch(audio_data)
        except RuntimeError as e:
            self._log(f"Error generating speaker embedding: {str(e)}")
            return torch.zeros(1, 512, device=self._device)  # Return a zero embedding as a fallback

    
    def start(self) -> Generator[ContextDatapoint, None, None]:
        """
        Generator for live voice analysis.

        Returns a generator object that continuously yields the current sentence that is recorded from the microphone.
        if "speculative" is set to True, the generator may correct itself in the next pass, if not, it yields the words as they are coming in.
        When a sentence is finished, the generator yields the full sentence, until the user continues speaking, which will reset the sentence.

        Returns:
            Generator[ContextDatapoint]: The generator that yields the data.
        """
        self._recording_thread.start()

        current_audio_data = None
        first_audio_chunk = None
        silence_counter = 0

        confirmed_transcription = ""
        speculative_transcription = ""

        while self._is_recording:
            if not self._audio_queue.empty():
                audio_chunk = self._audio_queue.get()
                audio_chunk = self._boost_speech(audio_chunk)
                speech_detected = self._detect_voice_activity(audio_chunk)
                if current_audio_data is None:
                    if not speech_detected:
                        first_audio_chunk = audio_chunk
                        continue
                else:
                    if not speech_detected:
                        silence_counter += 1
                        current_audio_data = np.concatenate((current_audio_data, audio_chunk))
                        if silence_counter >= self._max_silence_chunks:
                            continue
                
                if speech_detected:
                    silence_counter = 0

                if current_audio_data is None:
                    if first_audio_chunk is not None:
                        current_audio_data = np.concatenate((first_audio_chunk, audio_chunk))
                    else:
                        current_audio_data = audio_chunk
                else:
                    current_audio_data = np.concatenate((current_audio_data, audio_chunk))

                audio_tensor = torch.from_numpy(current_audio_data).float().to(self._device)
                
                transcription = self._transcribe(audio_tensor)

                self._current_sentence, self._locked_words = self._update_transcription(transcription)

                confirmed_transcription = []
                speculative_transcription = []
                for i, word in enumerate(self._current_sentence):
                    self._current_sentence[i].speaker_embedding = self._generate_speaker_embedding(audio_tensor, self._current_sentence[i].start, self._current_sentence[i].end)
                    if i < self._locked_words:
                        confirmed_transcription.append(word)

                    speculative_transcription.append(word)

                # Sentence is finished
                if len(confirmed_transcription) > 0:
                    if "." in confirmed_transcription[len(confirmed_transcription) - 1].text or "!" in confirmed_transcription[len(confirmed_transcription) - 1].text or "?" in confirmed_transcription[len(confirmed_transcription) - 1].text:
                        current_audio_data = None
                        self._current_sentence = []
                        self._locked_words = 0

                        # Construct the context datapoint
                        voice = self.resolve_speaker(words=confirmed_transcription)
                        datapoint = ContextDatapoint(
                            source=ContextSource_Voice(speaker=voice),
                            content=VoiceProcessingHelpers.word_array_to_string(confirmed_transcription)
                            )
                        
                        yield datapoint

    def resolve_speaker(self, words: List[Word]) -> str:
        """
        Finds the name of the current speaker 
        """
        embedding_list = []
        for word in words:
            embedding_list.append(word.speaker_embedding)

        avg_embedding = VoiceProcessingHelpers.take_average_embedding(embedding_list)

        self._voice_database_manager.open()

        voice = self._voice_database_manager.get_voice_name_from_embedding(avg_embedding)

        if voice and voice[1] > 0.8: # If voice was found and it's close enough use it. Otherwise create a new one
            voice_name = voice[0]
        else:
            voice_name = self._voice_database_manager.create_unknown_voice(avg_embedding)

        self._voice_database_manager.close()

        return voice_name
        
    def close(self) -> None:
        """
        Ends the execution of this script. No more data will be yielded by the generator.
        """
        self._is_recording = False
        self._audio_queue = queue.Queue()
        if self._recording_thread.is_alive():
            self._recording_thread.join()

    def _split_audio_by_timestamps(self, audio_data: torch.FloatTensor, start: float, end: float) -> torch.FloatTensor:
        start_sample = int(start * SAMPLE_RATE)
        end_sample = int(end * SAMPLE_RATE)

        audio_data = audio_data[:, start_sample:end_sample]

        return audio_data
    
    def _log(self, text: str) -> None: # Used to print debug information if verbose is set to True. Reduces if statements in the code.
        if self._verbose:
            print(text)

class VoiceProcessingHelpers:
    def __init__(self):
        """
        A collection of static methods that act as helpers.
        """
        pass
    
    @staticmethod
    def word_array_to_string(word_array: List[Word]) -> str:
        """
        Extracts the text from an array of word objects and returns it as a string.

        Arguments:
            word_array (List[Word]): The array of word objects that will be converted to a string.

        Returns:
            str: The full extracted text.
        """
        text = ""
        for word in word_array:
            text += word.text
        return text
        
    @staticmethod
    def compare_embeddings(emb1: torch.FloatTensor, emb2: torch.FloatTensor) -> float:
        """
        Compare how "close" two embeddings are to each other.

        Arguments:
            emb1 (torch.FloatTensor): The first embedding to compare.
            emb2 (torch.FloatTensor): The second embedding to compare.

        Returns:
            float: How "close" the embeddings are. Ranges from -1 to 1. Higher is closer.
        """
        return (F.cosine_similarity(emb1.squeeze(), emb2.squeeze(), dim=0).mean().item())
    
    @staticmethod
    def take_average_embedding(embeddings: List[torch.FloatTensor]) -> torch.FloatTensor:
        """
        Takes the average of a List of embeddings.

        Arguments:
            embeddings (List[torch.FloatTensor]): The embedding List.

        Returns:
            torch.FloatTensor: The average embedding.
        """
        return torch.mean(torch.stack(embeddings), dim=0)