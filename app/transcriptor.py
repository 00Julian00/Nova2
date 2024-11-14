"""
Description: This script uses fasterwhisper to continously transcribe audio data from the microphone. It also creates voice embeddings.
"""

import os
import queue
import threading
import time
from typing import Literal, List, Generator

import langcodes
import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from denoiser import pretrained
from denoiser.dsp import convert_audio
from faster_whisper import WhisperModel
from scipy.io import wavfile
from speechbrain.inference.speaker import EncoderClassifier

SAMPLE_RATE = 16000

class Word:
    def __init__(
            self,
            text: str = "",
            start: float = 0,
            end: float = 0,
            speaker_embedding: torch.FloatTensor = None
            ) -> None:
        
        """
        A class to represent a word in a transcription.

        This class holds a singular word from a transcription, together with other relevant information.

        Contents:
            text (str, optional): The text of the word. Defaults to "".
            start (float, optional): The start time of the word in seconds. Defaults to 0.
            end (float, optional): The end time of the word in seconds. Defaults to 0.
            speaker_embedding (torch.FloatTensor, optional): The speaker embedding of the voice that said the word. Defaults to None.
        """

        self.text = text
        self.start = start
        self.end = end
        self.speaker_embedding = speaker_embedding

class VoiceAnalysis:
    if os.name == "nt":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # Only necessary for windows

    def __init__(
                self,
                microphone_index: int = 0,
                speculative: bool = True,
                whisper_model: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2"] = "large-v3",
                device: Literal["cpu", "cuda"] = "cuda",
                voice_boost: float = 10.0,
                language: str = None,
                verbose: bool = True
                ) -> None:
        
        """
        A pipeline for live voice analysis.

        Arguments:
            microphone_index (int): The index of the microphone to use for recording.
            speculative (bool, optional): Whether to provide unconfirmed results. Allows you to access more of the transcription earlier, but the data may be inaccurate and is subject to change in the next pass. Defaults to True.
            whisper_model ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2", optional): The Whisper model to use. Defaults to "large-v3".
            device ("cpu", "cuda", optional): The device to use for the computations. Defaults to "cuda" or "cpu" if cuda is not available.
            voice_boost (float, optional): How much to boost the voice in the audio preprocessing stage. Setting it to 0 disables this feature. Defaults to 10.0.
            language (str, optional): The language to use for the transcription. Must be a valid language code. If None, the language will be autodetected. If possible, the language should be set to improve accuracy. Defaults to None.
            verbose (bool, optional): Whether to print debug information. Defaults to True.
        """

        self._verbose = verbose

        self._device = device
        if device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
        self._log(f"Using device '{self._device}'.")
        torch.set_default_dtype(torch.float32)

        self._model = WhisperModel(whisper_model, device=self._device, compute_type="float32")
        self._log(f"Whisper model of size '{whisper_model}' loaded.")
        # Check if the language is valid
        if language is not None:
            try:
                langcodes.Language.get(language)
            except:
                raise ValueError(f"{language} is not a valid language code.")
            self._log(f"Language set to '{language}'.")
        else:
            self._log("Language set to auto.")
        self._language = language
        
        if voice_boost != 0.0:
            self._denoise_model = pretrained.dns64()
            self._denoise_model.eval()
            self._denoise_model = self._denoise_model.to(self._device)
            self._log("Denoiser model loaded.")
        else:
            self._log("voice_boost is set to 0, skipping denoiser model loading.")
        
        self._vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True, verbose=False)
        self._vad_model = self._vad_model.to(self._device)
        (self._get_speech_ts, _, self._read_audio, self._VADIterator, self._collect_chunks) = utils
        self._vad_iterator = self._VADIterator(self._vad_model)
        self._log("VAD model loaded.")

        self._speaker_embedding_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={"device": self._device})
        self._log("Speaker embedding model loaded.")

        self._microphone_index = microphone_index
        self._max_silence_chunks = 3
        self._current_sentence = []
        self._locked_words = 0
        self._audio_queue = queue.Queue()
        self._is_recording = True
        self._voice_boost = voice_boost
        self._speculative = speculative
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
    
    def _detect_voice_activity(self, audio_chunk: torch.FloatTensor) -> tuple[bool, list[float]]:
        if type(audio_chunk) == np.ndarray:
            audio_tensor = torch.from_numpy(audio_chunk).float().to(self._device)
        else:
            audio_tensor = audio_chunk.float().to(self._device)
        
        min_audio_length = int(16000 / 31.25)
        if audio_tensor.shape[0] < min_audio_length:
            padding = torch.zeros(min_audio_length - audio_tensor.shape[0], device=self._device)
            audio_tensor = torch.cat([audio_tensor, padding])
        
        speech_probs = []
        for i in range(0, len(audio_tensor), 512):
            window = audio_tensor[i:i+512]
            if len(window) == 512:
                speech_prob = self._vad_model(window, 16000).item()
                speech_probs.append(speech_prob)

        speech_detected = any(prob > 0.95 for prob in speech_probs)
        return speech_detected, speech_probs
    
    def _transcribe(self, audio_tensor: torch.FloatTensor) -> list[Word]:
        if self._language is not None:
            segments, info = self._model.transcribe(audio_tensor.cpu().numpy(), beam_size=5, language=self._language, condition_on_previous_text=False, word_timestamps=True)
        else:
            segments, info = self._model.transcribe(audio_tensor.cpu().numpy(), beam_size=5, condition_on_previous_text=False, word_timestamps=True) # Leave the language undefined so whisper autodetects it
        transcription = []
        for segment in segments:
            for word in segment.words:
                transcription.append(Word(text=word.word, start=word.start, end=word.end))
        return transcription

    def _update_transcription(self, words: list[Word]) -> tuple[list[Word], int]:
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

    
    def start(self) -> Generator[List[Word], None, None]:
        """
        Generator for live voice analysis.

        Returns a generator object that continuously yields the current sentence that is recorded from the microphone.
        if "speculative" is set to True, the generator may correct itself in the next pass, if not, it yields the words as they are coming in.
        When a sentence is finished, the generator yields the full sentence, until the user continues speaking, which will reset the sentence.
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
                speech_detected, probabilities = self._detect_voice_activity(audio_chunk)

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

                if len(confirmed_transcription) > 0:
                    if "." in confirmed_transcription[len(confirmed_transcription) - 1].text or "!" in confirmed_transcription[len(confirmed_transcription) - 1].text or "?" in confirmed_transcription[len(confirmed_transcription) - 1].text:
                        current_audio_data = None
                        self._current_sentence = []
                        self._locked_words = 0
            
            if self._speculative:
                yield speculative_transcription
            else:
                yield confirmed_transcription

            time.sleep(0.1)
        
    def close(self) -> None:
        self._is_recording = False
        self._audio_queue = queue.Queue()
        if self._recording_thread.is_alive():
            self._recording_thread.join()

    def transcribe_from_file(self, file_path: str) -> List[Word]:
        startTime = time.time()
        audioData = self._load_from_wav(file_path).squeeze(0)
        if self._detect_voice_activity(audioData):
            transcription = self._transcribe(audioData)
            self._log(f"File transcribed in {round((time.time() - startTime) * 100) / 100} seconds.")
            return transcription
        else:
            self._log("No speech detected in the audio file.")
            return []

    def _split_audio_by_timestamps(self, audio_data: torch.FloatTensor, start: float, end: float) -> torch.FloatTensor:
        start_sample = int(start * SAMPLE_RATE)
        end_sample = int(end * SAMPLE_RATE)

        audio_data = audio_data[:, start_sample:end_sample]

        return audio_data
    
    def _log(self, text: str) -> None: # Used to print debug information if verbose is set to True. Reduces if statements in the code.
        if self._verbose:
            print(text)
    
    def __del__(self) -> None:
        self.close()

class VoiceProcessingHelpers:
    @staticmethod
    def word_array_to_string(word_array: List[Word]) -> str:
        """
        Extracts the text from an array of word objects and returns it as a string.
        """
        text = ""
        for word in word_array:
            text += word.text
        return text
        
    @staticmethod
    def compare_embeddings(emb1: torch.FloatTensor, emb2: torch.FloatTensor) -> float:
        """
        Returns a value between 0 and 1. Higher is more similar.
        """
        return (F.cosine_similarity(emb1.squeeze(), emb2.squeeze(), dim=0).mean().item() + 1) / 2.0
    
    @staticmethod
    def take_average_embedding(embeddings: List[torch.FloatTensor]) -> torch.FloatTensor:
        """
        Takes the average of a list of embeddings.
        """
        return torch.mean(torch.stack(embeddings), dim=0)

    @staticmethod
    def save_to_wav(file_path: str, audio_data: torch.FloatTensor, sample_rate: int) -> None:
        audio_data = audio_data.cpu()
        audio_numpy = audio_data.squeeze().numpy()
        audio_int16 = (audio_numpy * np.iinfo(np.int16).max).astype(np.int16)
        wavfile.write(file_path, sample_rate, audio_int16)

    @staticmethod
    def load_from_wav(file_path: str) -> torch.FloatTensor:
        sample_rate, audio = wavfile.read(file_path)
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        if sample_rate != SAMPLE_RATE:
            audio_data = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

        audio_data = torch.FloatTensor(audio_data)
        if audio_data.ndim == 1:
            audio_data = audio_data.unsqueeze(0)
        return audio_data