"""
Description: This script handles text-to-speech.
"""

import threading
import io
import queue
import time
import wave

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

from .inference_engines import inference_zonos, inference_elevenlabs
from . import tts_data

#TODO: Make it actually play the streaming audio instead of collecting it first
class AudioPlayer:
    def __init__(self):
        """
        The audio player. Playback can be interrupted.
        """
        self.playing = False
        self.current_playback = None
        self.stop_event = threading.Event()

        self._audio_chunks_queue = queue.Queue()
    
    def play_audio_stream(self, audio_stream: enumerate) -> None:
        self.producer_thread = threading.Thread(target=self._producer, args=(audio_stream,), daemon=True)
        self.producer_thread.start()

        self.consumer_thread = threading.Thread(target=self._consumer, daemon=True)
        self.consumer_thread.start()
    
    def _producer(self, audio_stream: list[bytes]) -> None:
        for chunk in audio_stream:
            if self.stop_event.is_set():
                return

            if chunk.startswith(b'RIFF'): # Handle wave audio
                with io.BytesIO(chunk) as bio:
                    with wave.open(bio, 'rb') as wave_file:
                        sample_width = wave_file.getsampwidth()
                        channels = wave_file.getnchannels()
                        framerate = wave_file.getframerate()
                        
                        audio_data = wave_file.readframes(wave_file.getnframes())
                        
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=sample_width,
                    frame_rate=framerate,
                    channels=channels
                )
            else: # Handle mp3 audio
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(chunk),
                    format='mp3'
                )

            self._audio_chunks_queue.put(audio_segment)

    def _consumer(self) -> None:
        """
        Plays the audio data stored in the audio chunks queue
        """
        while True:
            audio_segment = self._audio_chunks_queue.get()

            self.playing = True

            self.current_playback = _play_with_simpleaudio(audio_segment)
                
            # Wait for playback to finish or stop event
            while self.current_playback.is_playing() and not self.stop_event.is_set():
                threading.Event().wait(0.1)
                    
            # Stop playback if still playing
            if self.current_playback.is_playing():
                self.current_playback.stop()
                    
            self.playing = False
            self.current_playback = None

    def stop(self):
        """
        Interrupt the current playback.
        """
        if self.playing:
            self.stop_event.set()
            if self.current_playback:
                self.current_playback.stop()

        self.producer_thread.join()

        self._audio_chunks_queue = queue.Queue()

        self.playing = False

class TTSManager:
    def __init__(self):
        """
        This class runs TTS inference and play the resulting audio.
        """

        self._audio_player = AudioPlayer()

        self._audio_queue = queue.Queue()

        self._inference_engine = inference_zonos.InferenceEngine()

        self._inference_engine.initialize_model()

        self._conditioning = tts_data.TTSConditioning(
            voice="Laura", #5Aahq892EEb6MdNwMM3p
            language="de",
            expressivness=100,
            stability=2,
        )

        threading.Thread(target=self._play_from_queue, daemon=False).start()

    def _play_from_queue(self) -> None:
        while True:
            if not self._audio_queue.empty():
                stream = self._audio_queue.get()
                self._audio_player.play_audio_stream(audio_stream=stream)

            while self._audio_player.playing:
                time.sleep(0.1)

            time.sleep(0.1)

    def speak(self, text: str) -> None:
        """
        Convert text to speech and play the generated audio.

        Arguments:
            text (str): The text that will be converted to speech.
        """

        stream = False # Streaming is too slow to be used

        audio_stream = self._inference_engine.run_inference(
                                                            conditioning=self._conditioning,
                                                            text=text,
                                                            stream=stream
                                                            )
        
        self._audio_queue.put(audio_stream)

    def interrupt(self) -> None:
        """
        Stop the current playback of the speech.
        """
        self._audio_player.stop()
        self._audio_queue.queue = queue.Queue()