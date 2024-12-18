"""
Description: This script handles text-to-speech.
"""

import threading
import io
import queue
import time

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

from .security import SecretsManager
from .tool_manager import ToolManager

#TODO: Make it actually play the streaming audio instead of collecting it first
class AudioPlayer:
    def __init__(self):
        """
        The audio player. Playback can be interrupted.
        """
        self.playing = False
        self.current_playback = None
        self.stop_event = threading.Event()
    
    def play_audio_stream(self, audio_stream: enumerate) -> None:
        self._play_thread = threading.Thread(target=self._play, args=(audio_stream,))
        self._play_thread.start()
    
    def _play(self, audio_stream: enumerate) -> None:
        self.playing = True
        
        # Read stream data
        buffer = io.BytesIO()
        for chunk in audio_stream:
            if self.stop_event.is_set():
                return
            buffer.write(chunk)
        
        buffer.seek(0)
        
        # Convert MP3 to AudioSegment
        audio_segment = AudioSegment.from_mp3(buffer)
        
        # Reset stop event
        self.stop_event.clear()
        
        # Convert to wav
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

        self.playing = False

class TTSManager:
    def __init__(self):
        """
        This class manages the interaction with the elevenlabs API to generate speech from text.
        """
        self._key_manager = SecretsManager()
        self._tool_manager = ToolManager()

        key = self._key_manager.get_secret("elevenlabs_api_key")

        if not key:
            raise ValueError("Elevenlabs API key not found")

        self._elevenlabs_client = ElevenLabs(
            api_key=key
        )

        self._audio_player = AudioPlayer()

        self._audio_queue = queue.Queue()

        threading.Thread(target=self._play_from_queue).start()

    def _play_from_queue(self) -> None:
        while True:
            if not self._audio_queue.empty():
                stream = self._audio_queue.get()
                self._audio_player.play_audio_stream(audio_stream=stream)

            while self._audio_player.playing:
                time.sleep(0.1)

    def speak(self, text: str) -> None:
        """
        Convert text to speech and play the generated audio.

        Arguments:
            text (str): The text that will be converted to speech.
        """
        audio_stream = self._elevenlabs_client.generate(
            text = text,
            voice=Voice(
                voice_id="1wp9zlfEyG5CpejHSr4V",
                settings=VoiceSettings(stability=0.6, similarity_boost=1, style=0.2, use_speaker_boost=True)
            ),
            model = "eleven_turbo_v2_5",
            stream = True
        )

        self._audio_queue.put(audio_stream)


    def interrupt(self) -> None:
        """
        Stop the current playback of the speech.
        """
        self._audio_player.stop()
        self._audio_player = AudioPlayer()
        self._audio_queue.queue.clear()