"""
Description: This script is responsible for audio playback.
"""

import threading
import io

from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import wave

class AudioData:
    def __init__(self) -> None:
        self._audio_data = None

    def _store_chunk(self, data: bytes) -> None:
        self._audio_data = self._process_chunk(data)

    def _store_chunks(self, data: list[bytes]) -> None:
        audio_data = None
        silence = AudioSegment.silent(duration=500)
        
        for chunk in data:
            if audio_data is None:
                audio_data = self._process_chunk(chunk)
            else:
                audio_data += silence + self._process_chunk(chunk)

        self._audio_data = audio_data

    def _process_chunk(self, data: bytes) -> AudioSegment:
        if data.startswith(b'RIFF'): # Handle wave audio
            with io.BytesIO(data) as bio:
                with wave.open(bio, 'rb') as wave_file:
                    sample_width = wave_file.getsampwidth()
                    channels = wave_file.getnchannels()
                    framerate = wave_file.getframerate()
                        
                    audio_data = wave_file.readframes(wave_file.getnframes())
                        
            return AudioSegment(
                data=audio_data,
                sample_width=sample_width,
                frame_rate=framerate,
                channels=channels
            )
        else: # Handle mp3 audio
            return AudioSegment.from_file(
                io.BytesIO(data),
                format='mp3'
            )

class AudioPlayer:
    def __init__(self) -> None:
        """
        The audio player. Playback can be interrupted.
        """
        self._current_playback = None
        self._stop_event = threading.Event()

    def play_audio(self, audio_data: AudioData) -> None:
        self._player_thread = threading.Thread(target=self._player, daemon=True, args=(audio_data, ))
        self._player_thread.start()

    def _player(self, audio_data: AudioData) -> None:
        """
        Plays the audio data
        """
        self._current_playback = _play_with_simpleaudio(audio_data._audio_data)
                
        # Wait for playback to finish or stop event
        while self._current_playback.is_playing() and not self._stop_event.is_set():
            threading.Event().wait(0.1)
                    
        # Stop playback if still playing
        if self._current_playback.is_playing():
            self._current_playback.stop()
                
        self._current_playback = None

    def stop(self) -> None:
        """
        Interrupt the current playback.
        """
        if self._player_thread:
            self._stop_event.set()
            if self._current_playback:
                self._current_playback.stop()

            self._player_thread.join()
            self._player_thread = None
            self._current_playback = None

    def is_playing(self) -> bool:
        return self._player_thread is not None