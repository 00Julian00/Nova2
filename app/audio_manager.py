"""
Description: This script is responsible for audio playback.
"""

import threading

from pydub.playback import _play_with_simpleaudio

from .audio_data import AudioData
        
class AudioPlayer:
    def __init__(self) -> None:
        """
        The audio player. Playback can be interrupted.
        """
        self._current_playback = None
        self._stop_event = threading.Event()

    def play_audio(self, audio_data: AudioData) -> None:
        self._current_playback = 0 # Make sure is_playing() is true
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
        return self._current_playback is not None