from pydub import AudioSegment
import io

import wave

from Nova2.app.interfaces import AudioDataBase

class AudioData(AudioDataBase):
    def __init__(self, data: bytes | list[bytes]) -> None:
        if type(data) == bytes:
            self._store_audio([data])
        else:
            self._store_audio(data) # type: ignore

    def _store_audio(self, data: list[bytes]) -> None:
        data_full = b''.join(data)

        self._audio_data = self._process_audio(data_full)

    def _process_audio(self, data: bytes) -> AudioSegment:
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