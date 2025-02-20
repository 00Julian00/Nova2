from pathlib import Path
import io

import torch
import torchaudio
import numpy as np

from ...zonos.model import Zonos
from ...zonos.conditioning import make_cond_dict
from .inference_base_tts import InferenceEngineBaseTTS
from ...tts_data import TTSConditioning

import sounddevice as sd

class InferenceEngineZonos(InferenceEngineBaseTTS):
    def __init__(self) -> None:
        """
        This class provides the interface to run inference via Zonos.
        """

        self._model = None
        self._voice_files = Path(__file__).resolve().parent.parent.parent / "data" / "voices"
        self._device = "cuda"

        super().__init__()

        self.is_local = True

    def initialize_model(self, model: str = "Zyphra/Zonos-v0.1-transformer") -> None:
        self._model_name = model

        self._model = Zonos.from_pretrained(model, device="cuda").bfloat16()

        self._device = "cuda"

    def clone_voice(self, audio_dir: str, name: str) -> None:
        """
        Create a new voice embedding from a recording.

        Arguments:
            audio_dir (str): The directory of the audio file containing the voice.
            name (str): The name under shich the voice should be saved.
        """

        path = Path(audio_dir)

        if not path.exists():
            raise FileNotFoundError(f"File {audio_dir} was not found.")
        
        wav, sampling_rate = torchaudio.load(audio_dir)

        wav = wav.to(self._device)

        embedding = self._model.make_speaker_embedding(wav, sampling_rate)

        embedding = embedding.cpu().float().numpy()

        target_dir = self._voice_files / f"{name}.npy"

        np.save(str(target_dir), embedding)

    def _get_voice(self, name: str) -> torch.FloatTensor:
        """
        Get the voice embedding for the specified voice.

        Arguments:
            name (str): The name of the voice to be loaded.

        Returns:
            torch.FloatTensor: The voice embedding.
        """

        target_dir = self._voice_files / f"{name}.npy"

        if not target_dir.exists():
            raise FileNotFoundError(f"Voice {name} was not found.")
        
        embedding = torch.from_numpy(np.load(str(target_dir)))
        embedding = embedding.to(torch.bfloat16)
        embedding = embedding.to(device=self._device)

        return embedding

    def is_model_ready(self) -> bool:
        return self._model != None
    
    def get_current_model(self) -> str:
        return self._model_name
    
    def free(self) -> None:
        self._model = None

    def run_inference(self, text: str, conditioning: TTSConditioning, stream: bool = False) -> bytes:
        if not conditioning.emotion:
            # Defaults to happy
            emotion = torch.tensor([[0.5000, 0.0100, 0.0100, 0.0100, 0.2000, 0.0100, 0.1600, 0.1000]], device=self._device)
        else:
            emotion = conditioning.emotion.to(self._device)

        cond = make_cond_dict(
            text=text,
            speaker=self._get_voice(conditioning.voice),
            language=conditioning.language,
            emotion=emotion,
            pitch_std=conditioning.expressivness,
            speaking_rate=conditioning.speaking_rate
        )

        cond = self._model.prepare_conditioning(cond, device="cuda")

        audio = self._model.generate(
            cond,
            max_new_tokens=90 * 30,
            cfg_scale=conditioning.stability
        )

        wavs = self._model.autoencoder.decode(audio).cpu()

        if wavs.dim() == 3:
            wavs = wavs.squeeze(0)

        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            wavs,
            self._model.autoencoder.sampling_rate,
            format="wav",
            encoding="PCM_S",
            bits_per_sample=16,
            backend="soundfile"
        )

        wav_bytes = buffer.getvalue()
        buffer.close()
        return [wav_bytes]