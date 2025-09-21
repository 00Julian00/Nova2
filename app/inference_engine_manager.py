"""
Description: Manages imports of inference engines,
so requirements for an inference engine that is not used to not have to be installed.
"""
from typing import Literal
from pathlib import Path
import importlib.util

from Nova2.app.helpers import Singleton
from Nova2.app.interfaces import LLMInferenceEngineBase, STTInferenceEngineBase, TTSInferenceEngineBase

class InferenceEngineManager(Singleton):
    def __init__(self) -> None:
        self._loaded_stt_engines: list[STTInferenceEngineBase] = None # type: ignore
        self._loaded_llm_engines: list[LLMInferenceEngineBase] = None # type: ignore
        self._loaded_tts_engines: list[TTSInferenceEngineBase] = None # type: ignore

        self._stt_engines_dir = Path(__file__).parent.parent / "inference_engines" / "inference_stt"
        self._llm_engines_dir = Path(__file__).parent.parent / "inference_engines" / "inference_llm"
        self._tts_engines_dir = Path(__file__).parent.parent / "inference_engines" / "inference_tts"

    def request_engine(self, name: str, eng_type: Literal["STT", "LLM", "TTS"]) -> LLMInferenceEngineBase | STTInferenceEngineBase | TTSInferenceEngineBase: # type: ignore
        """
        Attempts to find and import the specified inference engine.
        Will throw an exception, if the engine could not be found, or failed to be imported.
        """
        path = ""
        base_class = None
        class_instance = None

        match eng_type:
            case "STT":
                path = self._stt_engines_dir
                base_class = STTInferenceEngineBase
            case "LLM":
                path = self._llm_engines_dir
                base_class = LLMInferenceEngineBase
            case "TTS":
                path = self._tts_engines_dir
                base_class = TTSInferenceEngineBase

        for engine in path.iterdir():
            if engine.stem == name:
                try:
                    spec = importlib.util.spec_from_file_location(engine.stem, engine)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        classes = [getattr(module, name) for name in dir(module) if isinstance(getattr(module, name), type)]

                        for cls in classes:
                            if issubclass(cls, base_class) and cls != base_class: # type: ignore
                                if class_instance != None:
                                    raise Exception(f"More than one class found that inherits from an engine base class in engine {name}. Only one class can inherit from an engine base class.")

                                class_instance = cls()

                        if not class_instance:
                            raise Exception(f"Engine {name} either contains no classes, or no class that inherits from the correct engine base class.")

                        return class_instance # type: ignore
                    else:
                        raise Exception("Failed to create spec from file.")
                except Exception as e:
                    raise ImportError(f"Failed to import engine {name}. Reason: {e}")
        raise FileExistsError(f"Engine {name} could not be found in directory {path}")
