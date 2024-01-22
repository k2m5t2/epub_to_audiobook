from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider
from audiobook_generator.core.utils import split_text, set_audio_tags

import onnxruntime
import soundfile as sf
from ttstokenizer import TTSTokenizer

import yaml


def get_supported_formats():
    return ["wav"]

class JLSpeechTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        self.config = config
        self.validate_config()
        with open("jlspeech-jets-onnx/config.yaml", "r", encoding="utf-8") as f:
            self.tts_config = yaml.safe_load(f)
        self.tokenizer = TTSTokenizer(self.tts_config["token"]["list"])
        self.model = onnxruntime.InferenceSession(
            "jlspeech-jets-onnx/model.onnx",
            providers=["CPUExecutionProvider"]
        )
        self.max_chars = 100

    def __str__(self) -> str:
        return f"{self.config}"

    def validate_config(self):
        # raise NotImplementedError
        pass

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        # inputs = 
        
        text_chunks = split_text(text, self.max_chars, self.config.language)
        
        wav_segments = []

        for i, chunk in enumerate(text_chunks, 1):
            outputs = self.model.run(None, {"text": self.tokenizer(chunk)})
            wav = outputs[0]
            wav_segments.append(wav)
        
        merged_wav = [val for wav_segment in wav_segments for val in wav_segment]
        # sf.write(output_file, *wav_segments, 22050)
        sf.write(output_file, merged_wav, 22050, format="wav")

        set_audio_tags(output_file, audio_tags)

    def estimate_cost(self, total_chars):
        return 0

    def get_break_string(self):
        # raise NotImplementedError
        return "   "
        # not sure about the purpose/function of this
        # nor the validity of this implementation, currently borrowed from openai_tts_provider.py

    def get_output_file_extension(self): # NOTE sets the default file extension when CLI arg is omitted, I think?
        return "wav"
    
