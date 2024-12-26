import datetime
import os

import base64
import torch
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor
from lang_trans.arabic import buckwalter

from audio_utils import save_audio_to_file

from .asr_interface import ASRInterface
import numpy as np
import librosa


class CustomASR(ASRInterface):
    def __init__(self, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # model_name = kwargs.get("model_name", "tarteel-ai/whisper-base-ar-quran")
        # self.asr_pipeline = pipeline(
        #     "automatic-speech-recognition",
        #     model=model_name,
        #     device=device,
        # )

    async def transcribe(self, client):
        file_path = await save_audio_to_file(
            client.scratch_buffer, client.get_file_name()
        )
        loaded_model = Wav2Vec2ForCTC.from_pretrained("IbrahimSalah/Wav2vecXXl_quran_syllables").eval()
        loaded_processor = Wav2Vec2Processor.from_pretrained("IbrahimSalah/Wav2vecXXl_quran_syllables")
        
        # convert audio to NDarray[float64]
        audio_input, _ = librosa.load(file_path, sr=16000)
        
        inputs = loaded_processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            predicted = torch.argmax(loaded_model(inputs.input_values).logits, dim=-1)
        predicted[predicted == -100] = loaded_processor.tokenizer.pad_token_id  # see fine-tuning script
        pred_1 = loaded_processor.tokenizer.batch_decode(predicted)[0]
        to_return = buckwalter.untrans(pred_1)
        

        # if client.config["language"] is not None:
        #     to_return = self.asr_pipeline(
        #         file_path,
        #         generate_kwargs={"language": client.config["language"]},
        #     )["text"]
        # else:
        #     to_return = self.asr_pipeline(file_path)["text"]

        # to_return = self.asr_pipeline(file_path)["text"]
        # os.remove(file_path)
        
        if(await self.is_noise(to_return, file_path)):
            with open(file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return {"text": ' ', "audio": audio_base64}
        
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {"text": to_return, "audio": audio_base64}
    
        to_return = {
            # "language": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
            # "language_probability": None,
            "text": to_return.strip(),
            # "words": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
        }
        return to_return

    async def log_reg(self, transcription, audio_file_path):
        log_dir = "audio_transcription"
        os.makedirs(log_dir, exist_ok=True)
        log_file_name = os.path.basename(audio_file_path)
        log_file_path = os.path.join(log_dir, log_file_name + ".txt")
        
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(transcription.strip())
            
    async def is_noise(self, transcription, audio_file_path):
        if(transcription == 'وَالْمُؤْمِنِينَ'):
            noise_dir = "noise"
            os.makedirs(noise_dir, exist_ok=True)
            log_file_name = os.path.basename(audio_file_path)
            noise_file_path = os.path.join(noise_dir, "noise_entries.log")
            log_entry = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_file_name}\n"
            with open(noise_file_path, "a", encoding="utf-8") as noise_log_file:
                noise_log_file.write(log_entry)
            return True
        else:
            await self.log_reg(transcription, audio_file_path)
            return False
        