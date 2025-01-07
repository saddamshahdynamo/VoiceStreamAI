import datetime
import os

import base64
import torch
from transformers import pipeline

from audio_utils import save_audio_to_file
from file_paths import FilePaths

from .asr_interface import ASRInterface
import numpy as np
import librosa
from logger.data_logger_factory import DataLoggerFactory


class CustomASR(ASRInterface):
    def __init__(self, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = kwargs.get("model_name", "tarteel-ai/whisper-base-ar-quran")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
        )

    async def transcribe(self, client, partial = False):
        # filepath for the audio file with partial transcription in separate folder
        if(partial == False):
            # filepath for the audio file with full transcription
            file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name(), FilePaths.AUDIO_FILES_DIR)
        else:            
            file_path = await save_audio_to_file(client.partial_scratch_buffer, client.get_file_name_partial(), FilePaths.PARTIAL_AUDIO_DIR)
            
        transcription = self.asr_pipeline(file_path)["text"]
        # os.remove(file_path)
        
        # if(partial == False):
        if(await self.is_noise(transcription, file_path, partial)):
            with open(file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return self.prepare_return_object(audio_base64=audio_base64, probability=0.2)
           
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        # return self.prepare_return_object(transcription,audio_base64)
        
        if(partial == False):
            return self.prepare_return_object(transcription,audio_base64)
        else:
            return self.prepare_return_object(transcription,audio_base64,probability=0.2)
        # else:
        #     print("ASR_TRANSCRIBE: Partial transcription generated.")
        #     # os.remove(file_path)
        #     return self.prepare_return_object(transcription)
    
        to_return = {
            # "language": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
            # "language_probability": None,
            "text": to_return.strip(),
            # "words": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
        }
        return to_return

    async def log_reg(self, transcription, audio_file_path):
        log_dir = FilePaths.AUDIO_TRANSCRIPTION_DIR
        log_file_name = os.path.basename(audio_file_path)
        log_file_path = os.path.join(log_dir, log_file_name + ".txt")
        
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(transcription.strip())
            
    async def is_noise(self, transcription, audio_file_path, partial = False):
        if(transcription == 'وَالْمُؤْمِنِينَ'):
            noise_dir = FilePaths.NOISE_DIR
            log_file_name = os.path.basename(audio_file_path)
            if(partial == True):
                noise_file_path = os.path.join(noise_dir, "partial_noise_entries.log")
            else:
                noise_file_path = os.path.join(noise_dir, "noise_entries.log")
            log_entry = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_file_name}\n"
            with open(noise_file_path, "a", encoding="utf-8") as noise_log_file:
                noise_log_file.write(log_entry)
            return True
        else:
            await self.log_reg(transcription, audio_file_path)
            return False
    
    def prepare_return_object(self, transcription="", audio_base64="",probability=1):
        return {
            "text": transcription, 
            "audio": audio_base64,
            "words": [{
                "word": transcription,
                "probability": probability,
            }]
        }