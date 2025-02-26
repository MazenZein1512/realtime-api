import os
import json
import logging
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from django.core.files.storage import default_storage

import numpy as np
import io
import torchaudio

import transformers
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, EncoderDecoderCache

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000   # Whisper expects audio at 16 kHz
model_name = "YoussefAshmawy/Graduation_Project_Whisper_base"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="ar", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="ar", task="transcribe")
model.config.forced_decoder_ids = forced_decoder_ids

class AudioConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()
        self.audio_chunks = []
        self.transcription_buffer = io.BytesIO()
        self.transcription_text = ""
        self.transcription_task = asyncio.create_task(self.transcribe_periodically())

    async def disconnect(self, close_code):
        if self.transcription_task:
            self.transcription_task.cancel()
        if self.audio_chunks:
            await self.process_audio_chunks()

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            logger.info(f"Text data received: {text_data}")
            await self.send(text_data=text_data)

        if bytes_data:
            logger.info(f"Binary data received: {bytes_data[:50]}...")  # Log first 50 bytes for brevity
            self.audio_chunks.append(bytes_data)
            self.transcription_buffer.write(bytes_data)

    async def transcribe_periodically(self):
        while True:
            await asyncio.sleep(2)
            await self.process_audio_chunks()

    async def process_audio_chunks(self):
        if not self.audio_chunks:
            return

        try:
            # Combine audio chunks into a single byte sequence
            complete_audio_data = b"".join(self.audio_chunks)
            self.audio_chunks = []

            # Load audio data using torchaudio
            audio_buffer = io.BytesIO(complete_audio_data)
            try:
                waveform, sample_rate = torchaudio.load(audio_buffer, format="ogg")
            except Exception as e:
                logger.error(f"Error loading audio data: {e}")
                response_data = {
                    "message": "Error processing audio",
                    "error": "Malformed audio data or unsupported format",
                }
                await self.send(text_data=json.dumps(response_data))
                return

            if sample_rate != SAMPLING_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLING_RATE)
                waveform = resampler(waveform)

            # Transcribe the audio using Whisper
            transcription = transcribe_audio(waveform)
            logger.info(f"Transcription: {transcription}")

            # Append the new transcription to the existing transcription text
            self.transcription_text += transcription + " "

            response_data = {
                "message": "Transcription completed",
                "transcription": self.transcription_text.strip(),
            }
            await self.send(text_data=json.dumps(response_data))
            # Clear the transcription buffer
            self.transcription_buffer.seek(0)
            self.transcription_buffer.truncate(0)

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            response_data = {
                "message": "Error processing audio",
                "error": str(e),
            }
            await self.send(text_data=json.dumps(response_data))


def transcribe_audio(waveform):
    """
    Transcribe a single audio file
    
    Args:
        waveform (torch.Tensor): Audio waveform tensor
    
    Returns:
        str: Transcribed text
    """
    # Extract features
    inputs = feature_extractor(
        waveform.squeeze().numpy(), 
        sampling_rate=SAMPLING_RATE, 
        return_tensors="pt"
    )
    
    # Generate transcription
    generated_ids = model.generate(
        inputs.input_features, 
        max_length=448, 
        num_beams=4, 
        repetition_penalty=1.1,
        past_key_values=EncoderDecoderCache.from_legacy_cache(None)  # Handle past_key_values deprecation
    )
    
    # Decode transcription
    transcription = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription