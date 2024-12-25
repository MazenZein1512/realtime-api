import os
import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from django.core.files.storage import default_storage

import numpy as np
import io
from pydub import AudioSegment
import torchaudio
from channels.generic.websocket import AsyncWebsocketConsumer

import transformers
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration

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

    async def disconnect(self, close_code):
        if self.audio_chunks:
            # Combine audio chunks into a single byte sequence
            complete_audio_data = b"".join(self.audio_chunks)

            try:

                audio = AudioSegment.from_file(io.BytesIO(complete_audio_data), format="webm")

                output_buffer = io.BytesIO()
                audio.export(output_buffer, format="mp3", bitrate="192k")  # Export to MP3
                output_buffer.seek(0)


                transcription = transcribe_audio(output_buffer)
   
                print(transcription)    

                # Transcribe the audio using Whisper
                # logger.info("Processing audio with Whisper...")
                # result = self.model.transcribe(audio_tensor, fp16=False)

                # # Send transcription result back to the client
                # transcription = result.get("text", "")
                response_data = {
                    "message": "Transcription completed",
                    "transcription": "transcription",
                }
                await self.send(text_data=json.dumps(response_data))
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                response_data = {
                    "message": "Error processing audio",
                    "error": str(e),
                }
                await self.send(text_data=json.dumps(response_data))

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            # Process and log the received text data (if necessary)
            logger.info(f"Text data received: {text_data}")
            await self.send(text_data=text_data)

        if bytes_data:
            # Log and store the binary audio chunks
            logger.info(f"Binary data received: {bytes_data[:50]}...")  # Log first 50 bytes for brevity
            self.audio_chunks.append(bytes_data)

            logger.info(f"chunks size received: {len(self.audio_chunks)}...")  # Log first 50 bytes for brevity
            logger.info(f"first: {self.audio_chunks[:5]}...")  # Log first 50 bytes for brevity

            # Echo the binary data back to the client in real-time
            await self.send(bytes_data=bytes_data)


def transcribe_audio(audio):
    """
    Transcribe a single audio file
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        str: Transcribed text
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio, format="mp3")
    if sample_rate != SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLING_RATE)
        waveform = resampler(waveform)
    
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
    )
    
    # Decode transcription
    transcription = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription