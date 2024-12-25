import json
import logging
import numpy as np
import io
import torchaudio
from channels.generic.websocket import AsyncWebsocketConsumer

import transformers
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration


logger = logging.getLogger(__name__)

class AudioConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.audio_buffer = b""  # Buffer to store audio data
        self.audio_chunks = []
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")
        self.SAMPLING_RATE = 16000  # Whisper expects audio at 16 kHz
        self.bytes_per_second = self.SAMPLING_RATE * 2  # Assuming 16-bit audio (2 bytes per sample)

    async def disconnect(self, close_code):
        if self.audio_chunks:
            # Combine audio chunks into a single byte sequence
            complete_audio_data = b"".join(self.audio_chunks)

            try:
                # Convert audio bytes to NumPy array
                audio_tensor, sample_rate = torchaudio.load(io.BytesIO(complete_audio_data), format="mp3")

                # Resample to 16 kHz if needed
                if sample_rate != self.SAMPLING_RATE:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.SAMPLING_RATE)
                    waveform = resampler(waveform)
                    inputs = self.feature_extractor(
                            waveform.squeeze().numpy(), 
                            sampling_rate=self.SAMPLING_RATE, 
                            return_tensors="pt"
                    )
                print(audio_tensor)    

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
            # Log and handle text data if received
            logger.info(f"Text data received: {text_data}")
            await self.send(text_data=text_data)

        if bytes_data:
            # Append binary audio chunks to the list
            logger.info(f"Binary data received: {bytes_data[:50]}...")  # Log first 50 bytes
            self.audio_chunks.append(bytes_data)

            # Echo the binary data back to the client
            await self.send(bytes_data=bytes_data)