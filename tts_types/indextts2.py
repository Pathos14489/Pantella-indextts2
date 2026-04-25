from src.logging import logging
logging.info("Importing indextts2.py...")
import src.tts_types.base_tts as base_tts
import random
import os
imported = False
try:
    logging.info("Trying to import indextts2")
    from addons.indextts2.index_tts.indextts.infer_v2 import IndexTTS2
    import torch
    import soundfile as sf
    from huggingface_hub import snapshot_download
    imported = True
    logging.info("Imported indextts2")
except Exception as e:
    logging.error(f"Failed to import torch and torchaudio: {e}")
    raise e
logging.info("Imported required libraries in indextts2.py")

tts_slug = "indextts2"
default_settings = {
    "indextts2_banned_voice_models": [],
}
settings_description = {
    "indextts2_banned_voice_models": "A list of voice models to ban from being used by IndexTTS2. This can be changed in config.json. This is useful if you have a voice model that causes issues with IndexTTS2, such as extremely long synthesis times or crashes."
}
options = {}
settings = {}
loaded = False
description = "IndexTTS2 description goes here."
class Synthesizer(base_tts.base_Synthesizer):
    def __init__(self, conversation_manager, ttses = []):
        global tts_slug, default_settings, loaded
        super().__init__(conversation_manager)
        self.tts_slug = tts_slug
        self._default_settings = default_settings
        logging.info(f"Initializing {self.tts_slug}...")
        snapshot_download(repo_id="IndexTeam/IndexTTS-2", revision="main", local_dir="./addons/indextts2/index_tts/models", local_dir_use_symlinks=False)
        self.model = IndexTTS2(cfg_path="./addons/indextts2/index_tts/checkpoints/config.yaml", model_dir="./addons/indextts2/index_tts/models", use_fp16=True, use_cuda_kernel=False, use_deepspeed=True)

        logging.info(f'{self.tts_slug} speaker wavs folders: {self.speaker_wavs_folders}')
        logging.config(f'{self.tts_slug} - Available voices: {self.voices()}')
        if len(self.voices()) > 0:
            random_voice = random.choice(self.voices())
            self._say("Index T T S Two is ready to go.",random_voice)
        loaded = True

    def voices(self):
        """Return a list of available voices"""
        voices = super().voices()
        for banned_voice in self.config.indextts2_banned_voice_models:
            if banned_voice in voices:
                voices.remove(banned_voice)
        return voices
    
    @property
    def default_voice_model_settings(self):
        return {}
    
    def _synthesize(self, voiceline, voice_model, voiceline_location, settings, aggro=0):
        """Synthesize the audio for the character specified using ParlerTTS"""
        logging.output(f'{self.tts_slug} - synthesizing {voiceline} with voice model "{voice_model}"...')
        speaker_wav_path = self.get_speaker_wav_path(voice_model)
        # settings = self.voice_model_settings(voice_model)
        logging.output(f'{self.tts_slug} - using voice model settings: {settings}')
        
        self.model.infer(spk_audio_prompt=speaker_wav_path, text=voiceline, output_path=voiceline_location, emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
        logging.output(f'{self.tts_slug} - synthesized {voiceline} with voice model "{voice_model}"')