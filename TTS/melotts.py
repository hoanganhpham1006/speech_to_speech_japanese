from MeloTTS.melo.api import TTS
import logging
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class MeloTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cuda:1",
        #language="EN_NEWEST",
        language="JP",
        blocksize=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.model = TTS(language=language, device=device)
        self.speaker_id = self.model.hps.data.spk2id[language]
        self.blocksize = blocksize
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.tts_to_file("text", self.speaker_id, quiet=True)

    def process(self, llm_sentence):
        llm_sentence = llm_sentence.replace("AI","エーアイ")
        llm_sentence = llm_sentence.replace("Python","パイソン")
        llm_sentence = llm_sentence.replace("A","エー")
        llm_sentence = llm_sentence.replace("B","ビー")
        llm_sentence = llm_sentence.replace("C","シー")
        llm_sentence = llm_sentence.replace("D","ディー")
        llm_sentence = llm_sentence.replace("E","イー")
        llm_sentence = llm_sentence.replace("F","エフ")
        llm_sentence = llm_sentence.replace("G","ジー")
        llm_sentence = llm_sentence.replace("H","エイチ")
        llm_sentence = llm_sentence.replace("I","アイ")
        llm_sentence = llm_sentence.replace("J","ジェイ")
        llm_sentence = llm_sentence.replace("K","ケイ")
        llm_sentence = llm_sentence.replace("L","エル")
        llm_sentence = llm_sentence.replace("M","エム")
        llm_sentence = llm_sentence.replace("N","エヌ")
        llm_sentence = llm_sentence.replace("O","オー")
        llm_sentence = llm_sentence.replace("P","ピー")
        llm_sentence = llm_sentence.replace("Q","キュー")
        llm_sentence = llm_sentence.replace("R","アール")
        llm_sentence = llm_sentence.replace("S","エス")
        llm_sentence = llm_sentence.replace("T","ティー")
        llm_sentence = llm_sentence.replace("U","ユー")
        llm_sentence = llm_sentence.replace("V","ブイ")
        llm_sentence = llm_sentence.replace("W","ダブリュー")
        llm_sentence = llm_sentence.replace("X","エックス")
        llm_sentence = llm_sentence.replace("Y","ワイ")
        llm_sentence = llm_sentence.replace("Z","ゼット")
        console.print(f"[green]ASSISTANT: {llm_sentence}")
        if self.device == "mps":
            import time
            start = time.time()
            torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
            torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.
            time_it_took = time.time()-start  # Removing this line makes it fail more often. I'm looking into it.
        audio_chunk = self.model.tts_to_file(llm_sentence, self.speaker_id, quiet=True)
        if len(audio_chunk) == 0:
            self.should_listen.set()
            return
        audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)
        for i in range(0, len(audio_chunk), self.blocksize):
            yield np.pad(
                audio_chunk[i : i + self.blocksize],
                (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
            )

        self.should_listen.set()
