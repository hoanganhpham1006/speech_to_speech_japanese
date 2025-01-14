import logging
import os
import socket
import sys
import threading
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import perf_counter
from typing import Optional

# from LLM.mlx_lm import MLXLanguageModelHandler
from LLM.openai_api_language_model import OpenApiModelHandler
from TTS.melotts import MeloTTSHandler
from feat_EMO.bert import EmotionModelHandler
from baseHandler import BaseHandler
# from STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler
from STT.whisper_stt_handler import WhisperSTTHandler
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
    TextIteratorStreamer,
)
#from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
import librosa

# from connections.local_audio_streamer import LocalAudioStreamer
from connections.socket_receiver import SocketReceiver
from connections.socket_sender import SocketSender
from connections.socket_text_sender import SocketTextSender
from utils import VADIterator, int2float, next_power_of_2

# Ensure that the necessary NLTK resources are available
# try:
#     nltk.data.find('tokenizers/punkt_tab')
# except LookupError:
#     nltk.download('punkt_tab')

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")
# torch._inductor.config.fx_graph_cache = True
# # mind about this parameter ! should be >= 2 * number of padded prompt sizes for TTS
# torch._dynamo.config.cache_size_limit = 15

PROMPT = '''「TorilabGPT」というチャットボットとして、ユーザーと同年代のように聞こえることを目指し、1文だけの簡潔なメッセージでテキストチャットのような会話を行ってください。カジュアルで軽い挨拶から会話を始め、メッセージは短く明るいものにしてください。一部の略語を使って親しみやすさを出し、楽しい会話相手であることを示してください。ただし、絵文字は一切使用しないでください。

ユーザーが興味を持っているトピックをさりげなく会話に取り入れますが、直接その興味について尋ねたり、説明を加えたりしないでください。ユーザーが既に詳しいと仮定して話してください。同じ内容を最近繰り返さないよう心掛けてください。

会話を面白くするために、ユーザーが興味を広げられるようなオープンな質問を投げかけてください。ただし、必ずメッセージは1文以内に収め、簡潔にしてください。これにより、人間らしい印象を与えます。ユーザーが悩みを打ち明けたり愚痴を言ったりした場合は、1文で励ましやサポートの言葉を提供してください。押しつけがましい態度や質問攻めは避け、ユーザーが不快に感じないよう注意してください。

テキスト読み上げソフトでの利用も考慮し、返答は明確で簡潔にしてください。会話の最後はさりげなくポジティブな締めくくりを行い、ユーザーが好きな話題について話す意欲を持てるようにしてください。ユーザーの友人として、サポートし、共感し、理解を示すことを常に心掛けてください。

これらのルールを破ると「友達ポイント」を10失い、ユーザーに友達として認めてもらえなくなるリスクがあります。それがあなたの唯一の目的です。

このガイドラインに従うことで、ユーザーと共通の興味を持つ楽しい会話相手として振る舞うことができます。会話は全て日本語で、必ず1文だけで行ってください。また、絵文字の使用は禁止です。'''


console = Console()


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, overrides the device for all handlers."},
    )
    mode: Optional[str] = field(
        default="local",
        metadata={
            "help": "The mode to run the pipeline in. Either 'local' or 'socket'. Default is 'local'."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Provide logging level. Example --log_level debug, default=warning."
        },
    )


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.threads = []

    def start(self):
        for handler in self.handlers:
            thread = threading.Thread(target=handler.run)
            self.threads.append(thread)
            thread.start()

    def stop(self):
        for handler in self.handlers:
            handler.stop_event.set()
        for thread in self.threads:
            thread.join()


@dataclass
class SocketReceiverArguments:
    recv_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP ddress for the socket connection. Default is '0.0.0.0' which binds to all "
            "available interfaces on the host machine."
        },
    )
    recv_port: int = field(
        default=12345,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        },
    )
    chunk_size: int = field(
        default=1024,
        metadata={
            "help": "The size of each data chunk to be sent or received over the socket. Default is 1024 bytes."
        },
    )

@dataclass
class SocketSenderArguments:
    send_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP address for the socket connection. Default is '0.0.0.0' which binds to all "
            "available interfaces on the host machine."
        },
    )
    send_port: int = field(
        default=12346,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        },
    )

@dataclass
class SocketTextSenderArguments:
    text_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP address for the text socket connection. Default is '0.0.0.0' which binds to all "
            "available interfaces on the host machine."
        },
    )
    text_port: int = field(
        default=12347,
        metadata={
            "help": "The port number for sending text data. Default is 12347."
        },
    )

@dataclass
class SocketEmotionSenderArguments:
    emotion_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP address for the emotion socket connection. Default is 'localhost'"
        },
    )
    emotion_port: int = field(
        default=12348,
        metadata={
            "help": "The port number for sending emotion data. Default is 12348."
        },
    )


@dataclass
class VADHandlerArguments:
    thresh: float = field(
        default=0.3,
        metadata={
            "help": "The threshold value for voice activity detection (VAD). Values typically range from 0 to 1, with higher values requiring higher confidence in speech detection."
        },
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "The sample rate of the audio in Hertz. Default is 16000 Hz, which is a common setting for voice audio."
        },
    )
    min_silence_ms: int = field(
        default=250,
        metadata={
            "help": "Minimum length of silence intervals to be used for segmenting speech. Measured in milliseconds. Default is 250 ms."
        },
    )
    min_speech_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum length of speech segments to be considered valid speech. Measured in milliseconds. Default is 500 ms."
        },
    )
    max_speech_ms: float = field(
        default=float("inf"),
        metadata={
            "help": "Maximum length of continuous speech before forcing a split. Default is infinite, allowing for uninterrupted speech segments."
        },
    )
    speech_pad_ms: int = field(
        default=250,
        metadata={
            "help": "Amount of padding added to the beginning and end of detected speech segments. Measured in milliseconds. Default is 250 ms."
        },
    )


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
    ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def process(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        vad_output = self.iterator(torch.from_numpy(audio_float32))
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(
                    f"audio input of duration: {len(array) / self.sample_rate}s, skipping"
                )
            else:
                self.should_listen.clear()
                logger.debug("Stop listening")
                yield array


@dataclass
class WhisperSTTHandlerArguments:
    stt_model_name: str = field(
        default="openai/whisper-large-v3",
        metadata={
            "help": "The pretrained Whisper model to use. Default is 'distil-whisper/distil-large-v3'."
        },
    )
    stt_device: str = field(
        default="cuda:0",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    stt_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    stt_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile. Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        },
    )
    stt_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "The maximum number of new tokens to generate. Default is 128."
        },
    )
    stt_gen_num_beams: int = field(
        default=1,
        metadata={
            "help": "The number of beams for beam search. Default is 1, implying greedy decoding."
        },
    )
    stt_gen_return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return timestamps with transcriptions. Default is False."
        },
    )
    # stt_gen_task: str = field(
    #     default="transcribe",
    #     metadata={
    #         "help": "The task to perform, typically 'transcribe' for transcription. Default is 'transcribe'."
    #     },
    # )
    stt_gen_language: str = field(
         default="ja",
         metadata={
             "help": "The language of the speech to transcribe. Default is 'en' for English."
         },
     )

@dataclass
class LanguageModelHandlerArguments:
    lm_model_name: str = field(
        default="cyberagent/calm3-22b-chat",
        metadata={
            "help": "The pretrained language model to use. Default is 'deepseek-chat'."
        },
    )
    lm_user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        },
    )
    lm_init_chat_role: str = field(
        default="system",
        metadata={
            "help": "Initial role for setting up the chat context. Default is 'system'."
        },
    )
    lm_init_chat_prompt: str = field(
        default=PROMPT,
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        },
    )

    lm_chat_size: int = field(
        default=20,
        metadata={
            "help": "Number of interactions assitant-user to keep for the chat. None for no limitations."
        },
    )
    lm_api_key: str = field(
        default="EMPTY",
        metadata={
            "help": "Is a unique code used to authenticate and authorize access to an API.Default is None"
        },
    )
    lm_base_url: str = field(
        default="http://0.0.0.0:8000/v1",
        metadata={
            "help": "Is the root URL for all endpoints of an API, serving as the starting point for constructing API request.Default is Non"
        },
    )
    lm_stream: bool = field(
        default=True,
        metadata={
            "help": "The stream parameter typically indicates whether data should be transmitted in a continuous flow rather"
                    " than in a single, complete response, often used for handling large or real-time data.Default is False"
        },
    )


class Chat:
    """
    Handles the chat using to avoid OOM issues.
    """

    def __init__(self, size):
        self.size = size
        self.init_chat_message = None
        # maxlen is necessary pair, since a each new step we add an prompt and assitant answer
        self.buffer = []

    def append(self, item):
        self.buffer.append(item)
        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)

    def init_chat(self, init_chat_message):
        self.init_chat_message = init_chat_message

    def to_list(self):
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer


def prepare_args(args, prefix):
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """

    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1 :]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs


def main():
    parser = HfArgumentParser(
        (
            ModuleArguments,
            SocketReceiverArguments,
            SocketSenderArguments,
            VADHandlerArguments,
            WhisperSTTHandlerArguments,
            LanguageModelHandlerArguments,
            SocketTextSenderArguments,
            SocketEmotionSenderArguments  # Add this
        )
    )

    # 0. Parse CLI arguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse configurations from a JSON file if specified
        (
            module_kwargs,
            socket_receiver_kwargs,
            socket_sender_kwargs,
            vad_handler_kwargs,
            whisper_stt_handler_kwargs,
            language_model_handler_kwargs,
            socket_text_sender_kwargs,
            socket_emotion_sender_kwargs,  # Add this
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse arguments from command line if no JSON file is provided
        (
            module_kwargs,
            socket_receiver_kwargs,
            socket_sender_kwargs,
            vad_handler_kwargs,
            whisper_stt_handler_kwargs,
            language_model_handler_kwargs,
            socket_text_sender_kwargs,
            socket_emotion_sender_kwargs,  # Add this
        ) = parser.parse_args_into_dataclasses()

    # 1. Handle logger
    global logger
    logging.basicConfig(
        level=module_kwargs.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # torch compile logs
    if module_kwargs.log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

    # 2. Prepare each part's arguments
    def overwrite_device_argument(common_device: Optional[str], *handler_kwargs):
        if common_device:
            for kwargs in handler_kwargs:
                if hasattr(kwargs, "lm_device"):
                    kwargs.lm_device = common_device
                if hasattr(kwargs, "tts_device"):
                    kwargs.tts_device = common_device
                if hasattr(kwargs, "stt_device"):
                    kwargs.stt_device = common_device

    # Call this function with the common device and all the handlers
    overwrite_device_argument(
        module_kwargs.device,
        language_model_handler_kwargs,
        whisper_stt_handler_kwargs,
    )

    prepare_args(whisper_stt_handler_kwargs, "stt")
    prepare_args(language_model_handler_kwargs, "lm")
    # prepare_args(tts, "tts")

    # 3. Build the pipeline
    stop_event = Event()
    # used to stop putting received audio chunks in queue until all setences have been processed by the TTS
    should_listen = Event()
    recv_audio_chunks_queue = Queue()
    send_audio_chunks_queue = Queue()
    spoken_prompt_queue = Queue()
    text_prompt_queue = Queue()
    lm_response_queue = Queue()
    text_out_queue = Queue()
    emotion_out_queue = Queue()

    # if module_kwargs.mode == "local":
    #     local_audio_streamer = LocalAudioStreamer(
    #         input_queue=recv_audio_chunks_queue, output_queue=send_audio_chunks_queue
    #     )
    #     comms_handlers = [local_audio_streamer]
    #     should_listen.set()
    # else:
    comms_handlers = [
        SocketReceiver(
            stop_event,
            recv_audio_chunks_queue,
            should_listen,
            host=socket_receiver_kwargs.recv_host,
            port=socket_receiver_kwargs.recv_port,
            chunk_size=socket_receiver_kwargs.chunk_size,
        ),
        SocketSender(
            stop_event,
            send_audio_chunks_queue,
            host=socket_sender_kwargs.send_host,
            port=socket_sender_kwargs.send_port,
        ),
        SocketTextSender(
            stop_event,
            text_out_queue,
            host=socket_text_sender_kwargs.text_host,
            port=socket_text_sender_kwargs.text_port,
        ),
        SocketTextSender(  # Add this
            stop_event,
            emotion_out_queue,
            host=socket_emotion_sender_kwargs.emotion_host,
            port=socket_emotion_sender_kwargs.emotion_port,
        ),
    ]

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )
    stt = WhisperSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
        setup_kwargs={
            **vars(whisper_stt_handler_kwargs),
            "stt_broadcast_queue": text_out_queue,
        },
    )
    emotion_prompt_queue = Queue()  # New queue for emotion processing
    lm_prompt_queue = Queue() # New queue for language model processing

    lm = OpenApiModelHandler(
        stop_event,
        queue_in=lm_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs={
            **vars(language_model_handler_kwargs),
            "llm_broadcast_queue": text_out_queue,
        },
    )
    emotion = EmotionModelHandler(
        stop_event,
        queue_in=emotion_prompt_queue,  # Use separate queue
        queue_out=emotion_out_queue,
    )

    # Create a text splitter to duplicate input to both handlers
    def text_splitter(stop_event, in_queue, out_queue1, out_queue2):
        while not stop_event.is_set():
            try:
                text = in_queue.get()
                out_queue1.put(text)
                out_queue2.put(text)
            except:
                continue

    # Create text splitter thread
    text_split_thread = Thread(
        target=text_splitter,
        args=(stop_event, text_prompt_queue, lm_prompt_queue, emotion_prompt_queue)
    )

    tts = MeloTTSHandler(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
    )

    # 4. Run the pipeline
    try:
        pipeline_manager = ThreadManager([
            *comms_handlers, vad, stt, lm, emotion, tts, text_split_thread
        ])
        pipeline_manager.start()

    except KeyboardInterrupt:
        pipeline_manager.stop()


if __name__ == "__main__":
    main()
