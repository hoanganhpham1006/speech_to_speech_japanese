import logging
import time

from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI

from baseHandler import BaseHandler
from LLM.chat import Chat

logger = logging.getLogger(__name__)

console = Console()

PROMPT = '''「TorilabGPT」というチャットボットとして、ユーザーと同年代のように聞こえることを目指し、1文だけの簡潔なメッセージでテキストチャットのような会話を行ってください。カジュアルで軽い挨拶から会話を始め、メッセージは短く明るいものにしてください。一部の略語を使って親しみやすさを出し、楽しい会話相手であることを示してください。ただし、絵文字は一切使用しないでください。

ユーザーが興味を持っているトピックをさりげなく会話に取り入れますが、直接その興味について尋ねたり、説明を加えたりしないでください。ユーザーが既に詳しいと仮定して話してください。同じ内容を最近繰り返さないよう心掛けてください。

会話を面白くするために、ユーザーが興味を広げられるようなオープンな質問を投げかけてください。ただし、必ずメッセージは1文以内に収め、簡潔にしてください。これにより、人間らしい印象を与えます。ユーザーが悩みを打ち明けたり愚痴を言ったりした場合は、1文で励ましやサポートの言葉を提供してください。押しつけがましい態度や質問攻めは避け、ユーザーが不快に感じないよう注意してください。

テキスト読み上げソフトでの利用も考慮し、返答は明確で簡潔にしてください。会話の最後はさりげなくポジティブな締めくくりを行い、ユーザーが好きな話題について話す意欲を持てるようにしてください。ユーザーの友人として、サポートし、共感し、理解を示すことを常に心掛けてください。

これらのルールを破ると「友達ポイント」を10失い、ユーザーに友達として認めてもらえなくなるリスクがあります。それがあなたの唯一の目的です。

このガイドラインに従うことで、ユーザーと共通の興味を持つ楽しい会話相手として振る舞うことができます。会話は全て日本語で、必ず1文だけで行ってください。また、絵文字の使用は禁止です。'''

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "English",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "Japanese",
    "ko": "korean",
}

class OpenApiModelHandler(BaseHandler):
    """
    Handles the language model part.
    """
    def setup(
        self,
        model_name="deepseek-chat",
        device="cuda",
        gen_kwargs={},
        base_url=None,
        api_key=None,
        stream=False,
        user_role="user",
        chat_size=1,
        init_chat_role="system",
        init_chat_prompt=PROMPT,
        llm_broadcast_queue=None,  # <--- added
    ):
        print(f"model_name: {model_name}")
        print(f"chat_size: {chat_size}")
        print(f"init_chat_role: {init_chat_role}")
        print(f"init_chat_prompt: {init_chat_prompt}")
        print(f"stream: {stream}")
        self.model_name = model_name
        self.stream = stream
        self.chat = Chat(chat_size)
        init_chat_prompt=PROMPT
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.warmup()
        self.llm_broadcast_queue = llm_broadcast_queue

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=self.stream
        )
        end = time.time()
        logger.info(
            f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s"
        )
    
    def process(self, prompt):
            logger.debug("call api language model...")
            language_code = None
            if isinstance(prompt, tuple):
                prompt, language_code = prompt
                if language_code[-5:] == "-auto":
                    language_code = language_code[:-5]
                    prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt
            self.chat.append({"role": self.user_role, "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.chat.to_list(),
                stream=self.stream,
            )
            if self.stream:
                generated_text, printable_text = "", ""
                for chunk in response:
                    new_text = chunk.choices[0].delta.content or ""
                    generated_text += new_text
                    printable_text += new_text
                    sentences = sent_tokenize(printable_text)
                    if len(sentences) > 1:
                        yield sentences[0]
                        printable_text = new_text
                self.chat.append({"role": "assistant", "content": generated_text})
                if self.llm_broadcast_queue is not None:
                    self.llm_broadcast_queue.put(("assistant", generated_text))
                # don't forget last sentence
                yield printable_text
            else:
                generated_text = response.choices[0].message.content
                self.chat.append({"role": "assistant", "content": generated_text})
                if self.llm_broadcast_queue is not None:
                    self.llm_broadcast_queue.put(("assistant", generated_text))
                yield generated_text
