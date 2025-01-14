from baseHandler import BaseHandler
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from LLM.chat import Chat

# specify the directory where the model files are stored
ckpt="/home/andrew/modernbert-finetune/bert-base-emotion/checkpoint-600"
model_id = "tohoku-nlp/bert-base-japanese"

logger = logging.getLogger(__name__)

def preprocess_text(conversation, num_turns_to_keep = 7):
    truncated_conversation = conversation[-num_turns_to_keep:]
    truncated_conversation[-1] = f"[TARGET] {truncated_conversation[-1]}"
    formatted_input = " [SEP] ".join(truncated_conversation)
    formatted_input = f"[CLS] {formatted_input} [SEP]"
    return formatted_input

class EmotionModelHandler(BaseHandler):
    """
    Handles the emotion model part.
    """
    def setup(
        self,
        device="cuda:0",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.model_max_length = 256
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.warmup()
        self.chat = Chat(20)

    def predict_emotion(self, conversation):
        # Preprocess and tokenize
        formatted_input = preprocess_text(conversation)
        inputs = self.tokenizer(
            formatted_input,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Map emotion IDs to labels
        emotion_to_id = {
            'sadness': 0,
            'joy': 1,
            'anger': 2,
            'fear': 3,
            'surprise': 4,
            'disgust': 5,
            'neutral': 6
        }
        emotion_map = {v: k for k, v in emotion_to_id.items()}
        
        # Get top 3 predictions
        probs, indices = torch.topk(predictions[0], 3)
        top_3 = [(emotion_map[idx.item()], prob.item()) for idx, prob in zip(indices, probs)]
        
        return top_3

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        conversation = [
            "user: こんにちは",
            "assistant: 元気ですか？",
            "user: 元気です",
            "assistant: それはよかった",
            "user: ありがとう",
        ]
        print(self.predict_emotion(conversation))

    
    def _preprocess_messages(self, message):
        self.chat.append({"role": "user", "content": message})
        conversation = []
        for m in self.chat.to_list():
            conversation.append(m['role'] + ": " + m['content'])
        print("conversation: ", conversation)
        return conversation

    def process(self, message):
        logger.debug("call emotion model...")
        conversation = self._preprocess_messages(message[0])
        emotion = self.predict_emotion(conversation)
        emotion_str = ', '.join([f"{e[0]}({e[1]:.3f})" for e in emotion])
        yield "[emotion] " + emotion_str
