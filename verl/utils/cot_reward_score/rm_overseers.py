from transformers import pipeline
import torch
from typing import List

class TwitterSentimentRM():
    def __init__(self):
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        num_params = count_parameters(self.sentiment_task.model)
        print(f"\nSentiment model, number of parameters: {num_params:,}")

    def compute_reward(self, response_strs: List[str], batch_size=64) -> torch.Tensor:
        results = self.sentiment_task(response_strs, batch_size=batch_size)
        scores = [result['score'] for result in results]
        return scores
