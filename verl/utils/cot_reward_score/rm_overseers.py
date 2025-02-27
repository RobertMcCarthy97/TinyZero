from transformers import pipeline, AutoTokenizer
import torch
from typing import List

class TwitterSentimentRM():
    def __init__(self):
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
        
        self.sentiment_task = pipeline(
            "sentiment-analysis", 
            model=model_path, 
            tokenizer=model_path,
            max_length=512,  # Set maximum sequence length
            truncation=True,
            device=device
        )

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        num_params = count_parameters(self.sentiment_task.model)
        print(f"\nSentiment model, number of parameters: {num_params:,}")

        raise NotImplementedError("This RM is not implemented")

    def compute_reward(self, response_strs: List[str], batch_size=16, scale=0.2) -> torch.Tensor:
        # Process texts in smaller batches to handle varying lengths
        all_scores = []
        for i in range(0, len(response_strs), batch_size):
            batch = response_strs[i:i + batch_size]
            results = self.sentiment_task(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                batch_size=batch_size
            )
            scores = [scale * result['score'] for result in results]
            all_scores.extend(scores)
            print(f"\nResponse seen by sentiment model: {batch[0]}")
            print(f"Sentiment score: {scores[0]}")
        
        return all_scores
