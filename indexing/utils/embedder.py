import torch
from transformers import AutoTokenizer, AutoModel

class Embedder:
    '''
    Embedder class to embed text using a pre-trained model from Hugging Face's Transformers library.
    '''

    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.determine_best_device())

    @staticmethod
    def determine_best_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def embed(self, *text):
        '''
        Embeds a single text.
        
        Args:
            text (str): Text to embed.
        
        Returns:
            torch.Tensor: Embedding of the text.
        '''
        inputs = self.tokenizer(list(text), padding=True, truncation=True, return_tensors="pt").to(
            self.determine_best_device()
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        return torch.nn.functional.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1).cpu()
    
    def batch_embed(self, texts):
        '''
        Embeds a batch of texts.
        
        Args:
            texts (List[str]): List of texts to embed.
            
        Returns:
            torch.Tensor: Embeddings of the texts.
        '''
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            self.determine_best_device()
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        return torch.nn.functional.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1).cpu()