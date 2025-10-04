
import os
from typing import Dict, Optional, List, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Qwen3Embedding:
    """
    Used for encoding text into high quality vector representations.
    """
    def __init__(self, model_name_or_path: str, instruction: Optional[str] = None, use_fp16: bool = True, use_cuda: bool = True, max_length: int = 8192):
        """
        Initialize the Qwen3-Embedding model and tokenizer.

        Args:
            model_name_or_path (str): Path to the Qwen3-Embedding model.
            instruction (str, optional): The text to be encoded.
            use_fp16 (bool): Whether to use float16.
            use_cuda (bool): Whether to load the model onto a CUDA device.
            max_length (int): Maximum length of the input text.
        """
        
        self.instruction = instruction
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        self.dtype = torch.float16 if use_fp16 else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        # Whether to use Flash Attention 2.
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() and use_cuda else "eager"
        
        try:
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True, 
                attn_implementation=attn_implementation, 
                torch_dtype=self.dtype
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True, 
                padding_side='left' # Left padding is crucial for last_token_pool.
            )
        except Exception as e:
            logger.error(f"error: {e}")
            raise

        if use_cuda:
            self.model.to(self.device)
        
        self.model.eval() 
        self.max_length = max_length
        logger.info("Qwen3-Embedding loaded successfully.")

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Pools the hidden states of the model's last layer to obtain a representation for the entire sequence.
        Due to the use of left padding, the vector of the last token serves as the effective representation of the sequence.

        Args:
            last_hidden_states (Tensor): The hidden states of the model's final layer.
            attention_mask (Tensor): attention mask.

        Returns:
            Tensor: The pooled text vector.
        """
        attention_mask[:, -1].sum() == attention_mask.shape[0]
        
        # With left padding, directly take the hidden state of the last token.
        return last_hidden_states[:, -1]
       

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """
        Constructs an input string for the query with an instruction.
        Args:
            task_description (str): The task description, which is empty in this context.
            query (str): The original query.

        Returns:
            str: The resulting input string.
        """
        if task_description is None:
            task_description = self.instruction
        return f'Instruct: {task_description}\nQuery:{query}'

    @torch.no_grad() # Disables gradient calculation within this method.
    def encode(self, sentences: Union[List[str], str], is_query: bool = False, instruction: Optional[str] = None, dim: int = -1, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Encodes a batch of texts into vectors.
        Args:
            sentences (Union[List[str], str]): The texts to be encoded.
            is_query (bool): False.
            instruction (str, optional): Custom instruction, which is empty here.
            dim (int): The dimension of the output vectors. If -1, the model's original dimension is used.
            batch_size (int): batch size.
            show_progress (bool): Whether to display a tqdm progress bar.

        Returns:
            np.ndarray: The encoded vectors.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating Text Embeddings")

        for i in iterator:
            batch_sentences = sentences[i:i+batch_size]
            
            if is_query:
                batch_sentences = [self.get_detailed_instruct(instruction, sent) for sent in batch_sentences]
            
            # tokenization
            inputs = self.tokenizer(batch_sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            inputs = inputs.to(self.device)

            # model forward
            model_outputs = self.model(**inputs)
            
            # pooling
            output = self.last_token_pool(model_outputs.last_hidden_state, inputs['attention_mask'])
            
            # dimension reduction
            if dim != -1:
                output = output[:, :dim]
            
            # L2 normalization
            output = F.normalize(output, p=2, dim=1)
            
            all_embeddings.append(output.cpu())
            
        return torch.cat(all_embeddings, dim=0).to(self.dtype).numpy()



