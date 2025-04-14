"""
Module for loading LLMs and their tokenizers from huggingface. 

"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, BitsAndBytesConfig

def get_llm_tokenizer(model_name: str, device: str, load_in_4bit: bool = False) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure a language model and its tokenizer.

    Args:
        model_name: Name or path of the pretrained model to load
        device: Device to load the model on ('cpu' or 'cuda')
        load_in_4bit: Whether to load the model in 4-bit quantization

    Returns:
        tuple containing:
            - The loaded language model
            - The configured tokenizer for that model
    """
    quantization_config = None
    model_kwargs = {"torch_dtype": torch.bfloat16} # Use bfloat16 by default

    if load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError("4-bit quantization requires a CUDA-enabled GPU.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        # device_map="auto" is recommended for 4-bit loading
        # Remove direct .to(device) call later when using device_map
        model_kwargs["device_map"] = "auto"
    else:
        # Original logic: Explicit device placement if not quantizing
        # Keep device_map=None and use .to(device) later
        model_kwargs["device_map"] = None
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Move model to device
    if not load_in_4bit:
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Ensure model config also uses the same pad token id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer