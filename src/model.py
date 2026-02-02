from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.utils import Constants
import logging

def extend_tokenizer_and_model(config, train_df, device):
    """
    Extends the tokenizer with Ethiopic characters found in the training data
    and resizes the model embeddings.
    """
    logger = logging.getLogger("TrOCR_Tigrinya")
    
    # 1. Load Base Processor
    logger.info(f"Loading base processor: {config.pretrained_model}")
    base_processor = TrOCRProcessor.from_pretrained(config.pretrained_model)
    
    # 2. Extract Tigrinya Characters from Data
    logger.info("Extracting Ethiopic characters from training set...")
    # Convert all text to string and join
    all_text = ' '.join(train_df['text'].astype(str))
    all_chars = set(all_text)
    
    ethiopic_chars = []
    for char in all_chars:
        code_point = ord(char)
        # Check against Tigrinya Unicode ranges defined in Constants
        if (Constants.ETHIOPIC_RANGE_1[0] <= code_point <= Constants.ETHIOPIC_RANGE_1[1] or
            Constants.ETHIOPIC_RANGE_2[0] <= code_point <= Constants.ETHIOPIC_RANGE_2[1] or
            Constants.ETHIOPIC_RANGE_3[0] <= code_point <= Constants.ETHIOPIC_RANGE_3[1] or
            Constants.ETHIOPIC_RANGE_4[0] <= code_point <= Constants.ETHIOPIC_RANGE_4[1]):
            ethiopic_chars.append(char)
            
    # 3. Add Tokens to Tokenizer
    # Sort to ensure deterministic order
    num_added = base_processor.tokenizer.add_tokens(sorted(list(set(ethiopic_chars))))
    new_vocab_size = len(base_processor.tokenizer)
    logger.info(f"Added {num_added} new Ethiopic tokens.")
    logger.info(f"New vocabulary size: {new_vocab_size}")
    
    # 4. Load Model and Resize Embeddings
    logger.info(f"Loading model: {config.pretrained_model}")
    model = VisionEncoderDecoderModel.from_pretrained(config.pretrained_model)
    model.decoder.resize_token_embeddings(new_vocab_size)
    
    # 5. Configure Model Parameters
    model.config.decoder_start_token_id = base_processor.tokenizer.bos_token_id
    model.config.pad_token_id = base_processor.tokenizer.pad_token_id
    model.config.eos_token_id = base_processor.tokenizer.eos_token_id
    model.config.vocab_size = new_vocab_size
    model.config.decoder.vocab_size = new_vocab_size
    model.config.max_length = config.max_length
    model.config.num_beams = config.num_beams_eval
    
    # 6. Move to Device (GPU/CPU)
    model.to(device)
    
    return model, base_processor