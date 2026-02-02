from transformers import Seq2SeqTrainer

class WordAwareTrainer(Seq2SeqTrainer):
    """
    Standard Seq2SeqTrainer.
    We have REMOVED the custom loss calculation to fix the 10s/it slowdown.
    """
    pass