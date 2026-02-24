from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("Yonatanhaile2026/tigrinya-trocr-printed-model")
model = VisionEncoderDecoderModel.from_pretrained("Yonatanhaile2026/tigrinya-trocr-printed-model")

image = Image.open("tigrinya_text_line.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
