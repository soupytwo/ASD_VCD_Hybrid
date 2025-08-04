import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration, LogitsProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

class VCDProcessor(LogitsProcessor):
    def __init__(self, adjusted_logits):
        super().__init__()
        self.adjusted = adjusted_logits
        self.count = 0

    def __call__(self, input_ids, scores):
        if self.count < self.adjusted.size(0):
            scores = self.adjusted[self.count]
        self.count += 1
        return scores


def gaussInator(image):
    og = np.array(image, dtype=np.float32) / 255.0

    mean = 0
    var = 0.10
    sigma = np.sqrt(var)
    noise = np.random.normal(loc=mean, scale=sigma, size=og.shape)

    corrupted = np.clip(og + noise, 0.0, 1.0)
    corrupted_pil = Image.fromarray((corrupted * 255).astype(np.uint8))
    return corrupted_pil

def VCDInator(image):
    clean = Image.open(image)
    noisy = gaussInator(clean)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": clean},  
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]

    inputsClean = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, torch.float16)
    conversation[0]["content"][0]["image"] = noisy
    inputsNoisy = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, torch.float16)

    with torch.no_grad():
        outputClean = model(**inputsClean, output_hidden_states=False, output_attentions=False, output_logits=True)
        outputNoisy = model(**inputsNoisy, output_hidden_states=False, output_attentions=False, output_logits=True)

    cleanLogits = outputClean.logits.squeeze(0)
    noisyLogits = outputNoisy.logits.squeeze(0)

    beta = 0.1
    plausibilityls = []
    for row in cleanLogits:
        rowPlausibility = []
        for l in row:
            if l > beta:
                rowPlausibility.append(True)
            else:
                rowPlausibility.append(False)
        plausibilityls.append(rowPlausibility)

    plausibilityTensor = torch.tensor(plausibilityls, dtype=torch.bool, device=cleanLogits.device)

    adjustedClean = torch.where(plausibilityTensor, cleanLogits, torch.tensor(0, device=cleanLogits.device))
    adjustedNoisy = torch.where(plausibilityTensor, noisyLogits, torch.tensor(0, device=noisyLogits.device))

    alpha = 1.0
    adjusted_logits = (1 + alpha) * adjustedClean - alpha * adjustedNoisy

    input_len = inputsClean.input_ids.shape[-1]  # Number of prompt tokens
    generated_logits = adjusted_logits[input_len : input_len + 64]

    vcd_processor = VCDProcessor(generated_logits)
    generated_ids = model.generate(
        input_ids=inputsClean["input_ids"],
        attention_mask=inputsClean["attention_mask"],
        pixel_values=inputsClean["pixel_values"],
        logits_processor=[vcd_processor],
        max_new_tokens=64,
        do_sample=False,
    )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

print(VCDInator("penguin.jpg"))  