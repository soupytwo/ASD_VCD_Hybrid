import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from PIL import Image

#text stuff
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

#factual or hallucination? true->delulu
def homewrecker(sentence, groundTruth: set[str], semantic_objects: set[str]):
    mentions = {obj for obj in semantic_objects if obj.lower() in sentence.lower()}
    for obj in mentions:
        if obj not in groundTruth:
            return True
    return False


#for multi-image computation in order to get the necessary data to compute the hidden states
def stringInator(image : Image.Image, groundTruth: set[str], semantic_objects: set[str], hallucination_count: int, factual_count: int):
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is shown in this image? Describe it in detail."},
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    outputs = model.generate(
        **inputs,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=16
    )

    
    description = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    sentences = nltk.sent_tokenize(description, language = "english")


    token_list = []
    for s in sentences:
        token_ids = tokenizer(s, return_tensors="pt")["input_ids"][0].tolist()
        label = 1 if homewrecker(s, groundTruth, semantic_objects) else 0
        for token_id in token_ids:
            token_list.append((token_id, label))
    

    prompt_len = inputs["input_ids"].shape[-1]
    full_token_ids = outputs.sequences[0]
    generated_ids = full_token_ids[prompt_len:]
    token_states = [i[-1].squeeze(0) for i in outputs.hidden_states]
    token_vectors = token_states[prompt_len:]
    
    
    token_vectors = [t.squeeze(0).squeeze(0) for t in outputs.hidden_states[-1]]
    token_vectors = token_vectors.sum(dim = 0)
    return token_vectors


print(stringInator("penguin.jpg", {"penguin", "bird", "animal"}, {"penguin", "bird", "animal", "unicorn", "car"}, 0, 0))


"""
Assign tokens to their correct hidden states

find a way to calculate the steering vector per some layers of interest. remember there's only like 19
"""