import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from PIL import Image

#text stuff
import nltk
import re
import string

torch.set_printoptions(threshold=float('inf'))

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float32, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

#factual or hallucination? true->delulu
def homewrecker(sentence: str, groundTruth: set[str], semantic_objects: set[str]):
    
    words = [word.strip(string.punctuation) for word in sentence.lower().split()]
    newSentence = ' '.join(words)

    badobj = [
        re.compile(rf'\b{re.escape(obj.lower())}\b')
        for obj in semantic_objects if obj.lower() not in groundTruth
    ]
    for obj in badobj:
        if obj.search(newSentence):
            return True
    
    return False

#to store stuff
def vectorSaver(image: Image.Image, hallucinated_sum, factual_sum, factual_vectors, hallucinated_vectors, layer, description, filename):
    with open("vectors.txt", "a") as file:
        file.write(f"Image: {filename}\n")

        file.write(f"Hallucinated Vector Sum (Layer {layer}):\n")
        if isinstance(hallucinated_sum, torch.Tensor):
            file.write(f"{hallucinated_sum.tolist()}\n")
        else:
            file.write(f"{hallucinated_sum}\n")

        len_hallucinated = len(hallucinated_vectors)
        file.write(f"Hallucinated Vector total length: {len_hallucinated}\n")

        file.write(f"Factual Vector Sum (Layer {layer}):\n")
        if isinstance(factual_sum, torch.Tensor):
            file.write(f"{factual_sum.tolist()}\n")
        else:
            file.write(f"{factual_sum}\n")

        len_factual = len(factual_vectors)
        file.write(f"Factual Vector total length: {len_factual}\n")

        file.write(f"Description: {description}\n")

        file.write("=" * 50 + "\n")

#for multi-image computation in order to get the necessary data to compute the hidden states
def tokenInator(image : Image.Image, groundTruth: set[str], semantic_objects: set[str], hallucination_count: int, factual_count: int, layer: int, filename):
    
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
        max_new_tokens=64
    )

    input_len = inputs.input_ids.shape[-1]  # Number of prompt tokens
    response_ids = outputs.sequences[0][input_len:]  # Skip prompt
    
    description = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(description)

    sentences = nltk.sent_tokenize(description, language = "english")

    temp_ids = []
    token_per_sentence = []
    count = 0
    for tk_id in response_ids:
        if(count >= len(sentences)):
            break
        temp_ids.append(tk_id)
        temp_text = tokenizer.decode(temp_ids, skip_special_tokens=True)

        if temp_text == sentences[count]:
            print("Match detected")
            count += 1
            token_per_sentence.append(temp_ids.copy())
            temp_ids = []
        
    hallucinated_vectors = []
    factual_vectors = []
    for idx, s in enumerate(sentences):
        label = 1 if homewrecker(s, groundTruth, semantic_objects) else 0
        for token_id in token_per_sentence[idx]:
            token_vector = outputs.hidden_states[layer][idx]
            if label == 1:                  
                hallucinated_vectors.append(token_vector)
                    
            if label == 0:
                factual_vectors.append(token_vector)
      
    if hallucinated_vectors:
        hallucinated_sum = torch.stack(hallucinated_vectors, dim=0).sum(dim=0)
    else:
        hallucinated_sum = "empty!"

    if factual_vectors:
        factual_sum = torch.stack(factual_vectors, dim=0).sum(dim=0)
    else:
        factual_sum = "empty!"
    
    vectorSaver(image, hallucinated_sum, factual_sum, factual_vectors, hallucinated_vectors, layer, description, filename)
    return hallucinated_sum, factual_sum, len(hallucinated_vectors), len(factual_vectors)


semantic_truths = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "trucks","boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

factual_count = 0
hallucination_count = 0


print(tokenInator("COCO9.jpg", {"laptop", "keyboard", "mouse"}, semantic_truths, hallucination_count, factual_count, 10, "COCO9.jpg"))

print(tokenInator("COCO10.jpg", {"pizza", "cup," "table", "fork", "dining table"}, semantic_truths, hallucination_count, factual_count, 10, "COCO10.jpg"))