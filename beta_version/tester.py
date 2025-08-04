import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

#text stuff
import nltk
nltk.download('punkt')

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

#for multi-image computation in order to get the necessary data to compute the hidden states
def initialSetup(image : Image.Image, groundTruth: set[str]):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image)},
                {"type": "text", "text": "What is shown in this image?"},
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
        return_dict_in_generate=True
    )

    return outputs.hidden_states




#print(initialSetup("penguin.jpg", {"penguin", "bird", "animal"}))

print(len(initialSetup('penguin.jpg', {'penguin', 'bird', 'animal'})))







# work on line 10 so index [11]

"""
You’ve basically mapped out the core pipeline! 🚀 Let’s turn it into a clean, step-by-step checklist so you can keep your workflow organized and modular. Here's how it could look:

---

### 🧠 Steering Vector Workflow Summary

#### **Step 1: Token-Level Labeling**
- Decode the generated token IDs → `outputs.sequences`
- Convert token IDs to strings → `convert_ids_to_tokens(...)`
- Segment output text into sentences → `nltk.sent_tokenize(...)`
- Compare each sentence to ground-truth object names
- Label each token: **factual** ✅ or **hallucinated** ❌

> 💡 This sets up the scaffolding for analyzing model behavior—no hidden states needed yet.

---

#### **Step 2: Hidden State Extraction**
- Once labels are solid, access hidden states → `outputs.hidden_states`
- Select a layer (e.g. `layer20`) for embedding analysis
- Match hidden state vectors to the labeled tokens

---

#### **Step 3: Steering Vector Computation**
- Calculate mean embedding for **factual** tokens → μ_factual  
- Calculate mean embedding for **hallucinated** tokens → μ_hallucinated  
- Compute steering vector → `v_steer = μ_factual - μ_hallucinated`

---

#### **Step 4: Apply Steering Vector**
- Incorporate `v_steer` during generation (e.g. modify hidden state or logits)
- Evaluate output behavior: Are hallucinations reduced? Are factuals promoted?

---

#### **Step 5: Reap the Benefits 🌱**
- Visualize the effect: Compare outputs before vs. after steering
- Iterate or expand: Try different layers or domain-specific ground truths
- Share findings! It’s a cool interpretability story in the making.

---

Want me to help scaffold one of these steps in code? Maybe the labeling loop or the mean vector calculator?

"""