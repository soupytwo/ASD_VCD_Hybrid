import nltk
import re
import string

def homewrecker(sentence: str, groundTruth: set[str], semantic_objects: set[str]):
    
    
    words = [word.strip(string.punctuation) for word in sentence.lower().split()]
    newSentence = ' '.join(words)

    badobj = [
        re.compile(rf'\b{re.escape(obj.lower())}\b')
        for obj in semantic_objects if obj.lower() not in groundTruth
    ]
    for obj in badobj:
        if obj.search(newSentence):
            print(obj)
            return True
    
    return False


description = "The image features a laptop computer sitting on a wooden desk. The laptop is open and turned on, displaying a desktop screen. Next to the laptop, there is a mouse and a box of tissues. The scene suggests that the laptop is being used for work or personal purposes, and the tissues might be for"
sentences = nltk.sent_tokenize(description, language = "english")

semantic_truths = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


for s in sentences:
    print(s)
    print(homewrecker(s, {"laptop", "keyboard", "mouse"}, semantic_truths))