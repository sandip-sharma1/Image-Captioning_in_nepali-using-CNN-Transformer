import os
import re
import numpy as np
from new_getcaption import get_captions_for_image
from onephotoinference import generate_captions
from nltk.translate.bleu_score import sentence_bleu


SEQ_LENGTH=25
caption_path='your caption file path for Bleu score evaluation'
image_path='Image path for Bleu score evaluation'

def load_captions_data(filename):
    
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            tokens = line.split()
            # Image name and captions are separated using a tab
            img_name, caption = tokens[0], tokens[1:]

            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            img_name = img_name.split("#")[0]
            img_name = os.path.join(image_path, img_name.strip())
            caption = ' '.join(caption)
            caption = re.sub(u'[\u0964]+', '', caption)

            # We will remove caption that are either too short to too long
            cap_len = caption.strip().split()

            if len(cap_len) < 2 or len(cap_len) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                # We will add a start and an end token to each caption
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data
    
def train_val_split(caption_data, train_size=0.95, shuffle=True):

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data

captions_mapping, text_data = load_captions_data(caption_path)
train_data, valid_data = train_val_split(captions_mapping)



def original_tokenization(caption):
    caption = caption.strip()  # Remove leading/trailing whitespace
    tokens = caption.split()  # Split by whitespace
    return tokens

num=len(valid_data)
check_num=num
counter=num
sum1=0
avg1=0 
for i in valid_data:
    full_path = i
    image_name = full_path.split('/')[-1]
    prediction = generate_captions(full_path)   
    # Get the 5 reference captions
    a, b, c, d, e = get_captions_for_image(image_name)   
    # Skip if captions are missing
    if a is None:
        continue  
    # Tokenize prediction and references
    pred_tokens = original_tokenization(prediction)
    ref_tokens = [
        original_tokenization(a),
        original_tokenization(b),
        original_tokenization(c),
        original_tokenization(d),
        original_tokenization(e)
    ]
    bleu_1 = sentence_bleu(ref_tokens, pred_tokens, weights=(2, 0, 0, 0))
    # bleu_2 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0))
    # bleu_3 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0))
    # bleu_4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    sum1=sum1+bleu_1
    counter=counter-1
    if counter==0:
        break

avg1=sum1/check_num
print(avg1)


