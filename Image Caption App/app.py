# from flask import Flask, render_template

from flask import Flask, request, jsonify, render_template
from io import BytesIO
from PIL import Image

import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

caption_model = load_model('caption_model.h5')

max_caption_length = 33
inception_v3_model = InceptionV3(weights = 'imagenet', input_shape=(299, 299, 3))
inception_v3_model.layers.pop()
inception_v3_model = Model(inputs=inception_v3_model.inputs, outputs=inception_v3_model.layers[-2].output)
cnn_output_dim = inception_v3_model.output_shape[1]


def extract_image_features(img):
    img = img.resize((299, 299)) 
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) 
    img = tf.keras.applications.inception_v3.preprocess_input(img) 
    features = inception_v3_model.predict(img, verbose=0)
    return features


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def top_k_sampling(predictions, k=5):
    sorted_indices = np.argsort(predictions)[-k:]
    top_k_probs = predictions[sorted_indices] 
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    return np.random.choice(sorted_indices, p=top_k_probs)

def caption_generator_with_sampling(image_features, tokenizer, caption_model, max_caption_length, cnn_output_dim, k=5, temperature=1.0):
    in_text = 'start'
    repeated_word_count = 0
    last_word = ''
    
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length).reshape((1, max_caption_length))
        
        predictions = caption_model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)[0]
        
        idx = top_k_sampling(predictions, k=k)
        word = tokenizer.index_word.get(idx, '')
        
        if word == 'end':
            break
        
        if word == last_word:
            repeated_word_count += 1
            if repeated_word_count > 2:
                break
        else:
            repeated_word_count = 0
        
        in_text += ' ' + word
        last_word = word

    in_text = in_text.replace('start ', '').replace(' end', '')

    return in_text


def beam_search_generator(image_features, K_beams=3, log=False, penalty=1.0):
    start = [tokenizer.word_index['start']]
    
    start_word = [[start, 0.0]]  # List of beams, each beam has (sequence, probability)
    max_repeat = 2               # Max number of times a word can repeat consecutively
    
    for _ in range(max_caption_length):
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=max_caption_length).reshape((1, max_caption_length))
            
            preds = caption_model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)[0]
            
            # Sort predictions by probability and take the top K words
            word_preds = np.argsort(preds)[-K_beams:]
            
            # Add each predicted word to the current sequence and calculate the new score
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                
                # Apply logarithmic scaling if required
                if log:
                    prob += np.log(preds[w] + 1e-8)  # Avoid log(0) by adding a small constant
                else:
                    prob += preds[w]
                
                # Penalty for repeated words
                if len(next_cap) > 1 and next_cap[-1] == next_cap[-2]:
                    prob -= penalty  # Decrease the probability if the same word repeats
                
                temp.append([next_cap, prob])
        
        # Sort candidates by their scores and select the top K beams
        start_word = sorted(temp, reverse=False, key=lambda l: l[1])[-K_beams:]

    # Take the best sequence (highest score) and decode it into words
    best_sequence = start_word[-1][0]
    captions_ = [tokenizer.index_word[i] for i in best_sequence]

    # Build the final caption, ignoring 'end' tokens
    final_caption = []
    for word in captions_:
        if word != 'end':
            final_caption.append(word)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])  # Join words into a sentence, ignoring the 'start' token
    return final_caption


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')



@app.route('/generate_caption', methods=['POST'])
def generate_caption_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img = Image.open(BytesIO(file.read()))
    except Exception as e:
        print(f"Error reading image: {e}")
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        img_features = extract_image_features(img)
        generated_caption = beam_search_generator(img_features)
    except Exception as e:
        print(f"Error generating caption: {e}")
        return jsonify({'error': 'Caption generation failed'}), 500

    return jsonify({'caption': generated_caption})
