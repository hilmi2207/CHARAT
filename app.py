# import files
from email import message
from urllib import response
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
app = Flask(__name__)

df = pd.read_csv('Dataset.csv', delimiter=";")
questions = df.Question.values
answers = df.Answer.values

import re
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'" - "+', "", text)
    text = re.sub(r'" +"', " ", text)
    text = re.sub(r'"^ "', "", text)
    text = re.sub(r'[-()\"#/@;:<>{}`+=~|.!?,]', "", text)
    text = re.sub(r'[\-\-]', "", text)
    text = re.sub(r'\.\.\.', "", text)
    text = re.sub(r'^- ', "", text)
    return text

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
#clean_answers = []
#for answer in answers:
#    clean_answers.append(clean_text(answer))


word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

print(len(word2count))

vocab = {}
word_num = 0
for word, count in word2count.items():
        vocab[word] = word_num
        word_num += 1

print(len(vocab))

for i in range(len(answers)):
    answers[i] = '<SOS> ' + answers[i] + ' <EOS>'

tokens = ['<EOS>', '<SOS>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1

inv_vocab = {w:v for v, w in vocab.items()}

encoder_inp = []
for line in clean_questions:
    lst = []
    for word in line.split():
        lst.append(vocab[word])
        
    encoder_inp.append(lst)
decoder_inp = []
for line in answers:
    lst = []
    for word in line.split():      
      lst.append(vocab[word])        
    decoder_inp.append(lst)
        
max_input_len = 261
lstm_layers = 256
VOCAB_SIZE = len(vocab)

encoder_inp = pad_sequences(encoder_inp, maxlen = max_input_len, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, maxlen = max_input_len, padding='post', truncating='post')

decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:])

decoder_final_output = pad_sequences(decoder_final_output, maxlen = max_input_len, padding='post', truncating='post')
decoder_final_output = to_categorical(decoder_final_output, VOCAB_SIZE)

embedding = Embedding(VOCAB_SIZE + 1,
                      output_dim = 128, 
                      input_length = max_input_len,
                      trainable = True,
                      mask_zero = True 
                      )
e_input = Input(shape=(max_input_len, ))
e_embeded = embedding(e_input)
e_lstm = LSTM(lstm_layers, return_sequences=True, return_state=True)
e_op, h_state, c_state = e_lstm(e_embeded)
e_states = [h_state, c_state]

d_input = Input(shape=(max_input_len, ))
d_embeded = embedding(d_input)
d_lstm = LSTM(lstm_layers, return_sequences=True, return_state=True)
d_output, _, _ = d_lstm(d_embeded, initial_state = e_states)
d_dense = Dense(VOCAB_SIZE, activation='softmax')
d_outputs = d_dense(d_output)

model = Model([e_input, d_input], d_outputs)

#model.compile(optimizer='adam', 
              #loss='categorical_crossentropy', 
              #metrics=['acc'])

#history = model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=70, validation_split=0.2)

######### Decode  #############

######### Encoder #############
enc_model = Model([e_input], e_states)

######### Decoder #############
d_input_h = Input(shape=(lstm_layers,))
d_input_c = Input(shape=(lstm_layers,))
d_states_inputs = [d_input_h, d_input_c]
d_outputs, d_state_h, d_state_c = d_lstm(d_embeded, initial_state = d_states_inputs)
d_states = [d_state_h, d_state_c]
dec_model = Model([d_input] + d_states_inputs, [d_outputs] + d_states)

#chatbot = ChatBot('ChatBot')

enc_model.load_weights('enc_model.h5')
dec_model.load_weights('dec_model.h5')
model.load_weights('main_model.h5')
def input_sentence(text):
    user_ = clean_text(text)
    user = [user_]

    inp_sentence = []
    for sentence in user:
        lst = []
        for y in sentence.split():
            
                lst.append(vocab[y])
            
        inp_sentence.append(lst)
    
    inputs_sentence = pad_sequences(inp_sentence, max_input_len, padding='post')
    states_value = enc_model.predict(inputs_sentence)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = vocab['<SOS>']

    return target_seq, states_value


@app.route("/")
def home():
    return render_template("index.html")

#from chatterbot import ChatBot

@app.post("/predict")
def predict():
    
    userText = request.get_json().get('message')
    #model = "main_model.h5"
    
    

    while userText != 'quit':
        
        
        

        stop_condition = False
        for y in userText.split():
            if y not in vocab:
                decoded = "Maaf, saya tidak memahami maksud anda."
                
                stop_condition = True
                break
            else:
                continue
        

        while not stop_condition :
            target_seq, states_value = input_sentence(userText)
            decoded = ''
            output_tokens , h, c= dec_model.predict([target_seq] + states_value )
            input_tokens = d_dense(output_tokens)
            word_index = np.argmax(input_tokens[0, -1, :])

        ### Invers dari angka ke word index dalam vocab ###
        
            word = inv_vocab[word_index] + ' '
            if inv_vocab[0] and word != '<EOS> ':
                decoded += word
            elif inv_vocab[0] and word != '<EOS> ' and re.search("[^0-9]\.", inv_vocab[word_index+1]):
                decoded += word  
      
            if re.search("[0-9]\.", inv_vocab[word_index+1]):
                decoded += '\n'
            elif word == '<EOS> ':
                stop_condition = True 
            

            target_seq = np.zeros((1 , 1))  
            target_seq[0 , 0] = word_index
            states_value = [h, c]
            
        message = {"answer": decoded}    
        return jsonify(message)


if __name__ == "__main__":
    app.run()
