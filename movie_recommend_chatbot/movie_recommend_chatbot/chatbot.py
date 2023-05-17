import pandas as pd
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer

from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))


import streamlit as st
from streamlit_chat import message
import joblib
import re

from transformers import PreTrainedTokenizerFast
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

device = torch.device('cpu')

@st.cache(allow_output_mutation=True)
def cached_model_st():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def cached_model_gpt():
    model = model_gpt = joblib.load('model_gpt.pkl')
    return model

@st.cache(allow_output_mutation=True)
def get_chatbot_data():
    df = pd.read_csv('chatbot_embedded2.csv')
    df['embedding'] = df['embedding'].apply(lambda x: np.array(x[1:-1].split(), dtype=float))
    # df['embedding'] = df['embedding'].apply(json.loads)
    return df

@st.cache(allow_output_mutation=True)
def get_movies_data():
    df = pd.read_csv('movies_embedded2.csv')
    df['embedding'] = df['embedding'].apply(lambda x: np.array(x[1:-1].split(), dtype=float))
    # df['embedding'] = df['embedding'].apply(json.loads)
    return df

@st.cache(allow_output_mutation=True)
def get_emo_dict():
    with open('emo_dict.pkl','rb') as f:
      dic = pickle.load(f)
    return dic

@st.cache(allow_output_mutation=True)
def make_list():
    li = []
    return li

model_st = cached_model_st()
model_gpt = cached_model_gpt()
chatbot_data = get_chatbot_data()
movies_data = get_movies_data()
emo_dict = get_emo_dict()
emotions = make_list()

st.header('Movie Recommand Chatbot')
st.markdown("힘드시조..?")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('user> ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedded = model_st.encode(user_input)

    chatbot_data['score'] = chatbot_data.apply(lambda x: cos_sim(x['embedding'], embedded), axis=1)
    genre = chatbot_data.iloc[chatbot_data['score'].idxmax(), 1]
    emotion = emo_dict[genre]

    # if st.session_state['past'] == []:
    #     first_emotion = emotion
    #     first_genre = genre
    emotions.append([emotion, genre])
    
    sent = str(genre)
    
    if re.findall('(?=.*영화)(?=.*추천)', user_input):  # 영화, 추천 모두 들어간다면
    # if user_input == '영화 추천해줘.':
        print(f'bot> {emotions[0][0]} 때 보기 좋은 영화 추천해 드릴게요.')

        df_movies = movies_data[movies_data['영화'] == emotions[0][1]]
        df_movies['score'] = df_movies.apply(lambda x: cos_sim(x['embedding'], embedded), axis=1)

        top3 = df_movies.sort_values('score', ascending=False).iloc[:3, 0]

        a = []
        for i, t in enumerate(top3.values):
            a.append(f'{i+1}. {t}')
            answer = '\n'.join(a)
        answer = f'{emotions[0][0]} 때 보기 좋은 영화 추천해 드릴게요. >>> ' + answer

    else:
        a = ""
        while(1):
          input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + user_input + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
          pred = model_gpt(input_ids)
          pred = pred.logits
          gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
          if gen == EOS:
            break
          a += gen.replace("▁", " ")
        answer = a.strip()
    
    
    
    st.session_state.past.append(user_input)
    # st.session_state.generated.append(answer['챗봇'])
    st.session_state.generated.append(answer)

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')