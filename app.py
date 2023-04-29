# Bring in deps and set up environment variables
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Display
st.set_page_config(page_title="🦜️🔗", page_icon="🦜️🔗", layout="wide")
st.title('🦜️🔗️ Youtube GPT Creator')
prompt = st.text_input('Plug your video topic in here:')

os.environ['OPENAI_API_KEY'] = apikey

title_template = PromptTemplate(
	input_variables  = ['video_topic'],
	template='write me a youtube video title about {video_topic}'
)

# Llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm,prompt=title_template,verbose=True)

# if prompt then show to screen
if prompt:
	response = title_chain.run(video_topic=prompt)
	st.write(response)