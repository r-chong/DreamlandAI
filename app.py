# Bring in deps and set up environment variables
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
import time
import uuid


st.set_page_config(page_title="StoryBookGPT", page_icon="🏰", layout="wide")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

gameOver = False

os.environ['OPENAI_API_KEY'] = apikey

# Prompt templates
intro_template = PromptTemplate(
	# story_concept is the general idea for the story
	# the purpose of intro_template is to expand upon the story_concept like an opening crawl to make it so that it doesn't seem like you are just starting in the middle of the story
	input_variables  = ['story_concept'],
	template='Write a short and mysterious exposition to a story based on this concept: {story_concept}. Ensure description is in second-person.'
)

middle_template = PromptTemplate(
	# this is the continuation of the story, which will be called multiple times
	# it will be called after the exposition and after each continuation until the middle of the story ends
	# ------------------
	# referenced as {current_state}
	input_variables  = ['prev_state'],
	template="""Based on what happened in {prev_state}, write a second-person description of what happens next in the story. Do not attempt to end the story.
		Then give the user three decisions the user can make based on the current state of the story.
	""",
	validate_template=False
)

format_decision_template = PromptTemplate(
	input_variables=['current_state'],
	template="""format the decisions in {current_state} as a list of three choices, each separated by a comma. For example, if the decisions are "go left", "go right", and "go straight", then the format should be: go left, go right, go straight
	"""
)

# Llms
llm = OpenAI(temperature=0.6)
intro_chain = LLMChain(llm=llm,prompt=intro_template,output_key='exposition')

middle_chain = LLMChain(llm=llm,prompt=middle_template,verbose=True,output_key='current_state')
format_decision_chain = LLMChain(llm=llm,prompt=format_decision_template,verbose=True,output_key='decisions')

overall_chain = SequentialChain(
	chains=[middle_chain,format_decision_chain],
	input_variables=['prev_state'],
	output_variables=['current_state','decisions'],
	verbose=True,
)

# -------------------

def continue_story(prev_state, isExposition):
	placeholder = st.empty()

	response = overall_chain({'prev_state':prev_state})
	st.write(response['current_state'])
	arr = response['decisions'].split(", ")

	if isExposition == False:
		with placeholder.container():
			if st.button(arr[0], key=uuid.uuid4()):
				st.write(f"{arr[0]} clicked!")

			if st.button(arr[1], key=uuid.uuid4()):
				st.write(f"{arr[1]} clicked!")
			
			if st.button(arr[2], key=uuid.uuid4()):
				st.write(f"{arr[2]} clicked!")

	prev_state=response

# Display
st.title('🌈🏰🌟 Dreamland GPT')
prompt = st.text_input('Plug your story idea in here')
if prompt:
	# introduction
	exposition = intro_chain.run(story_concept=prompt)
	st.write("This is the opening crawl\n" + exposition)
	st.divider()

	prev_state=exposition

	# bulk of the story
	for i in range(5):
		if i == 0:
			continue_story(prev_state,True)
		else:
			continue_story(prev_state,False)
		placeholder = st.empty()
		user_choice = st.text_input("",placeholder='What will you do next?', key=uuid.uuid4())

	


