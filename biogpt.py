import streamlit as st

st.markdown("## Chatten mit BioGPT")

#import subprocess
#import sys

#def install(transformers):
 #   subprocess.check_call([sys.executable, "-m", "pip", "install", transformers ])

#def install(sacremoses):
 #   subprocess.check_call([sys.executable, "-m", "pip", "install", sacremoses ])
    
# !pip install transformers
# !pip install sacremoses


st.markdown( "Hier können verschiedene Prompts mit BioGPT getestet werden." )

from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
import sacremoses

st.markdown("---")

prompt_list_dropdown = ["Prompt 1", 
                        "Prompt 2", 
                        "Prompt 3", 
                        "Prompt 4"
                       ]

prompt_option = st.selectbox("Prompt Auswahl", prompt_list_dropdown)

st.markdown("Du hast " + prompt_option + " gewählt.")

st.markdown("---")


model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
st.markdown("Model setting.")

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
st.markdown("Tokenizer set.")

generator = pipeline("text-generation",model=model,tokenizer=tokenizer)
#generator = pipeline("text-generation", model="BioMistral/BioMistral-7B")
st.markdown("Generator set.")

#set_seed(42)
#st.markdown("Seed set. Let's go!")

st.markdown("---")


input_text= "COVID-19 is"
st.markdown("Input text: " + input_text)


output = generator(input_text, max_length=20, num_return_sequences=5, do_sample=True)
st.markdown(output)



