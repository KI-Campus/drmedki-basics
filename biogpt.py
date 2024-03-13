import streamlit as st

st.markdown("## Chatten mit BioGPT")

import subprocess
import sys

def install(transformers):
    subprocess.check_call([sys.executable, "-m", "pip", "install", transformers ])

def install(sacremoses):
    subprocess.check_call([sys.executable, "-m", "pip", "install", sacremoses ])
    
# !pip install transformers
# !pip install sacremoses


st.markdown( "Hier können verschiedene Prompts mit BioGPT getestet werden." )

from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
import sacremoses


st.markdown("Loading model...")

model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
#st.markdown("Model set.")

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
#st.markdown("Tokenizer set.")

generator = pipeline("text-generation",model=model,tokenizer=tokenizer)
#st.markdown("Generator set.")

st.markdown("Model loaded.")

st.markdown("---")

prompt_list_dropdown = ["Wähle Prompt",
                        "Prompt 1: Generiere 5 Antworten für die Eingabe 'Covid is ...'", 
                        "Prompt 2: Beantworte mir die Frage 'What are the symptoms of migraine?'", 
                        "Prompt 3: ...", 
                        "Prompt 4: ..."
                       ]

prompt_option = st.selectbox("Prompt Auswahl", prompt_list_dropdown)

#st.markdown("Du hast " + prompt_option + " gewählt.")
#st.markdown("---")

# Aktion basierend auf dem ausgewählten Prompt

if prompt_option.startswith("Prompt 1"):

    input_text= "COVID-19 is"
    #st.markdown("Input text: " + input_text)
    
    output = generator(input_text, max_length=20, num_return_sequences=5, do_sample=True)
    st.markdown("Answer of BioGPT: ")
    st.markdown(output)

elif prompt_option.startswith("Prompt 2"):
    # Code für den zweiten Prompt: Ausgabe von "Hello World"
    st.write("Hello World!")
elif prompt_option.startswith("Prompt 3"):
    # Code für den dritten Prompt
    st.write("Dies ist der dritte Prompt")
elif prompt_option.startswith("Prompt 4"):
    # Code für den vierten Prompt
    st.write("Dies ist der vierte Prompt")

# Ende des Streamlit Seitenlayouts
st.markdown("---")


#set_seed(42)
#st.markdown("Seed set. Let's go!")






