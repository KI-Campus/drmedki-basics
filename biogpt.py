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

st.markdown("---")

prompt_list_dropdown = ["1 ) Generiere 5 answers 'Covid is ...'", 
                        "2) Generate answer to question 'What is ...'", 
                        "Prompt 3", 
                        "Prompt 4"
                       ]

prompt_option = st.selectbox("Prompt Auswahl", prompt_list_dropdown)

st.markdown("Du hast " + prompt_option + " gewählt.")

User
ich habe folgenden code und möchte je nach prompt 1,2,3 oder 4 andere ausgaben in streamlit geben. ZB bei prompt 1 soll es mir 1+1 ausrechnen. wenn ich prompt 2 selektiere, soll es mir "hello word ausgeben": st.markdown( "Hier können verschiedene Prompts mit BioGPT getestet werden." )

from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
import sacremoses

st.markdown("---")

prompt_list_dropdown = ["1) Generiere 5 answers 'Covid is ...'"", 
                        "2) Generate answer to question 'What is ...'", 
                        "3) Prompt 3", 
                        "4) Prompt 4"
                       ]

prompt_option = st.selectbox("Prompt Auswahl", prompt_list_dropdown)

st.markdown("Du hast " + prompt_option + " gewählt.")


st.markdown("---")

# Aktion basierend auf dem ausgewählten Prompt
if prompt_option.startswith("1"):
    # Code für den ersten Prompt: Berechnung von 1+1
    result = 1 + 1
    st.write("Das Ergebnis von 1 + 1 ist:", result)
elif prompt_option.startswith("2"):
    # Code für den zweiten Prompt: Ausgabe von "Hello World"
    st.write("Hello World!")
elif prompt_option.startswith("3"):
    # Code für den dritten Prompt
    st.write("Dies ist der dritte Prompt")
elif prompt_option.startswith("4"):
    # Code für den vierten Prompt
    st.write("Dies ist der vierte Prompt")

# Ende des Streamlit Seitenlayouts
st.markdown("---")

st.markdown("---")


model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
st.markdown("Model set.")

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
st.markdown("Tokenizer set.")

generator = pipeline("text-generation",model=model,tokenizer=tokenizer)
st.markdown("Generator set.")

#set_seed(42)
#st.markdown("Seed set. Let's go!")

st.markdown("---")


input_text= "COVID-19 is"
st.markdown("Input text: " + input_text)


output = generator(input_text, max_length=20, num_return_sequences=5, do_sample=True)
st.markdown(output)



