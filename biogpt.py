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
                        "Prompt 2: Beantworte mir die Frage 'What are the symptoms of a migraine?'", 
                        "Prompt 3: Beantworte mir, um welche Krankheit es sich handelt bei folgenden Symptomen: Intense headache often accompanied by nausea, vomiting, and sensitivity to light and sound", 
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

    for item in output:
        st.markdown(f"- {item['generated_text']}")

elif prompt_option.startswith("Prompt 2"):
    
    input_text= """ question: What is the name of the disease?
                context: Symptoms: Intense headache often accompanied by nausea, vomiting, and sensitivity to light and sound. Some people also experience visual disturbances known as auras, such as seeing flashing lights or zigzag lines.
                answer: the disease is called """
    output = generator(input_text , max_length=200, num_return_sequences=1, do_sample=False)

    st.markdown("Answer of BioGPT: ")
    for item in output:
        answer_start = item['generated_text'].find('answer:')
        if answer_start != -1:
            answer_text = item['generated_text'][answer_start + len('answer:'):].strip()
            st.markdown(answer_text)
            
elif prompt_option.startswith("Prompt 3"):
    
    st.markdown("Answer of BioGPT: ")
    st.write("Dies ist der dritte Prompt")
    
elif prompt_option.startswith("Prompt 4"):
    
    st.markdown("Answer of BioGPT: ")
    st.write("Dies ist der vierte Prompt")

# Ende des Streamlit Seitenlayouts
st.markdown("---")


#set_seed(42)
#st.markdown("Seed set. Let's go!")






