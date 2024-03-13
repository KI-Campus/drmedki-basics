import streamlit as st

st.markdown("## Chatten mit BioGPT")

!pip install transformers
# !pip install sacremoses


st.markdown( "Hier können verschiedene Prompts mit BioGPT getestet werden." )

from transformers import pipeline

st.markdown("---")


prompt_list_dropdown = ["Prompt 1", 
                        "Prompt 2", 
                        "Prompt 3", 
                        "Prompt 4"
                       ]

prompt_option = st.selectbox("Prompt Auswahl", prompt_list_dropdown)

st.markdown("Du hast " + prompt_option + " gewählt.")
