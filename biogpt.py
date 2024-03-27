import streamlit as st
import base64

# Ändern Sie das Streamlit-Thema mit benutzerdefinierten Farben
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8F8FF; /* Hintergrundfarbe */
    }
    .stTextInput>div>div>input {
        color: #6A5ACD; /* Textfarbe für Texteingaben */
    }
    .stTextArea>div>div>textarea {
        color: #6A5ACD; /* Textfarbe für Textbereiche */
    }
    .stButton>button {
        background-color: #6A5ACD; /* Hintergrundfarbe für Buttons */
        color: white; /* Textfarbe für Buttons */
    }
    .stProgress>div>div>div>div>div {
        background-color: #6A5ACD; /* Hintergrundfarbe für Fortschrittsbalken */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("## Chatten mit BioGPT")

#import subprocess
#import sys

#def install(transformers):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", transformers ])

#def install(sacremoses):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", sacremoses ])
    
# !pip install transformers
# !pip install sacremoses

# Infobox anzeigen
# Info-Box mit lila Hintergrund und lokalem Info-Symbol
with open("320px-Infobox_info_icon.svg.png", "rb") as img_file:
    img_str = base64.b64encode(img_file.read()).decode("utf-8")

st.markdown(
    f"""
    <div style="padding: 10px; background-color: #D2CBED; border-radius: 5px;">
        <p style="font-weight: bold;">Infobox BioGPT</p>
        <p>BioGPT ist ein spezielles generatives KI-Modell, das für die biomedizinische Texterzeugung und -analyse entwickelt wurde. 
        Es basiert auf dem Transformer-Sprachmodell und wurde von Grund auf mit einer umfangreichen Datenbank von 15 Millionen PubMed-Abstracts vorab trainiert. 
        Diese Datenbank ermöglicht es BioGPT, fundierte Einblicke in komplexe biologische Fragestellungen zu liefern und die biomedizinische Forschung zu unterstützen. 
        Trotz seiner Fortschritte funktioniert BioGPT noch rudimentär und ist nicht in der Lage, wie andere Chatbots alle Anfragen sinnvoll zu bearbeiten. 
        Es reagiert empfindlich auf Inputs und befindet sich noch in der Entwicklungsphase. Dies kannst du testen, indem du deine Eingaben in BioGPT variierst und die Ergebnisse vergleichst. .</p>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("Loading model...")

from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
import sacremoses


model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
#st.markdown("Model set.")

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
#st.markdown("Tokenizer set.")

generator = pipeline("text-generation",model=model,tokenizer=tokenizer)
#st.markdown("Generator set.")

st.markdown("Model loaded.")

st.markdown("---")

st.markdown( "Hier können verschiedene Prompts mit BioGPT getestet werden." )
prompt_list_dropdown = ["Wähle Prompt",
                        "Prompt 1: Generiere 5 Antworten für die Eingabe 'Covid is ...'", 
                        "Prompt 2: Beantworte die Frage: 'What are the symptoms of a migraine?'", 
                        """Prompt 3: Symptomchecker: 
                        Intense headache often accompanied by nausea, vomiting, and sensitivity to light and sound""", 
                        #"Prompt 4: ..."
                       ]

prompt_option = st.selectbox("Prompt Auswahl", prompt_list_dropdown)

#st.markdown("Du hast " + prompt_option + " gewählt.")
#st.markdown("---")

# Aktion basierend auf dem ausgewählten Prompt

if prompt_option.startswith("Prompt 1"):

    input_text= "COVID-19 is"
    #st.markdown("Input text: " + input_text)
    
    output = generator(input_text, max_length=20, num_return_sequences=5, do_sample=True)
    st.markdown("Antwort von BioGPT: ")

    for item in output:
        st.markdown(f"- {item['generated_text']}")

elif prompt_option.startswith("Prompt 2"):

    input_text= """ question: What are the symptoms?
                context: I would like to know the symptoms of migraine
                answer: The symptoms of a migraine are  """
    output = generator(input_text , max_length=200, num_return_sequences=1, do_sample=False)

    st.markdown("Antwort von BioGPT: ")
    for item in output:
        answer_start = item['generated_text'].find('answer:')
        if answer_start != -1:
            answer_text = item['generated_text'][answer_start + len('answer:'):].strip()
            st.markdown(answer_text)
   
            
elif prompt_option.startswith("Prompt 3"):
    
    input_text= """ question: What is the name of the disease?
                context: Symptoms: Intense headache often accompanied by nausea, vomiting, and sensitivity to light and sound. Some people also experience visual disturbances known as auras, such as seeing flashing lights or zigzag lines.
                answer: the disease is called """
    output = generator(input_text , max_length=200, num_return_sequences=1, do_sample=False)

    st.markdown("Antwort von BioGPT: ")
    for item in output:
        answer_start = item['generated_text'].find('answer:')
        if answer_start != -1:
            answer_text = item['generated_text'][answer_start + len('answer:'):].strip()
            st.markdown(answer_text)
    
elif prompt_option.startswith("Prompt 4"):
    
    st.markdown("Antwort von BioGPT: ")
    st.write("Dies ist der vierte Prompt")

# Abschnitt Code selber generieren
st.markdown("---")

prompt_text = """
Hier kannst du selber einen Prompt schreiben. 
- Schreibe den Anfang einen Satzes und lasse das Model deinen Satz vervollständigen, siehe Beispiel Prompt 1.
- Beachte auch, dass das Model nur Englisch versteht.
- Beispiel: 'Covid-19 is', 'Migraine has the following symptoms:'
"""

st.markdown(prompt_text)
                
input_text = st.text_area("Gib hier deinen Satzanfang ein:", "")
# Textfeld für die Anzahl der Outputs
num_outputs = st.text_input("Anzahl der generierten Antworten (1-10):", "3", max_chars=2)

# Überprüfe, ob die eingegebene Zahl zwischen 1 und 10 liegt
if num_outputs.isdigit() and 1 <= int(num_outputs) <= 10:
    num_outputs = int(num_outputs)
else:
    st.warning("Bitte geben Sie eine Zahl zwischen 1 und 10 ein.")

# Funktion zum Generieren des Texts
def generate_text(input_text):
    if input_text:
        
        output = generator(input_text, max_length=20, num_return_sequences=num_outputs, do_sample=True)
            
        return output
    else:
        return None

# Button zum Generieren des Texts
if st.button("Generieren"):
    generated_text = generate_text(input_text)
    if generated_text:
        st.markdown("Antwort von BioGPT: ")
        for item in generated_text:
            st.markdown(f"- {item['generated_text']}")
    else:
        st.warning("Bitte geben Sie einen Text ein, um fortzufahren.")

# Ende des Streamlit Seitenlayouts
#set_seed(42)
#st.markdown("Seed set. Let's go!")






