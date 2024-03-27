import streamlit as st
import base64
 #F8F8FF
# Ändern Sie das Streamlit-Thema mit benutzerdefinierten Farben
st.markdown(
    """
    <style>
    .stApp {
        background-color: #EAF4F8; /* Hintergrundfarbe */ 
    }
    .stTextInput>div>div>input {
        color: #6A5ACD; /* Textfarbe für Texteingaben */
    }
    .stTextArea>div>div>textarea {
        color: #6A5ACD; /* Textfarbe für Textbereiche */
    }
    .stButton>button {
        background-color: #3A2A78; /* Hintergrundfarbe für Buttons */
        color: white; /* Textfarbe für Buttons */
    }
    .stProgress>div>div>div>div>div {
        background-color: #3A2A78; /* Hintergrundfarbe für Fortschrittsbalken */
    }
    .chat-title {
        color: #1AA469; /* Grün für den Titel */
    }
    .infobox-container {
        padding: 10px;
        background-color: white; /* Weißer Hintergrund */
        border: 2px solid #3A2A78; /* Lianenrand in lila */
        border-radius: 5px;
        font-family: Arial, sans-serif; /* Schriftart */
    }
    .infobox-title {
        color: #3A2A78; /* Lila Schriftfarbe für den Titel */
        font-weight: bold;
    }
    .infobox-text {
        color: black; /* Schwarze Schriftfarbe für den Text */
    }
    .custom-box {
        padding: 10px;
        background-color: black; /* Schwarzer Hintergrund */
        color: white; /* Weiße Schriftfarbe */
        border-radius: 0px; /* Eckige Form */
    }
    .st-eb, .st-dd, .stTextInput>div>div>input{
        background-color: white !important; /* Weißer Hintergrund */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("## <span class='chat-title'>Chatten mit BioGPT</span>", unsafe_allow_html=True)

#import subprocess
#import sys

#def install(transformers):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", transformers ])

#def install(sacremoses):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", sacremoses ])
    
# !pip install transformers
# !pip install sacremoses



# Infobox
st.markdown(
    """
    <div class="infobox-container">
        <p class="infobox-title">Infobox BioGPT</p>
        <p class="infobox-text">BioGPT ist ein spezielles generatives KI-Modell, das für die biomedizinische Texterzeugung und -analyse entwickelt wurde. 
        Es basiert auf dem Transformer-Sprachmodell und wurde von Grund auf mit einer umfangreichen Datenbank von 15 Millionen PubMed-Abstracts vorab trainiert. 
        Diese Datenbank ermöglicht es BioGPT, fundierte Einblicke in komplexe biologische Fragestellungen zu liefern und die biomedizinische Forschung zu unterstützen. 
        Trotz seiner Fortschritte funktioniert BioGPT noch rudimentär und ist nicht in der Lage, wie andere Chatbots alle Anfragen sinnvoll zu bearbeiten. 
        Es reagiert empfindlich auf Inputs und befindet sich noch in der Entwicklungsphase. Dies kannst du testen, indem du deine Eingaben in BioGPT variierst und die Ergebnisse vergleichst. .</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='height: 1cm;'></div>", unsafe_allow_html=True)
# Meldung "Loading model..."
st.markdown("<div class='custom-box'>Loading model...</div>", unsafe_allow_html=True)

from transformers import pipeline
from transformers import BioGptTokenizer, BioGptForCausalLM

model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Meldung "Model loaded."
st.markdown("<div class='custom-box'>Model loaded.</div>", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    """
    <div class="infobox-container">
        <p class="infobox-text">Hier können verschiedene Prompts mit BioGPT getestet werden. .</p>
    </div>
    """,
    unsafe_allow_html=True
)
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

st.markdown("---")
# Abschnitt Code selber generieren
st.markdown(
    """
    <div class="infobox-container">
        <p class="infobox-text">Hier kannst du selber einen Prompt schreiben. .</p>
    </div>
    """,
    unsafe_allow_html=True
)

prompt_text = """
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






