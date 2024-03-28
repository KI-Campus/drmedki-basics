import streamlit as st
import base64
from transformers import pipeline, BioGptTokenizer, BioGptForCausalLM
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
        margin-bottom: 0px; /* Hier kannst du den Abstand anpassen */
    }
    .infobox-title {
        color: #3A2A78; /* Lila Schriftfarbe für den Titel */
        font-weight: bold;
    }
    .infobox-text {
        color: black; /* Schwarze Schriftfarbe für den Text */
    }
    .infobox-source {
        font-size: smaller; /* Kleinere Schriftgröße für die Quellenangabe */
        margin-top: 5px; /* Abstand nach oben */
    }
    .custom-box {
        padding: 10px;
        background-color: black; /* Schwarzer Hintergrund */
        color: white; /* Weiße Schriftfarbe */
        border-radius: 0px; /* Eckige Form */
    }
    .st-eb, .st-dd, .stTextInput>div>div>input {
        background-color: white !important; /* Weißer Hintergrund */
        color: black !important; /* Schwarzer Text */
    }
    .st-dd::before {
        color: black !important; /* Schwarzer Text für den Dropdown-Pfeil */
    }
    .st-dd-options {
        color: black !important; /* Schwarzer Text für die Dropdown-Optionen */
    }
    .st-dd-container {
        background-color: white !important; /* Weißer Hintergrund für das Dropdown-Menü */
    }
    .st-dd-options>li.stOptionSelected {
    background-color: green !important; /* Grüner Hintergrund für ausgewählte Option */
    color: white !important; /* Weißer Text für die ausgewählte Option */
    }
    .st-dd-options>li.stOptionSelected:focus {
        outline-color:  #1AA469 !important; /* Grüner Fokus-Rand */
    }
    .stButton:focus {
        outline-color:  #1AA469 !important; /* Grüner Fokus-Rand */
    }
    div[data-baseweb="select"] > div {
    background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Laden des Modells und der Tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    st.markdown("<div class='custom-box'>Loading model...</div>", unsafe_allow_html=True)
    model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    st.markdown("<div class='custom-box'>Model loaded.</div>", unsafe_allow_html=True)
    return generator

# Definiere die Optionen für die Selectbox
prompt_list_dropdown = ["Wähle Prompt",
                        "Prompt 1: Generiere 5 Antworten für die Eingabe 'Covid is ...'", 
                        "Prompt 2: Beantworte die Frage: 'What are the symptoms of a migraine?'", 
                        """Prompt 3: Symptomchecker: 
                        Intense headache often accompanied by nausea, vomiting, and sensitivity to light and sound""", 
                        #"Prompt 4: ..."
                       ]





# Hauptfunktionsstruktur
def main():

    st.markdown("## <span class='chat-title'>Chatten mit BioGPT</span>", unsafe_allow_html=True)
   
    # Infobox
    st.markdown(
       """
       <div class="infobox-container">
           <p class="infobox-title">Infobox BioGPT</p>
           <p class="infobox-text">BioGPT ist ein spezielles generatives KI-Modell, das für die biomedizinische Texterzeugung und -analyse entwickelt wurde. 
           Es basiert auf dem Transformer-Sprachmodell und wurde von Grund auf mit einer umfangreichen Datenbank von 15 Millionen PubMed-Abstracts vorab trainiert. 
           Im Vergleich zu anderen großen Sprachmodellen wie GPT-3 wurde BioGPT damit auf viel weniger Daten trainiert. 
           Deshalb funktioniert BioGPT teilweise rudimentär, ist nicht in der Lage, wie andere Chatbots alle Anfragen sinnvoll zu bearbeiten 
           und reagiert empfindlich auf Inputs. 
           Dennoch ermöglicht diese Datenbank es BioGPT, fundierte Einblicke in komplexe biologische Fragestellungen zu liefern und die biomedizinische Forschung zu unterstützen. 
           Du kannst dies testen, indem du deine Eingaben in BioGPT variierst und die Ergebnisse vergleichst.</p>
           <p class="infobox-source">Quelle: Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, Tie-Yan Liu, BioGPT: 
           generative pre-trained transformer for biomedical text generation and mining, Briefings in Bioinformatics, Volume 23, Issue 6, November 2022, bbac409, https://doi.org/10.1093/bib/bbac409</p>
       </div>
       """,
       unsafe_allow_html=True
    )

    # Überprüfe, ob das Modell bereits in der Session gespeichert ist, andernfalls lade es
    if 'generator' not in st.session_state:
        st.session_state.generator = load_model()

    # Erhalte den ausgewählten Prompt aus der Session State oder setze ihn auf den Standardwert
    prompt_option = st.session_state.get('prompt_option', prompt_list_dropdown[0])

    # Streamlit-Elemente
    st.markdown(
        """
        <div class="infobox-container">
            Hier können verschiedene Prompts mit BioGPT getestet werden.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dropdown-Selectbox für die Prompt-Auswahl
    prompt_option = st.selectbox("Prompt Auswahl", prompt_list_dropdown)

    # Speichere die ausgewählte Option in der Session State
    st.session_state.prompt_option = prompt_option

    # Aktion basierend auf dem ausgewählten Prompt, wenn der "Generieren"-Button gedrückt wird
    if st.button("Generieren"):
        generate_response(prompt_option)

# Funktion zum Generieren der Antwort basierend auf dem ausgewählten Prompt
def generate_response(prompt_option):
    generator = st.session_state.generator  # Lade das Modell aus der Session State
    input_text = None  # Setze den Eingabetext initial auf None
    # Führe die entsprechende Aktion basierend auf dem ausgewählten Prompt aus
    if prompt_option.startswith("Prompt 1"):
        input_text = "COVID-19 is"
    elif prompt_option.startswith("Prompt 2"):
        input_text = """ question: What are the symptoms?
                    context: I would like to know the symptoms of migraine
                    answer: The symptoms of a migraine are  """
    elif prompt_option.startswith("Prompt 3"):
        input_text = """ question: What is the name of the disease?
                    context: Symptoms: Intense headache often accompanied by nausea, vomiting, and sensitivity to light and sound. Some people also experience visual disturbances known as auras, such as seeing flashing lights or zigzag lines.
                    answer: the disease is called """
    elif prompt_option.startswith("Prompt 4"):
        st.write("Dies ist der vierte Prompt")
        return  # Beende die Funktion, wenn Prompt 4 ausgewählt ist

    if input_text:
        output = generator(input_text, max_length=20, num_return_sequences=5, do_sample=True)
        st.markdown("Antwort von BioGPT: ")
        for item in output:
            st.markdown(f"- {item['generated_text']}")

# Starte die Anwendung
if __name__ == "__main__":
    main()
