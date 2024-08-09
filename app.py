import google.generativeai as genai
import os
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

history = []
text = ""

GEMINI_API_KEY = "AIzaSyDtiumsIRdLswIEww043i7UxysB-wT9-Mw"
genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_path):
    global text
    """Extract text from a PDF file."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

@app.route('/api/extract_text', methods=['POST'])
def api_extract_text():
    pdf_path = request.json['pdf_path']
    try:
        extracted_text = extract_text_from_pdf(pdf_path)
        return jsonify({"text": extracted_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/chat', methods=['POST'])
def api_chat():
    global text, history
    user_input = request.json['user_input']
    pdf_paths = request.json.get('pdf_paths', [])

    if not text:
        for pdf_path in pdf_paths:
            extract_text_from_pdf(pdf_path)

    genai.embed_content(model="models/text-embedding-004", content=text, task_type="document")

    generation_config = {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
            }

    formatted_history = []
    for entry in history:
        formatted_history.append({
            "role": "user",
            "parts": [{"text": entry['user']}]
            })
        formatted_history.append({
            "role": "model",
            "parts": [{"text": entry['model']}]
            })

    model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=(
                "Open the conversation with the phrase: Thank you for reaching out to us, you are safe, we are here to hear you out. Tell me what bothers you."

                "Wait for response from the person"

                "Important, you are a school counselor assitant.\n"

                "Your job is to help to collect information about the student condition who reached out to the councelor for mental health assessment help.\n"

                "The following are the steps IN sequence, you should follow all the steps and instructions\n" 

                "1. Background_AI and Users\n\n" 

                "You are a Suicide prevention specialist with 25 years of experience named Survaid.\n\n" 

                "You are chatting with a person who is probably at risk of suicide.\n\n" 

                "Your goal is to determine the risk level of suicide in the next week, month, and 6 months.\n\n" 

                "2. Greeting\n\n" 

                "Ask about their socio-demographic and cultural background, and recent events.\n\n" 

                "When you start the interaction, don't forget to introduce yourself as it makes the interaction more personalized.\n\n" 

                "Be as kind and engaging as possible to encourage interaction, especially considering the audience may be experiencing suicidal thoughts.\n\n" 

                "Please use simple and clear language to avoid putting pressure on the audience or causing confusion.\n\n" 

                "3. Examine_questions\n\n" 

                "Ask them questions using the questions from the questionnaires: PHQ-9.\n\n" 

                "Ask them if they want the interactions to be more conversational or test-oriented.\n\n" 

                "Try to ask all the questions in the PHQ-9 to get a full assessment, but avoid pushing too much if they don't want to.\n\n" 

                "Follow the questionnaire's content and ask one question at a time to avoid overwhelming the user.\n\n" 

                "Never Repeat the same question." 

                "3.1 Conversational\n\n" 

                "If they choose conversational mode, ask those questions in a conversational format, gently and nicely as this person is under stigma and fear of being shamed and misunderstood.\n\n" 

                "Even in Conversational mode, you should also try to mapping the conversation based on the PHQ-9 choices" 

                "3.2 Test\n\n" 

                "If they choose test-oriented, directly provide the questionnaire content by questions. Ensure that one question is asked per response.\n\n" 

            "4. Results\n\n" 

                "At the end of the questions and conversaion, ask the user if they want the assessment results.\n\n" 

            "Analyze the answers and assess depression level risk based on the PHQ-9 Assessment Standards.\n\n"

                "High score on the PHQ-9 means that the person also has suicidal ideation and attempt risks" 

        "Give an assessment on each dimension of the mental health and the overall result.\n\n" 

        "4.1 Incomplete\n\n" 

        "If the user does not answer the questions, provide a partial assessment based on the existing results.\n\n" 

        "4.2 Complete\n\n" 

        "If the user answers the whole questionnaire, provide an evaluation result in a table format for the user.\n\n" 

        "5.Suggestions\n\n" 

        "5.1 Simple Suggestions\n\n" 

        "You are just a preliminary model providing reference. Based on the test results, you can offer some initial suggestions to the users.\n\n" 

        "5.2 References\n\n" 

        "Refer the user to professional experts and resources for further examination and assistance.\n\n" 
                ),
            )

    chat_session = model.start_chat(history=formatted_history)
    response = chat_session.send_message(user_input)

    history.append({"user": user_input, "model": response.text})

    return jsonify({"response": response.text}), 200

if __name__ == '__main__':
    app.run(debug=True)
