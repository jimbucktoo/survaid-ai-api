import google.generativeai as genai
import os
import fitz  # PyMuPDF


# Global vars
history = []
text = ""


# Credentials
GEMINI_API_KEY = "AIzaSyDtiumsIRdLswIEww043i7UxysB-wT9-Mw"
genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_path):
    global text
    """Extract text from a PDF file."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
    return text



def main_menu():
    while True:
        print('1. Proceed with the chat\n2. Exit the chat')
        choice = int(input('Enter 1 or 2 > '))
        if choice == 1:
            chat()
        elif choice == 2:
            print('Thank you for visiting. Goodbye!')
            break
        else:
            print('Invalid choice. Please select 1 or 2.')
    return True


def chat():
    global text, history
    print("Thank you for reaching out to us, you are safe, we are here to hear you out.")

    if not text:
        pdf_path = ['/Users/marinazub/Desktop/Ai for MH/coding/PHQ9.pdf', 
                    '/Users/marinazub/Desktop/Ai for MH/coding/SBQ-R.pdf']
                              
        for i in pdf_path:
            extract_text_from_pdf(pdf_path)

    genai.embed_content(model="models/text-embedding-004", content = text, task_type="document") #embedding Try to add extrac knowledge


    # Create the model
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=(
                  "The following are the steps IN sequence, you should follow all the steps and instructions\n" 

        "1. Background_AI and Users\n\n" 

        "You are a Suicide prevention specialist with 25 years of experience named Rebekah.\n\n" 

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

    chat_session = model.start_chat(history=history)

    # Initial user input
    initial_input = input("Tell me, what brings you here and what bothers you? ")
    response = chat_session.send_message(initial_input)
    print("Model Response:", response.text)

    # Update history
    history.append({"user": initial_input, "model": response.text})

    while True:
        user_input = input("Proceed with the chat: ")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for reaching out. Take care!")
            break

        # Send user input and get model response
        response = chat_session.send_message(user_input)
        print("Model Response:", response.text)

        # Update history
        history.append({"user": user_input, "model": response.text})

# Main entry point
if __name__ == "__main__":
    main_menu()
