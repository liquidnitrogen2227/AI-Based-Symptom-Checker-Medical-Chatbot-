import tkinter as tk
from tkinter import ttk, scrolledtext
from model import MedicalChatbot

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Assistant")
        
        # Language-specific UI text
        self.ui_text = {
            'en': {
                'title': 'Medical Assistant',
                'select_lang': 'Select Language:',
                'welcome': 'Hello! I\'m your medical assistant. Please describe your symptoms.',
                'enter_symptoms': 'Enter your symptoms:',
                'send': 'Send',
                'clear': 'Clear',
                'start_over': 'Start Over'
            },
            'hi': {
                'title': 'चिकित्सा सहायक',
                'select_lang': 'भाषा चुनें:',
                'welcome': 'नमस्ते! मैं आपका चिकित्सा सहायक हूं। कृपया अपने लक्षणों का वर्णन करें।',
                'enter_symptoms': 'अपने लक्षण दर्ज करें:',
                'send': 'भेजें',
                'clear': 'साफ़ करें',
                'start_over': 'फिर से शुरू करें'
            },
            'te': {
                'title': 'వైద్య సహాయకుడు',
                'select_lang': 'భాష ఎంచుకోండి:',
                'welcome': 'హలో! నేను మీ వైద్య సహాయకుడిని. దయచేసి మీ లక్షణాలను వివరించండి.',
                'enter_symptoms': 'మీ లక్షణాలను నమోదు చేయండి:',
                'send': 'పంపండి',
                'clear': 'క్లియర్',
                'start_over': 'మళ్ళీ ప్రారంభించండి'
            }
        }
        
        # Create language selection frame
        self.lang_frame = ttk.Frame(root)
        self.lang_frame.pack(pady=10)
        
        self.lang_var = tk.StringVar(value='en')
        
        # Language selection label
        ttk.Label(self.lang_frame, text=self.ui_text['en']['select_lang']).pack(side=tk.LEFT)
        
        # Language selection buttons
        ttk.Radiobutton(self.lang_frame, text="English", variable=self.lang_var, 
                       value='en', command=self.change_language).pack(side=tk.LEFT)
        ttk.Radiobutton(self.lang_frame, text="हिंदी", variable=self.lang_var, 
                       value='hi', command=self.change_language).pack(side=tk.LEFT)
        ttk.Radiobutton(self.lang_frame, text="తెలుగు", variable=self.lang_var, 
                       value='te', command=self.change_language).pack(side=tk.LEFT)
        
        # Initialize chatbot with default language
        self.chatbot = MedicalChatbot(language='en')
        self.current_symptoms = []
        
        # Create chat display
        self.chat_frame = ttk.Frame(root)
        self.chat_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, 
                                                    width=50, height=20)
        self.chat_display.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Create input frame
        self.input_frame = ttk.Frame(root)
        self.input_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.input_label = ttk.Label(self.input_frame, text=self.ui_text['en']['enter_symptoms'])
        self.input_label.pack(side=tk.LEFT)
        
        self.input_entry = ttk.Entry(self.input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT, padx=5)
        self.input_entry.bind('<Return>', lambda e: self.process_input())
        
        # Create button frame
        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(pady=5)
        
        self.send_button = ttk.Button(self.button_frame, text=self.ui_text['en']['send'],
                                    command=self.process_input)
        self.send_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(self.button_frame, text=self.ui_text['en']['clear'],
                                     command=self.clear_symptoms)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.start_over_button = ttk.Button(self.button_frame, text=self.ui_text['en']['start_over'],
                                          command=self.start_over)
        self.start_over_button.pack(side=tk.LEFT, padx=5)
        
        # Display welcome message
        self.display_bot_message(self.ui_text['en']['welcome'])

    def change_language(self):
        """Change the interface language"""
        lang = self.lang_var.get()
        
        # Update chatbot language
        self.chatbot = MedicalChatbot(language=lang)
        
        # Update UI text
        self.root.title(self.ui_text[lang]['title'])
        self.input_label.config(text=self.ui_text[lang]['enter_symptoms'])
        self.send_button.config(text=self.ui_text[lang]['send'])
        self.clear_button.config(text=self.ui_text[lang]['clear'])
        self.start_over_button.config(text=self.ui_text[lang]['start_over'])
        
        # Clear chat and show welcome message in new language
        self.chat_display.delete(1.0, tk.END)
        self.current_symptoms = []
        self.display_bot_message(self.ui_text[lang]['welcome'])

    def process_input(self):
        """Process user input"""
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        
        self.display_user_message(user_input)
        self.input_entry.delete(0, tk.END)
        
        if user_input.lower() in ['yes', 'हाँ', 'అవును']:
            self.start_over()
            return
        
        if user_input.lower() in ['done', 'समाप्त', 'పూర్తి']:
            self.make_prediction()
            return
        
        # Extract symptoms from input
        new_symptoms = self.chatbot.extract_symptoms_from_text(user_input)
        if new_symptoms:
            self.current_symptoms.extend(new_symptoms)
            self.make_prediction()
        else:
            self.display_bot_message(self.get_language_text('no_symptoms_found'))

    def get_language_text(self, key):
        """Get text in current language"""
        texts = {
            'no_symptoms_found': {
                'en': "I couldn't identify any symptoms. Please try describing them differently.",
                'hi': "मैं कोई लक्षण नहीं पहचान पाया। कृपया उन्हें अलग तरीके से बताएं।",
                'te': "నేను ఏ లక్షణాలను గుర్తించలేకపోయాను. దయచేసి వాటిని వేరే విధంగా వివరించండి."
            },
            'low_confidence': {
                'en': "Would you like to add more symptoms for a more accurate diagnosis?",
                'hi': "क्या आप अधिक सटीक निदान के लिए और लक्षण जोड़ना चाहेंगे?",
                'te': "మరింత ఖచ్చితమైన రోగనిర్ధారణ కోసం మరిన్ని లక్షణాలను జోడించాలనుకుంటున్నారా?"
            }
        }
        return texts[key][self.lang_var.get()]

    def display_user_message(self, message):
        """Display user message in chat"""
        self.chat_display.insert(tk.END, f"You: {message}\n\n")
        self.chat_display.see(tk.END)

    def display_bot_message(self, message):
        """Display bot message in chat"""
        self.chat_display.insert(tk.END, f"Bot: {message}\n\n")
        self.chat_display.see(tk.END)

    def clear_symptoms(self):
        """Clear current symptoms"""
        self.current_symptoms = []
        self.display_bot_message(self.ui_text[self.lang_var.get()]['welcome'])

    def start_over(self):
        """Start over the conversation"""
        self.clear_symptoms()
        self.chat_display.delete(1.0, tk.END)
        self.display_bot_message(self.ui_text[self.lang_var.get()]['welcome'])

    def make_prediction(self):
        """Make prediction based on current symptoms"""
        if not self.current_symptoms:
            self.display_bot_message(self.get_language_text('no_symptoms_found'))
            return
        
        predictions = self.chatbot.predict_condition(self.current_symptoms)
        
        # Get the top prediction
        main_condition, main_confidence = predictions[0]
        
        # Format the prediction message
        lang = self.lang_var.get()
        result = "\n" + "=" * 40 + "\n\n"
        
        # Show top 3 predictions with confidence
        prediction_texts = {
            'en': "Top 3 possible conditions:",
            'hi': "शीर्ष 3 संभावित स्थितियां:",
            'te': "మొదటి 3 సంభావ్య పరిస్థితులు:"
        }
        
        result += f"{prediction_texts[lang]}\n"
        for condition, confidence in predictions:
            result += f"• {condition} ({confidence:.1f}%)\n"
        
        result += "\n" + "=" * 40 + "\n"
        
        # Add warning for low confidence
        if main_confidence < 50:
            warning_texts = {
                'en': "⚠️ Warning: Low confidence prediction. Please provide more symptoms for a more accurate diagnosis.",
                'hi': "⚠️ चेतावनी: कम विश्वसनीयता वाली भविष्यवाणी। कृपया अधिक सटीक निदान के लिए अधिक लक्षण प्रदान करें।",
                'te': "⚠️ హెచ్చరిక: తక్కువ విశ్వసనీయత అంచనా. దయచేసి మరింత ఖచ్చితమైన రోగనిర్ధారణ కోసం మరిన్ని లక్షణాలను అందించండి."
            }
            result += f"\n{warning_texts[lang]}\n"
        
        # Get description and precautions
        description = self.chatbot.get_description(main_condition)
        precautions = self.chatbot.get_precautions(main_condition)
        
        # Add detailed information
        detail_texts = {
            'en': f"\nDetailed information for {main_condition}:",
            'hi': f"\n{main_condition} के लिए विस्तृत जानकारी:",
            'te': f"\n{main_condition} కోసం వివరణాత్మక సమాచారం:"
        }
        
        result += f"\n{detail_texts[lang]}\n"
        result += f"\nDescription:\n{description}\n"
        result += "\nPrecautions:\n"
        
        for i, precaution in enumerate(precautions, 1):
            result += f"{i}. {precaution}\n"
        
        self.display_bot_message(result)
        
        # Prompt for more symptoms if confidence is low
        if main_confidence < 30:
            prompt_texts = {
                'en': "\nWould you like to add more symptoms for a more accurate diagnosis? (Type 'yes' to continue or 'done' to finish)",
                'hi': "\nक्या आप अधिक सटीक निदान के लिए और लक्षण जोड़ना चाहेंगे? ('हाँ' जारी रखने के लिए या 'समाप्त' समाप्त करने के लिए टाइप करें)",
                'te': "\nమరింత ఖచ్చితమైన రోగనిర్ధారణ కోసం మరిన్ని లక్షణాలను జోడించాలనుకుంటున్నారా? (కొనసాగించడానికి 'అవును' లేదా ముగించడానికి 'పూర్తి' టైప్ చేయండి)"
            }
            self.display_bot_message(prompt_texts[lang])
        else:
            prompt_texts = {
                'en': "\nWould you like to check for other symptoms? (Type 'yes' to start over)",
                'hi': "\nक्या आप अन्य लक्षणों की जांच करना चाहेंगे? (फिर से शुरू करने के लिए 'हाँ' टाइप करें)",
                'te': "\nఇతర లక్షణాలను తనిఖీ చేయాలనుకుంటున్నారా? (మళ్ళీ ప్రారంభించడానికి 'అవును' టైప్ చేయండి)"
            }
            self.display_bot_message(prompt_texts[lang])

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 