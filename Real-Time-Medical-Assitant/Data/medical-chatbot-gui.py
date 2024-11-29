import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from model import MedicalChatbot

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Language Medical Assistant")
        self.root.geometry("600x700")  # Larger window
        
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
        
        # Improved initialization
        self.setup_ui()
        self.setup_styles()

    def setup_styles(self):
        """Create modern, consistent UI styles"""
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10))
        style.configure('TRadiobutton', font=('Arial', 10))

    def setup_ui(self):
        """Comprehensive UI setup with improved layout"""
        # Language Frame
        self.create_language_frame()
        
        # Chat Display
        self.create_chat_display()
        
        # Input Frame
        self.create_input_frame()
        
        # Buttons Frame
        self.create_button_frame()
        
        # Initialize chatbot
        self.chatbot = MedicalChatbot(language='en')
        self.current_symptoms = []
        
        # Initial welcome message
        self.display_welcome_message()

    def create_language_frame(self):
        """Create language selection frame with improved design"""
        self.lang_frame = ttk.Frame(self.root)
        self.lang_frame.pack(pady=10, padx=10, fill='x')
        
        ttk.Label(self.lang_frame, text="Select Language:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Initialize language variable with default
        self.lang_var = tk.StringVar(value='en')
        
        # Define language options
        languages = [
            ("English", 'en'),
            ("हिंदी", 'hi'),
            ("తెలుగు", 'te')
        ]
        
        # Create radio buttons with explicit values
        for text, value in languages:
            rb = ttk.Radiobutton(
                self.lang_frame, 
                text=text, 
                variable=self.lang_var, 
                value=value,  # Explicitly set the value
                command=lambda: self.change_language()  # Use lambda to avoid immediate execution
            )
            rb.pack(side=tk.LEFT, padx=5)

    def process_input(self):
        """Enhanced input processing with error handling"""
        user_input = self.input_entry.get().strip()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter symptoms.")
            return
        
        self.display_user_message(user_input)
        self.input_entry.delete(0, tk.END)
        
        # Enhanced input processing
        try:
            # Extract symptoms
            new_symptoms = self.chatbot.extract_symptoms_from_text(user_input)
            
            if new_symptoms:
                self.current_symptoms.extend(new_symptoms)
                self.make_prediction()
            else:
                # More informative message about symptom extraction
                messagebox.showinfo("Symptom Detection", 
                    "Could not detect specific symptoms. Please describe symptoms more clearly.")
        
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {str(e)}")

    def make_prediction(self):
        """Enhanced prediction with more detailed output"""
        if not self.current_symptoms:
            messagebox.showwarning("No Symptoms", "Please enter some symptoms.")
            return
        
        try:
            # Get predictions
            predictions = self.chatbot.predict_condition(self.current_symptoms)
            
            # Detailed result display
            result = "Possible Conditions:\n\n"
            for condition, confidence in predictions:
                result += f"• {condition} (Confidence: {confidence:.2f}%)\n"
                
                # Add description and precautions
                description = self.chatbot.get_description(condition)
                precautions = self.chatbot.get_precautions(condition)
                
                result += f"  Description: {description}\n"
                result += "  Precautions:\n"
                for i, precaution in enumerate(precautions, 1):
                    result += f"  {i}. {precaution}\n"
                result += "\n"
            
            # Display in chat
            self.display_bot_message(result)
        
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not complete prediction: {str(e)}")

    def change_language(self):
        """Change the interface language"""
        # Get current language selection
        lang = self.lang_var.get()
        
        # Validate language selection
        if lang not in ['en', 'hi', 'te']:
            print(f"Invalid language selection: {lang}")
            self.lang_var.set('en')  # Reset to English if invalid
            lang = 'en'
        
        try:
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
        except Exception as e:
            print(f"Error changing language: {str(e)}")
            messagebox.showerror("Error", "Failed to change language. Reverting to English.")
            self.lang_var.set('en')

    def create_chat_display(self):
        """Create chat display area with scrolling"""
        self.chat_frame = ttk.Frame(self.root)
        self.chat_frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Chat display with improved styling
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            width=50,
            height=20,
            font=('Arial', 10)
        )
        self.chat_display.pack(fill='both', expand=True)
        
        # Configure tag for bot messages
        self.chat_display.tag_configure('bot', foreground='#2E7D32')
        # Configure tag for user messages
        self.chat_display.tag_configure('user', foreground='#1565C0')

    def create_input_frame(self):
        """Create input area with improved design"""
        self.input_frame = ttk.Frame(self.root)
        self.input_frame.pack(pady=5, padx=10, fill='x')
        
        # Add input label
        self.input_label = ttk.Label(
            self.input_frame,
            text=self.ui_text['en']['enter_symptoms'],
            font=('Arial', 10)
        )
        self.input_label.pack(side=tk.TOP, anchor='w', pady=(0, 5))
        
        # Input entry
        self.input_entry = ttk.Entry(
            self.input_frame,
            font=('Arial', 10),
            width=50
        )
        self.input_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        
        # Bind Enter key to process input
        self.input_entry.bind('<Return>', lambda e: self.process_input())

    def create_button_frame(self):
        """Create buttons with improved layout"""
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=10, padx=10, fill='x')
        
        # Send button
        self.send_button = ttk.Button(
            self.button_frame,
            text="Send",
            command=self.process_input
        )
        self.send_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_button = ttk.Button(
            self.button_frame,
            text="Clear",
            command=self.clear_chat
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Start Over button
        self.start_over_button = ttk.Button(
            self.button_frame,
            text="Start Over",
            command=self.start_over
        )
        self.start_over_button.pack(side=tk.LEFT, padx=5)

    def display_welcome_message(self):
        """Display initial welcome message"""
        welcome_text = "Welcome to the Medical Assistant!\nPlease describe your symptoms."
        self.display_bot_message(welcome_text)

    def display_bot_message(self, message):
        """Display bot message with styling"""
        self.chat_display.insert(tk.END, f"Bot: {message}\n\n", 'bot')
        self.chat_display.see(tk.END)

    def display_user_message(self, message):
        """Display user message with styling"""
        self.chat_display.insert(tk.END, f"You: {message}\n\n", 'user')
        self.chat_display.see(tk.END)

    def clear_chat(self):
        """Clear chat display"""
        self.chat_display.delete(1.0, tk.END)
        self.current_symptoms = []
        self.display_welcome_message()

    def start_over(self):
        """Reset the chat and start over"""
        self.clear_chat()
        self.chatbot = MedicalChatbot(language=self.lang_var.get())

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
