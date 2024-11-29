import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class MedicalChatbot:
    def __init__(self, language='en'):
        # Set language first
        self.set_language(language)
        
        # Base datasets (English)
        self.df_training = pd.read_csv('Data/Training.csv')
        
        # Initialize language-specific components
        self.symptoms = list(self.df_training.columns[:-1])
        self.load_language_data()
        
        # Enhanced model training
        self.train_model_enhanced()

    def train_model_enhanced(self):
        """Enhanced model training with cross-validation and feature importance"""
        # Prepare training data
        X = self.df_training.drop('prognosis', axis=1)
        y = self.df_training['prognosis']
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Multiple models for ensemble
        models = [
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
        ]
        
        # Train and evaluate models
        self.trained_models = []
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            print(f"Model Accuracy: {accuracy_score(y_val, y_pred)}")
            self.trained_models.append(model)
        
        # Identify most important features
        feature_importances = np.mean([model.feature_importances_ for model in self.trained_models], axis=0)
        self.important_symptoms = list(zip(X.columns, feature_importances))
        self.important_symptoms.sort(key=lambda x: x[1], reverse=True)

    def predict_condition(self, symptoms):
        """Enhanced condition prediction with multiple model voting"""
        # Create input vector
        input_vector = pd.DataFrame(0, index=[0], columns=self.symptoms)
        for symptom in symptoms:
            if symptom in self.symptoms:
                input_vector[symptom] = 1
        
        # Ensemble prediction
        predictions = []
        for model in self.trained_models:
            prediction_proba = model.predict_proba(input_vector)[0]
            top_3_indices = prediction_proba.argsort()[-3:][::-1]
            
            for idx in top_3_indices:
                condition = self.encoder.inverse_transform([idx])[0]
                confidence = prediction_proba[idx] * 100
                predictions.append((condition, confidence))
        
        # Aggregate and rank predictions
        prediction_dict = {}
        for condition, confidence in predictions:
            if condition not in prediction_dict:
                prediction_dict[condition] = []
            prediction_dict[condition].append(confidence)
        
        # Average confidences and sort
        final_predictions = [
            (condition, np.mean(confidences)) 
            for condition, confidences in prediction_dict.items()
        ]
        final_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return final_predictions[:3]

    def extract_symptoms_from_text(self, text):
        """Enhanced symptom extraction with fuzzy matching"""
        text = text.lower().strip()
        extracted_symptoms = set()
        
        # More comprehensive symptom matching
        for local_term, eng_symptom in self.symptom_mapping.items():
            # Case-insensitive partial matching
            if local_term in text or text in local_term:
                extracted_symptoms.add(eng_symptom)
        
        # Fallback to manual symptom list if no matches
        if not extracted_symptoms:
            manual_check = self.find_matching_symptom(text)
            extracted_symptoms.update(manual_check)
        
        return list(extracted_symptoms)

    def find_matching_symptom(self, text):
        """Enhanced symptom matching with more flexible search"""
        text = text.lower().strip()
        matches = []
        
        # Weighted matching techniques
        for symptom in self.symptoms:
            symptom_normalized = symptom.replace('_', ' ')
            # Different matching strategies
            if (text in symptom_normalized or 
                symptom_normalized in text or 
                text in self.symptom_mapping.get(symptom_normalized, '')):
                matches.append(symptom)
        
        return list(set(matches))  # Remove duplicates
