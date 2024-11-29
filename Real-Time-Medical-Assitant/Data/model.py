import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class MedicalChatbot:
    def __init__(self, language='en'):
        self.language = language
        self.label_encoder = LabelEncoder()
        
        # Define language-specific data file paths
        self.language_files = {
            'en': {
                'description': 'Data/symptom_Description.csv',
                'precaution': 'Data/symptom_precaution.csv',
                'severity': 'Data/Symptom_severity.csv'
            },
            'hi': {
                'description': 'Data/symptom_Description_Hindi.csv',
                'precaution': 'Data/symptom_precaution_Hindi.csv',
                'severity': 'Data/Symptom_severity_Hindi.csv'
            },
            'te': {
                'description': 'Data/symptom_Description_Telugu.csv',
                'precaution': 'Data/symptom_precaution_Telugu.csv',
                'severity': 'Data/Symptom_severity_Telugu.csv'
            }
        }
        
        # Disease translations
        self.disease_mappings = {
            'hi': {
                'Fungal infection': 'फंगल संक्रमण',
                'Allergy': 'एलर्जी',
                'GERD': 'गैस्ट्रोइसोफेगल रिफ्लक्स रोग',
                'Chronic cholestasis': 'क्रोनिक कोलेस्टासिस',
                'Drug Reaction': 'दवा प्रतिक्रिया',
                'Peptic ulcer disease': 'पेप्टिक अल्सर रोग',
                'AIDS': 'एड्स',
                'Diabetes': 'मधुमेह',
                'Gastroenteritis': 'गैस्ट्रोएंटेराइटिस',
                'Bronchial Asthma': 'ब्रोंकियल अस्थमा',
                'Hypertension': 'उच्च रक्तचाप',
                'Migraine': 'माइग्रेन',
                'Cervical spondylosis': 'सर्वाइकल स्पॉन्डिलाइटिस',
                'Paralysis (brain hemorrhage)': 'लकवा (मस्तिष्क रक्तस्राव)',
                'Jaundice': 'पीलिया',
                'Malaria': 'मलेरिया',
                'Chicken pox': 'चेचक',
                'Dengue': 'डेंगू',
                'Typhoid': 'टाइफाइड',
                'Hepatitis A': 'हेपेटाइटिस ए',
                'Hepatitis B': 'हेपेटाइटिस बी',
                'Hepatitis C': 'हेपेटाइटिस सी',
                'Hepatitis D': 'हेपेटाइटिस डी',
                'Hepatitis E': 'हेपेटाइटिस ई',
                'Alcoholic hepatitis': 'अल्कोहलिक हेपेटाइटिस',
                'Tuberculosis': 'क्षय रोग',
                'Common Cold': 'सामान्य सर्दी',
                'Pneumonia': 'निमोनिया',
                'Dimorphic hemorrhoids(piles)': 'बवासीर',
                'Heart attack': 'दिल का दौरा',
                'Varicose veins': 'वैरिकोस नसें',
                'Hypothyroidism': 'हाइपोथायरायडिज्म',
                'Hyperthyroidism': 'हाइपरथायरायडिज्म',
                'Hypoglycemia': 'हाइपोग्लाइसीमिया',
                'Osteoarthritis': 'ऑस्टियोआर्थराइटिस',
                'Arthritis': 'गठिया',
                'Vertigo': 'वर्टिगो',
                'Acne': 'मुंहासे',
                'Urinary tract infection': 'मूत्र मार्ग संक्रमण',
                'Psoriasis': 'सोरायसिस',
                'Impetigo': 'इम्पेटिगो'
            },
            'te': {
                'Fungal infection': 'శిలీంధ్ర సంక్రమణ',
                'Allergy': 'అలెర్జీ',
                'GERD': 'గ్యాస్ట్రోఎసోఫేగల్ రిఫ్లక్స్ వ్యాధి',
                'Chronic cholestasis': 'దీర్ఘకాలిక కోలెస్టాసిస్',
                'Drug Reaction': 'మందుల ప్రతిచర్య',
                'Peptic ulcer disease': 'పెప్టిక్ అల్సర్ వ్యాధి',
                'AIDS': 'ఎయిడ్స్',
                'Diabetes': 'మధుమేహం',
                'Gastroenteritis': 'గ్యాస్ట్రోఎంటరైటిస్',
                'Bronchial Asthma': 'ఊపిరితిత్తుల ఆస్తమా',
                'Hypertension': 'అధిక రక్తపోటు',
                'Migraine': 'అర్ధశిరోవేదన',
                'Cervical spondylosis': 'మెడ స్పాండిలైటిస్',
                'Paralysis (brain hemorrhage)': 'పక్షవాతం (మెదడు రక్తస్రావం)',
                'Jaundice': 'కామెర్ల',
                'Malaria': 'మలేరియా',
                'Chicken pox': 'చికెన్ పాక్స్',
                'Dengue': 'డెంగ్యూ',
                'Typhoid': 'టైఫాయిడ్',
                'Hepatitis A': 'హెపటైటిస్ ఎ',
                'Hepatitis B': 'హెపటైటిస్ బి',
                'Hepatitis C': 'హెపటైటిస్ సి',
                'Hepatitis D': 'హెపటైటిస్ డి',
                'Hepatitis E': 'హెపటైటిస్ ఇ',
                'Alcoholic hepatitis': 'మద్యపాన హెపటైటిస్',
                'Tuberculosis': 'క్షయ',
                'Common Cold': 'జలుబు',
                'Pneumonia': 'న్యుమోనియా',
                'Dimorphic hemorrhoids(piles)': 'మూలవ్యాధి',
                'Heart attack': 'గుండెపోటు',
                'Varicose veins': 'వారికోస్ సిరలు',
                'Hypothyroidism': 'హైపోథైరాయిడిజం',
                'Hyperthyroidism': 'హైపర్థైరాయిడిజం',
                'Hypoglycemia': 'హైపోగ్లైసీమియా',
                'Osteoarthritis': 'ఆస్టియోఆర్థ్రైటిస్',
                'Arthritis': 'కీళ్ల వాతం',
                'Vertigo': 'వెర్టిగో',
                'Acne': 'మొటిమలు',
                'Urinary tract infection': 'మూత్ర మార్గ సంక్రమణ',
                'Psoriasis': 'సోరియాసిస్',
                'Impetigo': 'ఇంపెటిగో'
            }
        }
        
        # Load training data
        self.df_training = pd.read_csv('Data/Training.csv')
        
        # Initialize symptoms from training data
        self.symptoms = list(self.df_training.columns[:-1])  # All columns except 'prognosis'
        
        # Initialize language-specific symptom mappings
        self.symptom_mappings = {
            'en': {symptom.lower().replace('_', ' '): symptom for symptom in self.symptoms},
            'hi': {
                        'खुजली': 'itching',
                        'त्वचा पर चकत्ते': 'skin_rash',
                        'गांठदार त्वचा पर दाने': 'nodal_skin_eruptions',
                        'लगातार छींक आना': 'continuous_sneezing',
                        'कंपकंपी': 'shivering',
                        'ठंड लगना': 'chills',
                        'जोड़ों में दर्द': 'joint_pain',
                        'पेट में दर्द': 'stomach_pain',
                        'एसिडिटी': 'acidity',
                        'जीभ पर छाले': 'ulcers_on_tongue',
                        'मांसपेशियों में कमजोरी': 'muscle_wasting',
                        'उल्टी': 'vomiting',
                        'पेशाब में जलन': 'burning_micturition',
                        'दाग-धब्बे पेशाब': 'spotting_urination',
                        'थकान': 'fatigue',
                        'वजन बढ़ना': 'weight_gain',
                        'चिंता': 'anxiety',
                        'ठंडे हाथ और पैर': 'cold_hands_and_feets',
                        'मूड में बदलाव': 'mood_swings',
                        'वजन घटना': 'weight_loss',
                        'बेचैनी': 'restlessness',
                        'सुस्ती': 'lethargy',
                        'गले में पैच': 'patches_in_throat',
                        'अनियमित शुगर स्तर': 'irregular_sugar_level',
                        'खांसी': 'cough',
                        'तेज बुखार': 'high_fever',
                        'धँसी हुई आँखें': 'sunken_eyes',
                        'सांस फूलना': 'breathlessness',
                        'पसीना': 'sweating',
                        'निर्जलीकरण': 'dehydration',
                        'अपच': 'indigestion',
                        'सिरदर्द': 'headache',
                        'त्वचा का पीला पड़ना': 'yellowish_skin',
                        'गहरे रंग का मूत्र': 'dark_urine',
                        'मतली': 'nausea',
                        'भूख न लगना': 'loss_of_appetite',
                        'आँखों के पीछे दर्द': 'pain_behind_the_eyes',
                        'पीठ दर्द': 'back_pain',
                        'कब्ज': 'constipation',
                        'पेट दर्द': 'abdominal_pain',
                        'दस्त': 'diarrhoea',
                        'हल्का बुखार': 'mild_fever',
                        'पेशाब पीला': 'yellow_urine',
                        'आँखों का पीला होना': 'yellowing_of_eyes',
                        'तीव्र यकृत विफलता': 'acute_liver_failure',
                        'तरल पदार्थ की अधिकता': 'fluid_overload',
                        'पेट का फूलना': 'swelling_of_stomach',
                        'लिम्फ नोड्स में सूजन': 'swelled_lymph_nodes',
                        'अस्वस्थता': 'malaise',
                        'धुंधली और विकृत दृष्टि': 'blurred_and_distorted_vision',
                        'कफ': 'phlegm',
                        'गले में जलन': 'throat_irritation',
                        'आंखों की लाली': 'redness_of_eyes',
                        'साइनस का दबाव': 'sinus_pressure',
                        'नाक बहना': 'runny_nose',
                        'कब्ज': 'congestion',
                        'सीने में दर्द': 'chest_pain',
                        'अंगों में कमजोरी': 'weakness_in_limbs',
                        'दिल की तेज़ गति': 'fast_heart_rate',
                        'मल त्याग के दौरान दर्द': 'pain_during_bowel_movements',
                        'गुदा क्षेत्र में दर्द': 'pain_in_anal_region',
                        'मल में खून': 'bloody_stool',
                        'गुदा में जलन': 'irritation_in_anus',
                        'गर्दन में दर्द': 'neck_pain',
                        'चक्कर आना': 'dizziness',
                        'ऐंठन': 'cramps',
                        'चोट': 'bruising',
                        'मोटापा': 'obesity',
                        'सूजे हुए पैर': 'swollen_legs',
                        'सूजी हुई रक्त वाहिकाएं': 'swollen_blood_vessels',
                        'फूला हुआ चेहरा और आंखें': 'puffy_face_and_eyes',
                        'बढ़ा हुआ थायरॉइड': 'enlarged_thyroid',
                        'भंगुर नाखून': 'brittle_nails',
                        'सूजे हुए हाथ-पैर': 'swollen_extremeties',
                        'अत्यधिक भूख': 'excessive_hunger',
                        'विवाहेतर संपर्क': 'extra_marital_contacts',
                        'होठों का सूखना और झुनझुनी': 'drying_and_tingling_lips',
                        'अस्पष्ट बोलना': 'slurred_speech',
                        'घुटनों का दर्द': 'knee_pain',
                        'कूल्हे के जोड़ों का दर्द': 'hip_joint_pain',
                        'मांसपेशियों में कमजोरी': 'muscle_weakness',
                        'गर्दन में अकड़न': 'stiff_neck',
                        'जोड़ों में सूजन': 'swelling_joints',
                        'गति में कठोरता': 'movement_stiffness',
                        'घूमना': 'spinning_movements',
                        'संतुलन खोना': 'loss_of_balance',
                        'अस्थिरता': 'unsteadiness',
                        'शरीर के एक तरफ की कमजोरी': 'weakness_of_one_body_side',
                        'गंध की हानि': 'loss_of_smell',
                        'मूत्राशय में असुविधा': 'bladder_discomfort',
                        'मूत्र की दुर्गंध': 'foul_smell_of_urine',
                        'लगातार पेशाब का अहसास': 'continuous_feel_of_urine',
                        'गैसों का निकलना': 'passage_of_gases',
                        'आंतरिक खुजली': 'internal_itching',
                        'विषै��ा रूप (टाइफोस)': 'toxic_look_(typhos)',
                        'अवसाद': 'depression',
                        'चिड़चिड़ापन': 'irritability',
                        'मांसपेशियों में दर्द': 'muscle_pain',
                        'संवेदी संवेदना में बदलाव': 'altered_sensorium',
                        'शरीर पर लाल धब्बे': 'red_spots_over_body',
                        'पेट में दर्द': 'belly_pain',
                        'असामान्य मासिक धर्म': 'abnormal_menstruation',
                        'त्वचा पर धब्बे': 'dischromic_patches',
                        'आँखों से पानी आना': 'watering_from_eyes',
                        'भूख बढ़ना': 'increased_appetite',
                        'बहुमूत्रता': 'polyuria',
                        'पारिवारिक इतिहास': 'family_history',
                        'श्लेष्मा बलगम': 'mucoid_sputum',
                        'जंग लगा बलगम': 'rusty_sputum',
                        'एकाग्रता की कमी': 'lack_of_concentration',
                        'दृष्टि विकार': 'visual_disturbances',
                        'रक्त चढ़ाना': 'receiving_blood_transfusion',
                        'अस्वच्छ इंजेक्शन': 'receiving_unsterile_injections',
                        'कोमा': 'coma',
                        'पेट से रक्तस्राव': 'stomach_bleeding',
                        'पेट का फूलना': 'distention_of_abdomen',
                        'शराब पीने का इतिहास': 'history_of_alcohol_consumption',
                        'तरल पदार्थ अधिकता': 'fluid_overload',
                        'बलगम में खून': 'blood_in_sputum',
                        'पिंडली की उभरी नसें': 'prominent_veins_on_calf',
                        'धड़कन': 'palpitations',
                        'चलने में दर्द': 'painful_walking',
                        'मवाद भरे फुंसी': 'pus_filled_pimples',
                        'ब्लैकहेड्स': 'blackheads',
                        'त्वचा खरोंच': 'scurring',
                        'त्वचा का छिलना': 'skin_peeling',
                        'चांदी जैसी धूल': 'silver_like_dusting',
                        'नाखूनों में छोटे गड्ढे': 'small_dents_in_nails',
                        'सूजे हुए नाखून': 'inflammatory_nails',
                        'छाला': 'blister',
                        'नाक के आसपास लाल घाव': 'red_sore_around_nose',
                        'पीली पपड़ी': 'yellow_crust_ooze'
                    }, # Add Hindi mappings here
            'te': {
                    'దురద': 'itching',
                    'చర్మం_దద్దుర్లు': 'skin_rash',
                    'నోడల్_చర్మం_విస్ఫోటనాలు': 'nodal_skin_eruptions',
                    'నిరంతర_తుమ్ములు': 'continuous_sneezing',
                    'వణుకు': 'shivering',
                    'చలి': 'chills',
                    'కీళ్ల_నొప్పి': 'joint_pain',
                    'కడుపు_నొప్పి': 'stomach_pain',
                    'ఆమ్లత్వం': 'acidity',
                    'నాలుకపై_పూత': 'ulcers_on_tongue',
                    'కండరాలు_వ్యర్థం': 'muscle_wasting',
                    'వాంతులు': 'vomiting',
                    'మంట_మూత్రవిసర్జన': 'burning_micturition',
                    'మచ్చలు మూత్రవిసర్జన': 'spotting_urination',
                    'అలసట': 'fatigue',
                    'బరువు_పెరుగడం': 'weight_gain',
                    'ఆందోళన': 'anxiety',
                    'చలి_చేతులు_కాళ్లు': 'cold_hands_and_feets',
                    'మూడ్_స్వింగ్స్': 'mood_swings',
                    'బరువు_తగ్గడం': 'weight_loss',
                    'అశాంతి': 'restlessness',
                    'బద్ధకం': 'lethargy',
                    'గొంతులో_పాచెస్': 'patches_in_throat',
                    'సక్రమంగా_షుగర్_లెవల్': 'irregular_sugar_level',
                    'దగ్గు': 'cough',
                    'అధిక_జ్వరము': 'high_fever',
                    'ముంచిన_కళ్ళు': 'sunken_eyes',
                    'ఊపిరి_ఆడకపోవడం': 'breathlessness',
                    'చెమటలు': 'sweating',
                    'నిర్జలీకరణం': 'dehydration',
                    'అజీర్ణం': 'indigestion',
                    'తలనొప్పి': 'headache',
                    'పసుపురంగు_చర్మం': 'yellowish_skin',
                    'ముదురు_మూత్రం': 'dark_urine',
                    'వికారం': 'nausea',
                    'ఆకలి_లేకపోవడం': 'loss_of_appetite',
                    'కళ్ల_వెనుక_నొప్పి': 'pain_behind_the_eyes',
                    'వెన్నునొప్పి': 'back_pain',
                    'మలబద్ధకం': 'constipation',
                    'పొత్తికడుపు_నొప్పి': 'abdominal_pain',
                    'అతిసారం': 'diarrhoea',
                    'తేలికపాటి_జ్వరం': 'mild_fever',
                    'పసుపు_మూత్రం': 'yellow_urine',
                    'కళ్లు_పసుపు_రంగు': 'yellowing_of_eyes',
                    'తీవ్ర_కాలేయ_వైఫల్యం': 'acute_liver_failure',
                    'ద్రవ_అధిక_భారం': 'fluid_overload',
                    'పొత్తికడుపు_వాపు': 'swelling_of_stomach',
                    'వాపు_లింఫ్_గ్రంధులు': 'swelled_lymph_nodes',
                    'అస్వస్థత': 'malaise',
                    'మసకబారిన_దృష్టి': 'blurred_and_distorted_vision',
                    'కఫం': 'phlegm',
                    'గొంతు_మంట': 'throat_irritation',
                    'కళ్ళు_ఎరుపు': 'redness_of_eyes',
                    'సైనస్_ఒత్తిడి': 'sinus_pressure',
                    'ముక్కు_కారడం': 'runny_nose',
                    'కంజెషన్': 'congestion',
                    'ఛాతీ_నొప్పి': 'chest_pain',
                    'అవయవాలలో_బలహీనత': 'weakness_in_limbs',
                    'వేగవంతమైన_గుండె_కొట్టుకోవడం': 'fast_heart_rate',
                    'మలవిసర్జన_సమయంలో_నొప్పి': 'pain_during_bowel_movements',
                    'పాయువు_ప్రాంతంలో_నొప్పి': 'pain_in_anal_region',
                    'రక్తపు_మలం': 'bloody_stool',
                    'పాయువులో_దురద': 'irritation_in_anus',
                    'మెడ_నొప్పి': 'neck_pain',
                    'తలతిరగడం': 'dizziness',
                    'నొప్పులు': 'cramps',
                    'గాయాలు': 'bruising',
                    'బొజ్జ': 'obesity',
                    'కాళ్ళు_వాపు': 'swollen_legs',
                    'రక్తనాళాలు_వాపు': 'swollen_blood_vessels',
                    'ముఖం_కళ్ళు_వాపు': 'puffy_face_and_eyes',
                    'థైరాయిడ్_పెరుగుదల': 'enlarged_thyroid',
                    'గోళ్ళు_సులువుగా_విరిగిపోవడం': 'brittle_nails',
                    'చేతులు_కాళ్ళు_వాపు': 'swollen_extremeties',
                    'అధిక_ఆకలి': 'excessive_hunger',
                    'వివాహేతర_సంబంధాలు': 'extra_marital_contacts',
                    'పెదవులు_ఎండిపోవడం': 'drying_and_tingling_lips',
                    'మాటలు_తడబడటం': 'slurred_speech',
                    'మోకాలి_నొప్పి': 'knee_pain',
                    'తుంటి_నొప్పి': 'hip_joint_pain',
                    'కండరాల_బలహీనత': 'muscle_weakness',
                    'మెడ_బిగుసుకుపోవడం': 'stiff_neck',
                    'కీళ్ళు_వాపు': 'swelling_joints',
                    'కదలికలో_బిగుసుకుపోవడం': 'movement_stiffness',
                    'తిరుగుతున్నట్లు_అనిపించడం': 'spinning_movements',
                    'సమతుల్యత_కోల్పోవడం': 'loss_of_balance',
                    'అస్థిరత': 'unsteadiness',
                    'ఒక_వైపు_శరీరం_బలహీనత': 'weakness_of_one_body_side',
                    'వాసన_తెలియకపోవడం': 'loss_of_smell',
                    'మూత్రాశయ_అసౌకర్యం': 'bladder_discomfort',
                    'మూత్రం_దుర్వాసన': 'foul_smell_of_urine',
                    'నిరంతరం_మూత్రం_వస్తున్నట్లు_అనిపించడం': 'continuous_feel_of_urine',
                    'వాయువులు_వెళ్ళడం': 'passage_of_gases',
                    'లోపలి_దురద': 'internal_itching',
                    'టైఫస్_లక్షణాలు': 'toxic_look_(typhos)',
                    'నిరాశ': 'depression',
                    'చిరాకు': 'irritability',
                    'కండరాల_నొప్పి': 'muscle_pain',
                    'మార్పు_చెందిన_స్పృహ': 'altered_sensorium',
                    'శరీరంపై_ఎరుపు_మచ్చలు': 'red_spots_over_body',
                    'కడుపు_నొప్పి': 'belly_pain',
                    'అసాధారణ_బహిష్టు': 'abnormal_menstruation',
                    'చర్మంపై_మచ్చలు': 'dischromic_patches',
                    'కళ్ళ_నుండి_నీరు_కారడం': 'watering_from_eyes',
                    'ఆకలి_పెరగడం': 'increased_appetite',
                    'అధిక_మూత్రం': 'polyuria',
                    'కుటుంబ_చరిత్ర': 'family_history',
                    'శ్లేష్మం_కఫం': 'mucoid_sputum',
                    'తుప్పు_రంగు_కఫం': 'rusty_sputum',
                    'ఏకాగ్రత_లేకపోవడం': 'lack_of_concentration',
                    'దృష్టి_సమస్యలు': 'visual_disturbances',
                    'రక్తమార్పిడి_చేయించుకోవడం': 'receiving_blood_transfusion',
                    'అపరిశుభ్ర_సూదులు_వాడటం': 'receiving_unsterile_injections',
                    'కోమా': 'coma',
                    'కడుపులో_రక్తస్రావం': 'stomach_bleeding',
                    'పొత్తికడుపు_ఉబ్బరం': 'distention_of_abdomen',
                    'మద్యం_సేవించే_చరిత్ర': 'history_of_alcohol_consumption',
                    'ద్రవ_అధిక_భారం': 'fluid_overload',
                    'కఫంలో_రక్తం': 'blood_in_sputum',
                    'పిక్కల_సిరలు_వాపు': 'prominent_veins_on_calf',
                    'గుండె_దడ': 'palpitations',
                    'నడవడంలో_నొప్పి': 'painful_walking',
                    'చీము_మొటిమలు': 'pus_filled_pimples',
                    'నల్లమచ్చలు': 'blackheads',
                    'చర్మం_గీరడం': 'scurring',
                    'చర్మం_ఒలవడం': 'skin_peeling',
                    'వెండి_రంగు_పొడి': 'silver_like_dusting',
                    'గోళ్ళలో_చిన్న_గుంటలు': 'small_dents_in_nails',
                    'వాపుతో_కూడిన_గోళ్ళు': 'inflammatory_nails',
                    'బొబ్బ': 'blister',
                    'ముక్కు_చుట్టూ_ఎర్రని_పుండు': 'red_sore_around_nose',
                    'పసుపు_రంగు_కారుతున్న_గాయం': 'yellow_crust_ooze'
                }
        }
        
        # Set current language mapping
        self.symptom_mapping = self.symptom_mappings[self.language]
        
        # Train the model
        self.train_model()
        
        # Load language data
        self.load_language_data()

    def change_language(self, new_language):
        """Change the chatbot's language"""
        self.language = new_language
        self.symptom_mapping = self.symptom_mappings[self.language]
        self.load_language_data()

    def load_language_data(self):
        """Load datasets for current language"""
        files = self.language_files[self.language]
        
        try:
            # Load language-specific datasets with error handling
            self.df_description = pd.read_csv(files['description'], 
                                            encoding='utf-8',
                                            on_bad_lines='skip')  # Skip problematic lines
            
            self.df_precaution = pd.read_csv(files['precaution'], 
                                            encoding='utf-8',
                                            on_bad_lines='skip')
            
            self.df_severity = pd.read_csv(files['severity'], 
                                         encoding='utf-8',
                                         on_bad_lines='skip')
            
            # Create language-specific symptom mappings
            self.create_symptom_mappings()
            
        except Exception as e:
            print(f"Error loading language data: {str(e)}")
            # Fallback to English if there's an error
            if self.language != 'en':
                print("Falling back to English")
                self.language = 'en'
                self.load_language_data()

    def create_symptom_mappings(self):
        """Create symptom mappings for current language"""
        # Base English mappings
        self.symptom_mapping = {
            symptom.lower().replace('_', ' '): symptom 
            for symptom in self.symptoms
        }
        
        # Language-specific mappings
        if self.language == 'hi':
            self.symptom_mapping.update({
                'खुजली': 'itching',
                'त्वचा पर चकत्ते': 'skin_rash',
                'गांठदार त्वचा पर दाने': 'nodal_skin_eruptions',
                'लगातार छींक आना': 'continuous_sneezing',
                'कंपकंपी': 'shivering',
                'ठंड लगना': 'chills',
                'जोड़ों में दर्द': 'joint_pain',
                'पेट में दर्द': 'stomach_pain',
                'एसिडिटी': 'acidity',
                'जीभ पर छाले': 'ulcers_on_tongue',
                'मांसपेशियों में कमजोरी': 'muscle_wasting',
                'उल्टी': 'vomiting',
                'पेशाब में जलन': 'burning_micturition',
                'दाग-धब्बे पेशाब': 'spotting_urination',
                'थकान': 'fatigue',
                'वजन बढ़ना': 'weight_gain',
                'चिंता': 'anxiety',
                'ठंडे हाथ और पैर': 'cold_hands_and_feets',
                'मूड में बदलाव': 'mood_swings',
                'वजन घटना': 'weight_loss',
                'बेचैनी': 'restlessness',
                'सुस्ती': 'lethargy',
                'गले में पैच': 'patches_in_throat',
                'अनियमित शुगर स्तर': 'irregular_sugar_level',
                'खांसी': 'cough',
                'तेज बुखार': 'high_fever',
                'धँसी हुई आँखें': 'sunken_eyes',
                'सांस फूलना': 'breathlessness',
                'पसीना': 'sweating',
                'निर्जलीकरण': 'dehydration',
                'अपच': 'indigestion',
                'सिरदर्द': 'headache',
                'त्वचा का पीला पड़ना': 'yellowish_skin',
                'गहरे रंग का मूत्र': 'dark_urine',
                'मतली': 'nausea',
                'भूख न लगना': 'loss_of_appetite',
                'आँखों के पीछे दर्द': 'pain_behind_the_eyes',
                'पीठ दर्द': 'back_pain',
                'कब्ज': 'constipation',
                'पेट दर्द': 'abdominal_pain',
                'दस्त': 'diarrhoea',
                'हल्का बुखार': 'mild_fever',
                'पेशाब पीला': 'yellow_urine',
                'आँखों का पीला होना': 'yellowing_of_eyes',
                'तीव्र यकृत विफलता': 'acute_liver_failure',
                'तरल पदार्थ की अधिकता': 'fluid_overload',
                'पेट का फूलना': 'swelling_of_stomach',
                'लिम्फ नोड्स में सूजन': 'swelled_lymph_nodes',
                'अस्वस्थता': 'malaise',
                'धुंधली और विकृत दृष्टि': 'blurred_and_distorted_vision',
                'कफ': 'phlegm',
                'गले में जलन': 'throat_irritation',
                'आंखों की लाली': 'redness_of_eyes',
                'साइनस का दबाव': 'sinus_pressure',
                'नाक बहना': 'runny_nose',
                'कब्ज': 'congestion',
                'सीने में दर्द': 'chest_pain',
                'अंगों में कमजोरी': 'weakness_in_limbs',
                'दिल की तेज़ गति': 'fast_heart_rate',
                'मल त्याग के दौरान दर्द': 'pain_during_bowel_movements',
                'गुदा क्षेत्र में दर्द': 'pain_in_anal_region',
                'मल में खून': 'bloody_stool',
                'गुदा में जलन': 'irritation_in_anus',
                'गर्दन में दर्द': 'neck_pain',
                'चक्कर आना': 'dizziness',
                'ऐंठन': 'cramps',
                'चोट': 'bruising',
                'मोटापा': 'obesity',
                'सूजे हुए पैर': 'swollen_legs',
                'सूजी हुई रक्त वाहिकाएं': 'swollen_blood_vessels',
                'फूला हुआ चेहरा और आंखें': 'puffy_face_and_eyes',
                'बढ़ा हुआ थायरॉइड': 'enlarged_thyroid',
                'भंगुर नाखून': 'brittle_nails',
                'सूजे हुए हाथ-पैर': 'swollen_extremeties',
                'अत्यधिक भूख': 'excessive_hunger',
                'विवाहेतर संपर्क': 'extra_marital_contacts',
                'होठों का सूखना और झुनझुनी': 'drying_and_tingling_lips',
                'अस्पष्ट बोलना': 'slurred_speech',
                'घुटनों का दर्द': 'knee_pain',
                'कूल्हे के जोड़ों का दर्द': 'hip_joint_pain',
                'मांसपेशियों में कमजोरी': 'muscle_weakness',
                'गर्दन में अकड़न': 'stiff_neck',
                'जोड़ों में सूजन': 'swelling_joints',
                'गति में कठोरता': 'movement_stiffness',
                'घूमना': 'spinning_movements',
                'संतुलन खोना': 'loss_of_balance',
                'अस्थिरता': 'unsteadiness',
                'शरीर के एक तरफ की कमजोरी': 'weakness_of_one_body_side',
                'गंध की हानि': 'loss_of_smell',
                'मूत्राशय में असुविधा': 'bladder_discomfort',
                'मूत्र की दुर्गंध': 'foul_smell_of_urine',
                'लगातार पेशाब का अहसास': 'continuous_feel_of_urine',
                'गैसों का निकलना': 'passage_of_gases',
                'आंतरिक खुजली': 'internal_itching',
                'विषैला रूप (टाइफोस)': 'toxic_look_(typhos)',
                'अवसाद': 'depression',
                'चिड़चिड़ापन': 'irritability',
                'मांसपेशियों में दर्द': 'muscle_pain',
                'संवेदी संवेदना में बदलाव': 'altered_sensorium',
                'शरीर पर लाल धब्बे': 'red_spots_over_body',
                'पेट में दर्द': 'belly_pain',
                'असामान्य मासिक धर्म': 'abnormal_menstruation',
                'त्वचा पर धब्बे': 'dischromic_patches',
                'आँखों से पानी आना': 'watering_from_eyes',
                'भूख बढ़ना': 'increased_appetite',
                'बहुमूत्रता': 'polyuria',
                'पारिवारिक इतिहास': 'family_history',
                'श्लेष्मा बलगम': 'mucoid_sputum',
                'जंग लगा बलगम': 'rusty_sputum',
                'एकाग्रता की कमी': 'lack_of_concentration',
                'दृष्टि विकार': 'visual_disturbances',
                'रक्त चढ़ाना': 'receiving_blood_transfusion',
                'अस्वच्छ इंजेक्शन': 'receiving_unsterile_injections',
                'कोमा': 'coma',
                'पेट से रक्तस्राव': 'stomach_bleeding',
                'पेट का फूलना': 'distention_of_abdomen',
                'शराब पीने का इतिहास': 'history_of_alcohol_consumption',
                'तरल पदार्थ अधिकता': 'fluid_overload',
                'बलगम में खून': 'blood_in_sputum',
                'पिंडली की उभरी नसें': 'prominent_veins_on_calf',
                'धड़कन': 'palpitations',
                'चलने में दर्द': 'painful_walking',
                'मवाद भरे फुंसी': 'pus_filled_pimples',
                'ब्लैकहेड्स': 'blackheads',
                'त्वचा खरोंच': 'scurring',
                'त्वचा का छिलना': 'skin_peeling',
                'चांदी जैसी धूल': 'silver_like_dusting',
                'नाखूनों में छोटे गड्ढे': 'small_dents_in_nails',
                'सूजे हुए नाखून': 'inflammatory_nails',
                'छाला': 'blister',
                'नाक के आसपास लाल घाव': 'red_sore_around_nose',
                'पीली पपड़ी': 'yellow_crust_ooze'
            })
        
        elif self.language == 'te':
            self.symptom_mapping.update({
                    'దురద': 'itching',
                    'చర్మం_దద్దుర్లు': 'skin_rash',
                    'నోడల్_చర్మం_విస్ఫోటనాలు': 'nodal_skin_eruptions',
                    'నిరంతర_తుమ్ములు': 'continuous_sneezing',
                    'వణుకు': 'shivering',
                    'చలి': 'chills',
                    'కీళ్ల_నొప్పి': 'joint_pain',
                    'కడుపు_నొప్పి': 'stomach_pain',
                    'ఆమ్లత్వం': 'acidity',
                    'నాలుకపై_పూత': 'ulcers_on_tongue',
                    'కండరాలు_వ్యర్థం': 'muscle_wasting',
                    'వాంతులు': 'vomiting',
                    'మంట_మూత్రవిసర్జన': 'burning_micturition',
                    'మచ్చలు మూత్రవిసర్జన': 'spotting_urination',
                    'అలసట': 'fatigue',
                    'బరువు_పెరుగడం': 'weight_gain',
                    'ఆందోళన': 'anxiety',
                    'చలి_చేతులు_కాళ్లు': 'cold_hands_and_feets',
                    'మూడ్_స్వింగ్స్': 'mood_swings',
                    'బరువు_తగ్గడం': 'weight_loss',
                    'అశాంతి': 'restlessness',
                    'బద్ధకం': 'lethargy',
                    'గొంతులో_పాచెస్': 'patches_in_throat',
                    'సక్రమంగా_షుగర్_లెవల్': 'irregular_sugar_level',
                    'దగ్గు': 'cough',
                    'అధిక_జ్వరము': 'high_fever',
                    'ముంచిన_కళ్ళు': 'sunken_eyes',
                    'ఊపిరి_ఆడకపోవడం': 'breathlessness',
                    'చెమటలు': 'sweating',
                    'నిర్జలీకరణం': 'dehydration',
                    'అజీర్ణం': 'indigestion',
                    'తలనొప్పి': 'headache',
                    'పసుపురంగు_చర్మం': 'yellowish_skin',
                    'ముదురు_మూత్రం': 'dark_urine',
                    'వికారం': 'nausea',
                    'ఆకలి_లేకపోవడం': 'loss_of_appetite',
                    'కళ్ల_వెనుక_నొప్పి': 'pain_behind_the_eyes',
                    'వెన్నునొప్పి': 'back_pain',
                    'మలబద్ధకం': 'constipation',
                    'పొత్తికడుపు_నొప్పి': 'abdominal_pain',
                    'అతిసారం': 'diarrhoea',
                    'తేలికపాటి_జ్వరం': 'mild_fever',
                    'పసుపు_మూత్రం': 'yellow_urine',
                    'కళ్లు_పసుపు_రంగు': 'yellowing_of_eyes',
                    'తీవ్ర_కాలేయ_వైఫల్యం': 'acute_liver_failure',
                    'ద్రవ_అధిక_భారం': 'fluid_overload',
                    'పొత్తికడుపు_వాపు': 'swelling_of_stomach',
                    'వాపు_లింఫ్_గ్రంధులు': 'swelled_lymph_nodes',
                    'అస్వస్థత': 'malaise',
                    'మసకబారిన_దృష్టి': 'blurred_and_distorted_vision',
                    'కఫం': 'phlegm',
                    'గొంతు_మంట': 'throat_irritation',
                    'కళ్ళు_ఎరుపు': 'redness_of_eyes',
                    'సైనస్_ఒత్తిడి': 'sinus_pressure',
                    'ముక్కు_కారడం': 'runny_nose',
                    'కంజెషన్': 'congestion',
                    'ఛాతీ_నొప్పి': 'chest_pain',
                    'అవయవా���లో_బలహీనత': 'weakness_in_limbs',
                    'వేగవంతమైన_గుండె_కొట్టుకోవడం': 'fast_heart_rate',
                    'మలవిసర్జన_సమయంలో_నొప్పి': 'pain_during_bowel_movements',
                    'పాయువు_ప్రాంతంలో_నొప్పి': 'pain_in_anal_region',
                    'రక్తపు_మలం': 'bloody_stool',
                    'పాయువులో_దురద': 'irritation_in_anus',
                    'మెడ_నొప్పి': 'neck_pain',
                    'తలతిరగడం': 'dizziness',
                    'నొప్పులు': 'cramps',
                    'గాయాలు': 'bruising',
                    'బొజ్జ': 'obesity',
                    'కాళ్ళు_వాపు': 'swollen_legs',
                    'రక్తనాళాలు_వాపు': 'swollen_blood_vessels',
                    'ముఖం_కళ్ళు_వాపు': 'puffy_face_and_eyes',
                    'థైరాయిడ్_పెరుగుదల': 'enlarged_thyroid',
                    'గోళ్ళు_సులువుగా_విరిగిపోవడం': 'brittle_nails',
                    'చేతులు_కాళ్ళు_వాపు': 'swollen_extremeties',
                    'అధిక_ఆకలి': 'excessive_hunger',
                    'వివాహేతర_సంబంధాలు': 'extra_marital_contacts',
                    'పెదవులు_ఎండిపోవడం': 'drying_and_tingling_lips',
                    'మాటలు_తడబడటం': 'slurred_speech',
                    'మోకాలి_నొప్పి': 'knee_pain',
                    'తుంటి_నొప్పి': 'hip_joint_pain',
                    'కండరాల_బలహీనత': 'muscle_weakness',
                    'మెడ_బిగుసుకుపోవడం': 'stiff_neck',
                    'కీళ్ళు_వాపు': 'swelling_joints',
                    'కదలికలో_బిగుసుకుపోవడం': 'movement_stiffness',
                    'తిరుగుతున్నట్లు_అనిపించడం': 'spinning_movements',
                    'సమతుల్యత_కోల్పోవడం': 'loss_of_balance',
                    'అస్థిరత': 'unsteadiness',
                    'ఒక_వైపు_శరీరం_బలహీనత': 'weakness_of_one_body_side',
                    'వాసన_తెలియకపోవడం': 'loss_of_smell',
                    'మూత్రాశయ_అసౌకర్యం': 'bladder_discomfort',
                    'మూత్రం_దుర్వాసన': 'foul_smell_of_urine',
                    'నిరంతరం_మూత్రం_వస్తున్నట్లు_అనిపించడం': 'continuous_feel_of_urine',
                    'వాయువులు_వెళ్ళడం': 'passage_of_gases',
                    'లోపలి_దురద': 'internal_itching',
                    'టైఫస్_లక్షణాలు': 'toxic_look_(typhos)',
                    'నిరాశ': 'depression',
                    'చిరాకు': 'irritability',
                    'కండరాల_నొప్పి': 'muscle_pain',
                    'మార్పు_చెందిన_స్పృహ': 'altered_sensorium',
                    'శరీరంపై_ఎరుపు_మచ్చలు': 'red_spots_over_body',
                    'కడుపు_నొప్పి': 'belly_pain',
                    'అసాధారణ_బహిష్టు': 'abnormal_menstruation',
                    'చర్మంపై_మచ్చలు': 'dischromic_patches',
                    'కళ్ళ_నుండి_నీరు_కారడం': 'watering_from_eyes',
                    'ఆకలి_పెరగడం': 'increased_appetite',
                    'అధిక_మూత్రం': 'polyuria',
                    'కుటుంబ_చరిత్ర': 'family_history',
                    'శ్లేష్మం_కఫం': 'mucoid_sputum',
                    'తుప్పు_రంగు_కఫం': 'rusty_sputum',
                    'ఏకాగ్రత_లేకపోవడం': 'lack_of_concentration',
                    'దృష్టి_సమస్యలు': 'visual_disturbances',
                    'రక్తమార్పిడి_చేయించుకోవడం': 'receiving_blood_transfusion',
                    'అపరిశుభ్ర_సూదులు_వాడటం': 'receiving_unsterile_injections',
                    'కోమా': 'coma',
                    'కడుపులో_రక్తస్రావం': 'stomach_bleeding',
                    'పొత్తికడుపు_ఉబ్బరం': 'distention_of_abdomen',
                    'మద్యం_సేవించే_చరిత్ర': 'history_of_alcohol_consumption',
                    'ద్రవ_అధిక_భారం': 'fluid_overload',
                    'కఫంలో_రక్తం': 'blood_in_sputum',
                    'పిక్కల_సిరలు_వాపు': 'prominent_veins_on_calf',
                    'గుండె_దడ': 'palpitations',
                    'నడవడంలో_నొప్పి': 'painful_walking',
                    'చీము_మొటిమలు': 'pus_filled_pimples',
                    'నల్లమచ్చలు': 'blackheads',
                    'చర్మం_గీరడం': 'scurring',
                    'చర్మం_ఒలవడం': 'skin_peeling',
                    'వెండి_రంగు_పొడి': 'silver_like_dusting',
                    'గోళ్ళలో_చిన్న_గుంటలు': 'small_dents_in_nails',
                    'వాపుతో_కూడిన_గోళ్ళు': 'inflammatory_nails',
                    'బొబ్బ': 'blister',
                    'ముక్కు_చుట్టూ_ఎర్రని_పుండు': 'red_sore_around_nose',
                    'పసుపు_రంగు_కారుతున్న_గాయం': 'yellow_crust_ooze'  
            })

    def extract_symptoms_from_text(self, text):
        """Extract symptoms from text in current language"""
        text = text.lower().strip()
        extracted_symptoms = set()
        
        # Remove common words based on language
        common_words = {
            'en': {'i', 'am', 'have', 'having', 'with', 'and', 'also', 'feeling'},
            'hi': {'मुझे', 'है', 'हैं', 'और', 'भी', 'महसूस', 'कर', 'रहा', 'रही'},
            'te': {'నాకు', 'ఉంది', 'మరియు', 'కూడా', 'అనిపిస్తోంది'}
        }[self.language]
        
        # Clean text
        words = [w for w in text.split() if w not in common_words]
        cleaned_text = ' '.join(words)
        
        # Check for symptoms in cleaned text
        for local_term, eng_symptom in self.symptom_mapping.items():
            if local_term in cleaned_text:
                extracted_symptoms.add(eng_symptom)
        
        return list(extracted_symptoms)

    def get_description(self, condition):
        """Get description in current language"""
        try:
            description = self.df_description[
                self.df_description['disease'].str.strip() == condition.strip()
            ]['description'].values[0]
            return description
        except:
            return self.get_language_text('no_description')

    def get_precautions(self, condition):
        """Get precautions in current language"""
        try:
            precautions = []
            row = self.df_precaution[
                self.df_precaution['Disease'].str.strip() == condition.strip()
            ]
            
            for i in range(1, 5):
                precaution = row[f'Precaution_{i}'].values[0]
                if isinstance(precaution, str) and precaution.strip():
                    precautions.append(precaution.strip())
            
            return precautions if precautions else [self.get_language_text('no_precautions')]
        except:
            return [self.get_language_text('no_precautions')]

    def get_language_text(self, key):
        """Get language-specific text"""
        texts = {
            'en': {
                'no_description': 'No description available.',
                'no_precautions': 'No specific precautions available.',
                'low_confidence': 'Please provide more symptoms for a more accurate diagnosis.'
            },
            'hi': {
                'no_description': 'कोई विवरण उपलब्ध नहीं है।',
                'no_precautions': 'कोई विशेष सावधानियां उपलब्ध नहीं हैं।',
                'low_confidence': 'कृपया अधिक सटीक निदान के लिए अधिक लक्षण बताएं।'
            },
            'te': {
                'no_description': 'వివరణ అందుబాటులో లేదు.',
                'no_precautions': 'ప్రత్యేక జాగ్రత్తలు అందుబాటులో లేవు.',
                'low_confidence': 'దయచేసి మరింత ఖచ్చితమైన రోగనిర్ధారణ కోసం మరిన్ని లక్షణాలను అందించండి.'
            }
        }
        return texts[self.language][key]

    def train_model(self):
        # Prepare training data
        X = self.df_training.drop('prognosis', axis=1)
        y = self.df_training['prognosis']
        
        # Encode disease labels
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

    def predict_condition(self, symptoms):
        # Create input vector
        input_vector = pd.DataFrame(0, index=[0], columns=self.symptoms)
        for symptom in symptoms:
            if symptom in self.symptoms:
                input_vector[symptom] = 1
        
        # Get prediction probabilities
        prediction_proba = self.model.predict_proba(input_vector)[0]
        prediction = self.model.predict(input_vector)[0]
        
        # Get top 3 predictions with their probabilities
        top_3_indices = prediction_proba.argsort()[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            condition = self.encoder.inverse_transform([idx])[0]
            if self.language != 'en':
                condition = self.disease_mappings[self.language].get(condition, condition)
            confidence = prediction_proba[idx] * 100
            top_3_predictions.append((condition, confidence))
        
        return top_3_predictions

    def find_matching_symptom(self, text):
        """Find matching symptoms from text"""
        text = text.lower().strip()
        matches = []
        
        # Check direct matches in symptom phrases
        if text in self.symptom_mapping:
            return [self.symptom_mapping[text]]
        
        # Check for partial matches
        for phrase, symptom in self.symptom_mapping.items():
            if phrase in text or text in phrase:
                matches.append(symptom)
        
        # Check original symptom list
        for symptom in self.symptoms:
            symptom_normalized = symptom.replace('_', ' ')
            if text in symptom_normalized or symptom_normalized in text:
                matches.append(symptom)
        
        return list(set(matches))  # Remove duplicates

    def ensure_consistent_diseases(self):
        """Ensure disease names are consistent across all language datasets"""
        try:
            # Clean disease names in training data
            self.df_training['prognosis'] = self.df_training['prognosis'].str.strip()
            self.df_training['prognosis'] = self.df_training['prognosis'].replace({
                'Dimorphic hemmorhoids(piles)': 'Dimorphic hemorrhoids(piles)',
                'Hypertension ': 'Hypertension',
                'Diabetes ': 'Diabetes'
            })
            
            # Clean disease names in description data
            self.df_description['disease'] = self.df_description['disease'].str.strip()
            
            # Clean disease names in precaution data
            self.df_precaution['Disease'] = self.df_precaution['Disease'].str.strip()
            
            # Get list of diseases from training data
            self.diseases = self.df_training['prognosis'].unique()
            
            # Encode diseases for model
            self.label_encoder.fit(self.diseases)
            
            # Verify diseases exist in description and precaution datasets
            missing_in_description = set(self.diseases) - set(self.df_description['disease'])
            missing_in_precaution = set(self.diseases) - set(self.df_precaution['Disease'])
            
            if missing_in_description or missing_in_precaution:
                print("Warning: Some diseases are missing translations:")
                if missing_in_description:
                    print(f"Missing in description: {missing_in_description}")
                if missing_in_precaution:
                    print(f"Missing in precaution: {missing_in_precaution}")
                
        except Exception as e:
            print(f"Error in disease consistency check: {str(e)}")
            # Initialize with training data diseases anyway
            self.diseases = self.df_training['prognosis'].unique()
            self.label_encoder.fit(self.diseases)
