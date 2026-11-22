# Medical-Text-Symptom-Information-Extractor
1. Introduction
This code implements a Medical Named Entity Recognition (NER) System using Natural Language Processing (NLP) and a pre-trained biomedical language model. The system is designed to identify and extract important medical information from unstructured clinical text, such as patient medical histories, symptoms, diseases, medications, and treatment details.

It uses the pre-trained d4data/biomedical-ner-all model from Hugging Face Transformers to recognize medical entities and classify them into appropriate categories. A Gradio-based graphical user interface (GUI) allows users to enter medical text and view the extracted information in highlighted and tabular formats.
2. Technologies Used
Technology	Purpose
Python	Main programming language
Gradio	Creates the interactive user interface
Hugging Face Transformers	Loads and executes the NER model
Biomedical NER Model	Identifies medical entities
Pandas	Organizes extracted entities into a table
Tokenizer	Converts input text into model-readable tokens

3. Working of the Code
The system works through the following steps:
System Workflow
1. Input Medical Text

User enters clinical notes or patient medical history.


2. Tokenization

Tokenizer converts the text into tokens for model processing.


3. Biomedical NER Model

Identifies medical terms and predicts their entity categories.


4. Entity Extraction

Extracts entity names, categories, and confidence scores.


5A. Highlighted Text

Displays medical entities highlighted in the original text.


5B. Structured Table

Displays entity names, categories, and confidence scores.




4. Explanation of Each Code Section
A. Importing Required Libraries
import gradio as grfrom transformers import AutoTokenizer, AutoModelForTokenClassification, pipelineimport pandas as pd



- Gradio: Creates the web-based user interface.
- Transformers: Provides pre-trained NLP models and tokenization.
- Pandas: Creates and manages structured tabular data.
B. Loading the Biomedical NER Model
MODEL_NAME = "d4data/biomedical-ner-all"tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)ner_pipe = pipeline(    "ner",    model=model,    tokenizer=tokenizer,    aggregation_strategy="simple")



This section loads the pre-trained biomedical NER model.
- The tokenizer converts medical text into tokens.
- The model identifies medical entities.
- The NER pipeline performs named entity recognition.
- aggregation_strategy="simple" combines token-level predictions into complete entity predictions where possible.
C. Medical Entity Extraction Function
def extract_medical_entities(text: str):



This is the main processing function of the system.

It accepts medical text from the user and extracts relevant medical information.

Main operations:
1. Checks whether the input text is empty.
2. Sends the text to the biomedical NER pipeline.
3. Extracts entity positions and names.
4. Identifies the predicted entity category.
5. Calculates and formats confidence scores.
6. Creates highlighted text.
7. Generates a structured DataFrame.
D. Extracting Entity Information
start = entity['start']end = entity['end']word = entity['word']entity_group = entity['entity_group']score = round(float(entity['score']), 4)



This section retrieves the information associated with each recognized entity.
Variable	Description
start	Starting position of the entity in the input text
end	Ending position of the entity
word	Recognized entity text
entity_group	Predicted medical entity category
score	Model confidence score

For example, the system may identify:
- Headache → Symptom
- Diabetes mellitus → Disease
- Metformin → Medication
The exact category depends on the model's predictions.
E. Generating Highlighted Text
highlighted_output.append((text[start:end], entity_group))



This section highlights recognized medical entities within the original clinical text.

It also preserves the remaining text that was not identified as an entity.

This makes it easier for users to visually identify important medical terms.
F. Creating the Structured Entity Table
structured_records.append({    "Entity": text[start:end],    "Category": entity_group,    "Confidence Score": f"{score * 100:.2f}%"})



This section stores the extracted information in a structured format.

Example output:
Entity	Category	Confidence Score
Headache	Symptom	98.50%
Fever	Symptom	96.70%
Bronchitis	Disease	97.20%
Amoxicillin	Medication	95.80%

The scores above are illustrative, not actual model output.
G. Sample Clinical Text
The code provides two sample medical notes for testing.

Sample 1:

A patient with severe headache, high fever, shortness of breath, acute bronchitis, and prescribed medications.

Sample 2:

A 58-year-old male with type 2 diabetes mellitus and hypertension, taking Metformin and Lisinopril.

Users can select these examples to test the entity extraction system without manually entering text.
H. Designing the Gradio Interface
with gr.Blocks(title="Medical NLP - Entity Extraction") as demo:



This section creates the graphical user interface.

It contains:
1. Title: Medical Entity Extraction System.
2. Input Textbox: Accepts clinical notes.
3. Extract Entities Button: Starts the entity extraction process.
4. Examples: Provides sample clinical notes.
5. Highlighted Text Output: Displays recognized medical entities.
6. Dataframe Output: Displays extracted entities in tabular form.
I. Connecting the Button to the Function
submit_btn.click(    fn=extract_medical_entities,    inputs=[text_input],    outputs=[highlighted_output, df_output])



When the user clicks the Extract Entities button:
- The input medical text is sent to the extraction function.
- The NER model processes the text.
- Highlighted entities are displayed.
- The structured entity table is generated.
J. Launching the Application
if __name__ == "__main__":    demo.launch()



This command launches the Gradio application and provides a local web interface through which users can interact with the system.

5. Expected Output
Medical Entity Extraction System
Illustrative interface preview

Input Medical Text
Patient presents with severe headache, high fever, and shortness of breath. Diagnosed with acute bronchitis. Prescribed Amoxicillin.
Highlighted Text Analysis
Entities detected

headache · Symptom
high fever · Symptom
acute bronchitis · Disease
Amoxicillin · Medication

Structured Entity Summary
Entity	Category	Confidence
Headache	Symptom	98.5%
High fever	Symptom	96.7%
Acute bronchitis	Disease	97.2%
Amoxicillin	Medication	95.8%


6. Applications
1. Clinical Information Extraction: Extracts important medical terms from patient records.
2. Medical Record Analysis: Helps organize unstructured clinical notes.
3. Disease Information Identification: Identifies disease names mentioned in medical documents.
4. Medication Extraction: Recognizes medicines and their mentions in clinical text.
5. Healthcare Research: Helps researchers process large amounts of medical text.
6. Medical NLP Systems: Can be integrated into larger healthcare information processing applications.
7. Advantages
- Automates medical entity identification.
- Reduces manual effort in reviewing clinical text.
- Provides confidence scores for predictions.
- Displays results in both highlighted and structured formats.
- Uses a pre-trained biomedical NLP model.
- Provides a simple and interactive interface.
8. Limitations
- The model may incorrectly identify some medical terms.
- It does not independently diagnose diseases.
- Confidence scores do not guarantee medical correctness.
- It may not understand the full clinical context, such as negation (e.g., "no fever") or historical mentions.
- It depends on the quality of the input text and the model's training data.
Important: This system performs medical named entity recognition, not disease prediction or medical diagnosis.
9. Conclusion
The Medical Named Entity Recognition System uses Natural Language Processing and a pre-trained biomedical model to extract important medical information from unstructured clinical text. By combining entity recognition, confidence scoring, highlighted text, and structured tabular output, the system makes medical information easier to identify, organize, and analyze. The Gradio interface provides a user-friendly way to interact with the system.

In simple words: The code takes medical text as input, identifies important medical terms such as symptoms, diseases, and medicines, calculates their model confidence scores, highlights them in the original text, and displays the extracted information in a table.
