# Medical-Text-Symptom-Information-Extractor
Interactive clinical NLP web app using Python, Gradio, and Hugging Face's d4data/biomedical-ner-all to extract symptoms, diseases, and medications from clinical notes. Displays inline color-coded highlights and a structured summary table with confidence scores. For demo/educational use only.
Project Description
The Medical Text Symptom & Entity Information Extractor is an interactive clinical Natural Language Processing (NLP) web application built with Python, Gradio, PyTorch, Pandas, and Hugging Face Transformers using the d4data/biomedical-ner-all token-classification model. Unstructured clinical notes—such as physician intake summaries, symptom descriptions, discharge notes, and patient histories—contain vital clinical signals buried inside free text. This system automates Named Entity Recognition (NER) to detect and categorize clinically relevant spans, including symptoms, diseases and disorders, medications, dosages, and anatomical references, presenting them both inline within the source text and as a structured tabular summary with confidence scores.

Designed with a gr.Blocks user interface, the application supports custom clinical text entry as well as one-click sample notes (gr.Examples) so users can immediately test how symptom and treatment mentions are parsed without manual data entry.

End-to-End Workflow
Input Acquisition: A user enters unstructured clinical text into the gr.Textbox component or clicks one of the preloaded clinical sample notes (sample_text_1 or sample_text_2).

Input Validation: When the user clicks Extract Entities, extract_medical_entities(text) checks whether text.strip() is non-empty. Empty or whitespace-only inputs immediately return an empty highlight list and an empty Pandas DataFrame with defined columns so the UI never crashes on blank submissions.

Tokenization & Biomedical NER Inference: Valid text is passed into the Hugging Face ner pipeline backed by d4data/biomedical-ner-all with aggregation_strategy="simple". Subword pieces belonging to the same entity are merged into coherent entity spans along with character offsets (start, end), predicted group labels (entity_group), and confidence scores (score).

Dual Output Formatting:

Highlighted Span List: Iterates through detected entities using character indices (start and end) to slice exact substrings from the original text, interleaving un-annotated segments as (segment, None) and entity segments as (text[start:end], entity_group). Preserving character offsets avoids tokenizer subword artifacts (such as ## prefixes) and keeps original casing and spacing intact.

Structured Summary Table: Builds a list of dictionaries containing each extracted Entity, Category (entity_group), and percentage-formatted Confidence Score (rounded to four decimal places before converting to percentage string), then wraps it in a pandas.DataFrame.

UI Rendering & Inspection: Gradio updates gr.HighlightedText (with combine_adjacent=True and show_legend=True) and gr.Dataframe (interactive=False) so users can visually inspect entities in context and review them systematically in a table.

Pipeline Architecture
Plaintext
[Clinical Text Input / Sample Note]
               │
               ▼
   [Whitespace Check (text.strip())]
        │                  │
   (Empty)            (Non-Empty)
        │                  │
        ▼                  ▼
[Empty List & DF]  [AutoTokenizer: d4data/biomedical-ner-all]
                           │
                           ▼
             [AutoModelForTokenClassification]
                           │
                           ▼
         [HF NER Pipeline (aggregation_strategy="simple")]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
  [Character-Span Slicer]      [Record Builder]
   (start/end -> text[s:e])   (Entity, Category, Score)
             │                           │
             ▼                           ▼
  [gr.HighlightedText]          [pd.DataFrame -> gr.Dataframe]
How It Works Under the Hood
1. Model Loading & Aggregation Strategy
At startup, the script loads AutoTokenizer and AutoModelForTokenClassification from d4data/biomedical-ner-all and wraps them in transformers.pipeline("ner", ..., aggregation_strategy="simple"). Transformer tokenizers often split medical terms into subwords (for example, breaking a compound word or medication name into multiple tokens). Setting aggregation_strategy="simple" groups contiguous tokens predicted to belong to the same entity class and computes an aggregated confidence score for the span.

2. Offset-Based Span Reconstruction
Rather than relying solely on the pipeline's reconstructed word field, extract_medical_entities uses entity['start'] and entity['end'] to slice text[start:end]. A last_idx pointer tracks progress across the string:

Any slice between last_idx and start is appended with label None (plain background text).

The entity slice text[start:end] is appended with entity_group as its tag.

After the loop finishes, any trailing text from last_idx to len(text) is appended with label None.

This guarantees that the reconstructed text matches the user's input character-for-character.

3. Tabular Summary Generation
Each extracted entity is also appended to structured_records with its exact span text, biomedical category, and formatted confidence score (for example, 99.85%). Converting structured_records into a pandas.DataFrame gives gr.Dataframe a clean tabular structure; if no entities are found in a non-empty string, an empty DataFrame with ["Entity", "Category", "Confidence Score"] columns is returned so the table headers remain consistent.

Limitations & Clinical Disclaimer
Model & Context Scope: NER models identify named spans and categories in text—they do not perform full clinical reasoning, negation detection (such as distinguishing "no fever" from "fever"), or temporal timeline verification unless explicitly modeled downstream.

Non-Clinical Use Only: This application is intended for demonstration, prototyping, and educational exploration of biomedical NLP and should not be used for medical diagnosis, triage, or treatment decisions.
