import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd

# Load the medical NER pipeline
MODEL_NAME = "d4data/biomedical-ner-all"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_medical_entities(text: str):
    """
    Extracts entities from medical text and formats output for Gradio UI.
    Returns:
1. Highlighted Text for the visual display.
2. Dataframe summarizing organized entities.
    """
    if not text.strip():
        return [], pd.DataFrame(columns=["Entity", "Type", "Confidence Score"])

    # Extract entities using transformers pipeline
    raw_entities = ner_pipe(text)

    # Format 1: Output structure for Gradio's gr.HighlightedText
    highlighted_output = []
    last_idx = 0

    # Format 2: Tabular structured summary
    structured_records = []

    for entity in raw_entities:
        start = entity['start']
        end = entity['end']
        word = entity['word']
        entity_group = entity['entity_group']
        score = round(float(entity['score']), 4)

        # Append preceding non-entity text
        if start > last_idx:
            highlighted_output.append((text[last_idx:start], None))

        # Append the entity with its entity group/label
        highlighted_output.append((text[start:end], entity_group))
        last_idx = end

        # Record for dataframe
        structured_records.append({
            "Entity": text[start:end],
            "Category": entity_group,
            "Confidence Score": f"{score * 100:.2f}%"
        })

    # Append any remaining un-annotated text
    if last_idx < len(text):
        highlighted_output.append((text[last_idx:], None))

    # Convert records to Pandas DataFrame
    df_summary = pd.DataFrame(structured_records)
    if df_summary.empty:
        df_summary = pd.DataFrame(columns=["Entity", "Category", "Confidence Score"])

    return highlighted_output, df_summary


# Sample clinical text examples for testing
sample_text_1 = (
    "Patient presents with severe headache, high fever, and shortness of breath over the past 3 days. "
    "Diagnosed with acute bronchitis. Prescribed Amoxicillin 500mg and Acetaminophen for pain management."
)

sample_text_2 = (
    "58-year-old male reporting type 2 diabetes mellitus and hypertension. "
    "Currently taking Metformin 1000mg twice daily and Lisinopril 10mg. Patient complains of fatigue and mild dizziness."
)

# Build Gradio Interface
with gr.Blocks(title="Medical NLP - Entity Extraction") as demo:
    gr.Markdown("# 🩺 Medical Entity Extraction System")
    gr.Markdown("Extract **symptoms, diseases, medications, procedures, and anatomical entities** from unstructured clinical notes.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input Medical Text / Clinical Note",
                placeholder="Enter patient note, medical history, or diagnostic text here...",
                lines=6
            )
            submit_btn = gr.Button("Extract Entities", variant="primary")

            gr.Examples(
                examples=[[sample_text_1], [sample_text_2]],
                inputs=[text_input]
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Highlighted Text Analysis")
            highlighted_output = gr.HighlightedText(
                label="Extracted Entities",
                combine_adjacent=True,
                show_legend=True
            )

            gr.Markdown("### Structured Entity Summary")
            df_output = gr.Dataframe(
                headers=["Entity", "Category", "Confidence Score"],
                interactive=False
            )

    # Wire event listener
    submit_btn.click(
        fn=extract_medical_entities,
        inputs=[text_input],
        outputs=[highlighted_output, df_output]
    )

if __name__ == "__main__":
    demo.launch() 