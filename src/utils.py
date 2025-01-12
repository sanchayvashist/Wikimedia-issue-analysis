import torch
import os
import wikitextparser as wtp

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai


load_dotenv("src/.env")
# Initialize the embedding model (BGE)
model_path = "BAAI/bge-small-en"

# Save the tokenizer and model locally
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

MODEL_HIDDEN_SIZE = model.config.hidden_size

# Get api key from enviroment and intialize model
key = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=key)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")


def extract_plain_text(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        wikitext = file.read()

    # Parse the Wikitext
    parsed = wtp.parse(wikitext)

    # Extract plain text by stripping markup
    plain_text = parsed.plain_text()
    return plain_text


# Function to generate embeddings using BGE model
def generate_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(
            dim=1
        )  # Average pooling of token embeddings
    return embeddings.cpu().numpy()


def get_gemini_response(message):
    response = gemini_model.generate_content(message)
    usage_dict = response.usage_metadata
    return response.text, usage_dict
