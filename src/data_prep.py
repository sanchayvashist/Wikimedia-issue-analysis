import os
import json

from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from utils import generate_embeddings, extract_plain_text, MODEL_HIDDEN_SIZE

load_dotenv()

# Initialize the Qdrant client
qdrant_client = QdrantClient(url = os.getenv("QDRANT_URL"), api_key= os.getenv("QDRANT_API_KEY"))  # Update with your Qdrant configuration

# Create a Qdrant collection
collection_name = "doctor-droid"
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=MODEL_HIDDEN_SIZE, distance=Distance.COSINE),
)

# # Function to load text files and create embeddings
def load_files_and_create_embeddings(directory):
    documents = []
    file_metadata = []

    for filename in tqdm(os.listdir(directory)):
        try:
            file_path = os.path.join(directory, filename)
            text = extract_plain_text(file_path)
            documents.append(text)
            file_metadata.append({"file_name":filename, "path": os.path.dirname(file_path)})
        except Exception as e:
            print(file_path, e)
    # Generate embeddings
    embeddings = generate_embeddings(documents)

    return embeddings, file_metadata

if __name__ == "__main__":
    # Load files and generate embeddings
    embeddings, metadata = load_files_and_create_embeddings("data/assignment_data")

    # Upload embeddings to Qdrant
    points = [
        {
            "id": i,
            "vector": embedding.tolist(),
            "payload": metadata[i],
        }
        for i, embedding in enumerate(embeddings)
    ]

    # Save vector data locally
    with open("data/vector_data.json", "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=4)

    qdrant_client.upsert(collection_name=collection_name, points=points)

    print(f"Embeddings for {len(points)} files have been uploaded to the Qdrant collection '{collection_name}'.")