PROJECT_ID = PROJECT_ID
REGION = us-central1
MODEL_ID = MODEL_ID

import vertexai
from vertexai.language_models import TextEmbeddingModel

vertexai.init(project=PROJECT_ID, location=REGION)

model = TextEmbeddingModel.from_pretrained(MODEL_ID)
embeddings = model.get_embeddings(...)