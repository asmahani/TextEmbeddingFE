from TextEmbeddingFE.embed import Textcol2Mat
import pandas as pd
import os
from dotenv import load_dotenv

dfText = pd.read_csv('C:/Users/alire/OneDrive/data/statman_bitbucket/aki/LLM/AI_in_Medicine_Submission/all_text_columns.csv')

load_dotenv()

google_project_id = os.getenv('VERTEXAI_PROJECT')
google_location = os.getenv('VERTEXAI_LOCATION')

ret = Textcol2Mat(
    type = 'google'
    , google_project_id = google_project_id
    , google_location = google_location
).fit_transform(dfText[['operation']][:10])
