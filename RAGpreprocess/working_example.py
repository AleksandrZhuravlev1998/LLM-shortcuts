from ChatBotLLM import RunLLM
from RAGpreprocess import RAG_preprocess
import os 
import pandas as pd
from tqdm import tqdm 
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


#==============================================================
# Translate PDFs into pure text and then vectorise them
#==============================================================

# The destination needs to contain folders with PDFs to turn into text
main_path = r"path_to_your_file"
folders_list = os.listdir(main_path)
print(os.getcwd())
parent_db = pd.DataFrame()
count = 0

# For each folder, iterate over all documents 
for folder in tqdm(folders_list):
    folder_path = Path(main_path) / folder
    documents_list = os.listdir(folder_path)
    for doc in tqdm(documents_list):
        doc_path =  folder_path / doc
        if not doc[-3:]=="pdf":
            continue 
        # PDF to text and then to vector 
        db_temp = RAG_preprocess.create_vector_db_entry(path=doc_path,
                                      chunk_length=1500, 
                                      split_by_page=False)
        db_temp["Folder name"] = folder
        parent_db = pd.concat([parent_db, db_temp])
        count+1
        # Backup 
        if count%100 == 0:
            parent_db.to_parquet("Fund DB vectorised")

parent_db.to_parquet("Fund DB vectorised")



# Get the matrix of embeddings and store as parquet 
embeddings_matrix = RAG_preprocess.get_embedding_matrix(parent_db)
table = pa.Table.from_arrays(
    [pa.array(embeddings_matrix[:, i]) for i in range(embeddings_matrix.shape[1])],
    names=[f"dim_{i}" for i in range(embeddings_matrix.shape[1])]
)

pq.write_table(table, "embeddings matrix.parquet")


# Normalise the embeddings matrix to efficiently compute similarities
normalised_embeddings = RAG_preprocess.normalise_db(embeddings_matrix)
normalised_table = pa.Table.from_arrays(
    [pa.array(normalised_embeddings[:, i]) for i in range(normalised_embeddings.shape[1])],
    names=[f"dim_{i}" for i in range(normalised_embeddings.shape[1])]
)
pq.write_table(normalised_table, "normalised embeddings matrix.parquet")


#====================================
# Use the vector DB to enrich prompts
#====================================

prompt = "TYPE YOUR PROMPT HERE"
# Enrich with data
enriched_prompt = RAG_preprocess.enrich_prompt(prompt, parent_db, normalised_embeddings)
out = RunLLM.single_run(enriched_prompt)
print(out)