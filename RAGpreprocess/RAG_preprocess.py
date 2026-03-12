from pypdf import PdfReader
import math
from ChatBotLLM import RunLLM
import pandas as pd 
from pathlib import Path
import numpy as np 

def pdf_to_text(path:str, return_by_page:bool=False)->str|list:
    """
    Translates a pdf document into a string

    Parameters
    ----------
    path : str
        The path to the pdf file.
    return_by_page : bool
        Should the text be returned as a single string or list with entries per page? Default is False 

    Returns
    -------
    output_text : str | list
        Text version of the pdf. A single string if return_by_page=False or list of string entries otherwise.
    """
    # creating a pdf reader object
    reader = PdfReader(path)

    if return_by_page:
        output_text = []
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            # extracting text from page
            text = page.extract_text()
            output_text.append(text)
    else:
        output_text = ""
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            # extracting text from page
            text = page.extract_text()
            output_text+=text

    return output_text

def get_document_chunks(document:str, chunk_length:int) -> list:
    """
    Divides a given text document into equally spaced chunks 

    Parameters
    ----------
    document : str
        Document text.
    chunk_length : int
        The length of chunks to be generated. 

    Returns
    -------
    chunks_list : list
        List of equally sized text chunks
    """

    # Define the number of chunks to be produced 
    n_chunks = math.ceil(len(document)/chunk_length)
    chunks_list = []
    for i in range(n_chunks):
        chunk = document[chunk_length*i:chunk_length*(i+1)]
        chunks_list.append(chunk)
    
    return chunks_list

def create_vector_db_entry(path, chunk_length:int |None=200, split_by_page:bool=False) -> pd.DataFrame:
    """
    Creates a single vector database entry for a given PDF document. Depending on the document length, it produces a dataframe with several entries per document, where each entry is a vectorised chunk from the document 

    Parameters
    ----------
    path : str
        Path to the pdf file
    chunk_length : int
        The length of chunks to be generated. Default is 200. 
    split_by_page: int|None
        Should the document be split into pages or in equally spaced chunks? Default is False. Cannot be True if chunk_length is specified. 

    Returns
    -------
    df : pd.DataFrame
        A dataframe rows indicating a chunk of the document. The dataframe also contains the original Text of the chunk, its ID, and an embedding vector.
    """    
    if split_by_page and chunk_length is not None:
        raise(ValueError("You cannot set chunk length if split_by_page is True"))
    
    document_text = pdf_to_text(path=path,
                                return_by_page=split_by_page)
    
    if not split_by_page:
        document_text = get_document_chunks(document_text, chunk_length)
    
    embeddings_list = [] 

    for t in document_text:
        embed = RunLLM.tokeniser_single_run(t)
        embeddings_list.append(embed)
    
    df = pd.DataFrame({"Text": document_text,
                       "ChunkID": [i for i in range(len(document_text))],
                       "Embedding": embeddings_list})
    path_standardised = Path(path)
    df["DocumentName"] = path_standardised.name

    return df

def get_embedding_matrix(df: pd.DataFrame) -> np.array:
    """
    Extracts the embeddings matrix from the vectorised database.

    Parameters
    ----------
    df : pd.DataFrame
        The vectorised database in the format returned by create_vector_db_entry()

    Returns
    -------
    embedding_matrix : np.array
        Matrix of embeddings 
    """
    embedding_matrix  = np.array(df["Embedding"].to_list())
    return embedding_matrix


def normalise_db(vector_db: np.array) -> np.array:
    """
    Performs a simple normalisation of an embedding matrix.

    Parameters
    ----------
    vector_db : np.array
        A matrix of embeddings produced from a vectorised database.

    Returns
    -------
    matrix_normalised : np.array
        Matrix of normalised document embeddings.
    """
    matrix_normalised = vector_db / np.linalg.norm(vector_db, axis=1, keepdims=True)
    return matrix_normalised


def compute_similarity(vectorised_prompt: list, 
                       vector_db_normalised: np.array,
                       return_n_most_similar: int=3) -> list:
    """
    Computes cosine similarity between a vectorised LLM prompt and the vector database to return database indices of n most similar chunks.

    Parameters
    ----------
    vectorised_prompt : list
        An embedding vector obtained from vectorising a user prompt.
    vector_db_normalised : np.array
        Matrix of normalised document embeddings for the database
    return_n_most_similar : int 
        The number of most similar document chunks to return 

    Returns
    -------
    top_n_idx : list
        List of indices of most similar documents from the database
    """
    # Convert list to numpy array
    v = np.asarray(vectorised_prompt)
    
    # Normalise vector
    v_norm = v / np.linalg.norm(v)
    
    # Compute cosine similarity
    similarities = vector_db_normalised @ v_norm   # shape (m,)
    
    # Get indices of top k similarities (largest values)
    top_n_idx = np.argsort(similarities)[-return_n_most_similar:][::-1].tolist()
    
    return top_n_idx

def enrich_prompt(prompt_initial:str, db:pd.DataFrame, normalised_embeddings:np.array, n: int|None = None,
                   length_instructions: int|None = "======== \n Final note: keep your response to 1 sentence. No more! It's a strict requirement"
                   ) -> str:
    """
    Performs RAG (Retrieval-Augmented Generation) procedures on a user-defined prompt by enriching it with documents from the database that are the most likely to be relevant to help the LLM to produce a correct response.

    Parameters
    ----------
    prompt_initial : str
        User's prompt
    db : pd.DataFrame
        Database with original texts 
    normalised_embeddings : np.array 
        Matrix of normalised embeddings
    n : int | None
        Number of chunks to be appended to the prompt. Default is None. If None, the function appends all chunks of the document that contains the most relevant chunk. Keeping at None is advisable, as it ensures that the data come from the same source.  
    length_instructions: int | None
        Additional prompt instructions appended at the end of the augmented prompt. By default the LLM is asked to return strictly one sentence. 

    Returns
    -------
    prompt_enriched : str
        A prompt enriched with text from the most relevant document(s)
    """

    # First, the prompt is vectorised 
    vectorised_prompt = RunLLM.tokeniser_single_run(prompt_initial)

    # Then, search for the most relevant chunk(s)
    if n is None:
        # If none, we append ALL chunks from the most similar document 
        similar_indices = compute_similarity(vectorised_prompt, normalised_embeddings, 1)
        doc_name = db.iloc[similar_indices, :]["DocumentName"].item()
        texts_to_append = db.loc[db["DocumentName"]==doc_name, :]["Text"].to_list()
    
    # Else - we append the most similar chunks (they are likely to come from different docs)
    else:    
        similar_indices = compute_similarity(vectorised_prompt, normalised_embeddings, n)
        texts_to_append = db.iloc[similar_indices, :]["Text"].to_list()

    prompt_enriched = prompt_initial + "\n To aid your reasoning, please consult the texts below. Whenever referring to the document, stick to 'According to our database'.  If the supplemented documents do not contain relevant information, politely ask the user to reformulate the request. \n TEXTS \n"

    for t in texts_to_append:
        prompt_enriched = prompt_enriched + t + "\n"
    if length_instructions is not None:
        prompt_enriched+= length_instructions 
    return prompt_enriched


    