def doc_split(doc_text:str , model_name:str, chunk_size: int, chunk_overlap:int = 0):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name=model_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(doc_text)
    return chunks