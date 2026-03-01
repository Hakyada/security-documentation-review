# Step 1: Import Libraries
from sentence_transformers import SentenceTransformer
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 2: Load the Pre-trained Model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Step 3: Extract Text from PDF
def extract_text(pdf_path):
    """
    Extracts text from a PDF file.
    """
    with open(pdf_path, 'rb') as file: 
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Step 4: Split Text into Chunks
def chunk_text(text, chunk_size=300):
    """
    Splits the text into smaller chunks (e.g., 300 words per chunk).
    """
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 5: Encode Document Chunks
def encode_document(document_chunks):
    """
    Converts document chunks into embeddings using the model.
    """
    return np.array(model.encode(document_chunks))  # Ensure embeddings are stored as a NumPy array

# Step 6: Find Relevant Information
def find_relevant_info(query, document_chunks, document_embeddings, top_k=3):
    """
    Finds the most relevant chunks for a given query.
    """
    query_embedding = np.array(model.encode(query)).reshape(1, -1)  # Ensure 2D format
    similarities = cosine_similarity(query_embedding, document_embeddings)  # Compute similarity

    top_indices = np.argsort(similarities[0])[-top_k:]  # Get top K matches
    return [document_chunks[i] for i in reversed(top_indices)], similarities[0][top_indices]

# Step 7: Main Function (Interactive Mode)
def main(pdf_path):
    """
    Main function to process the PDF and allow user to ask questions.
    """
    print("Extracting text from PDF...")
    document_text = extract_text(pdf_path)
    
    if not document_text.strip():
        print("Error: No extractable text found in the PDF.")
        return
    
    print("Splitting text into chunks...")
    document_chunks = chunk_text(document_text)
    
    print("Encoding document chunks...")
    document_embeddings = encode_document(document_chunks)

    print("\nPDF processing complete. You can now ask questions.")

    # Interactive Loop for Questions
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting...")
            break

        print("\nSearching for relevant information...\n")
        results, scores = find_relevant_info(query, document_chunks, document_embeddings)

        print(f"Top results for: {query}")
        for i, (result, score) in enumerate(zip(results, scores), 1):
            print(f"\nMatch {i} (Score: {score:.4f}):\n{result}")

# Step 8: Run the Script
if __name__ == "__main__":
    pdf_path = "/home/hkyada/User Manual v2024.r3.pdf"  # Replace with your PDF file path
    main(pdf_path)
