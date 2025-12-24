import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load môi trường
load_dotenv()

def test_index():
    path = "backend/faiss_index"
    if not os.path.exists(path):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{path}'")
        return

    # --- THỬ NGHIỆM 1: DÙNG MODEL GOOGLE (Ưu tiên) ---
    print("\n🔵 TEST 1: Thử đọc bằng Google Embeddings (text-embedding-004)...")
    try:
        gg_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = FAISS.load_local(path, gg_embeddings, allow_dangerous_deserialization=True)
        print(f"   -> Load thành công! Tổng số vector: {vectorstore.index.ntotal}")
        
        # Test search KHÔNG CÓ THRESHOLD
        print("   -> Đang tìm thử 'Panadol Extra'...")
        docs_and_scores = vectorstore.similarity_search_with_score("Thuốc Panadol Extra có công dụng gì", k=3)
        
        for doc, score in docs_and_scores:
            # Lưu ý: FAISS L2 distance càng thấp càng tốt, Cosine similarity càng cao càng tốt.
            # Langchain thường trả về distance.
            print(f"      - Score: {score:.4f} | Tên thuốc: {doc.metadata.get('source', 'Unknown')}")
            # print(f"        Content: {doc.page_content[:100]}...")
            
    except Exception as e:
        print(f"   -> ❌ Thất bại với Google: {e}")

    # --- THỬ NGHIỆM 2: DÙNG MODEL HUGGINGFACE (Fallback) ---
    print("\n🟠 TEST 2: Thử đọc bằng HuggingFace (all-MiniLM-L6-v2)...")
    try:
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(path, hf_embeddings, allow_dangerous_deserialization=True)
        print(f"   -> Load thành công! Tổng số vector: {vectorstore.index.ntotal}")
        
        print("   -> Đang tìm thử 'Panadol Extra'...")
        docs_and_scores = vectorstore.similarity_search_with_score("Thuốc Panadol Extra có công dụng gì", k=3)
        for doc, score in docs_and_scores:
            print(f"      - Score: {score:.4f} | Tên thuốc: {doc.metadata.get('source', 'Unknown')}")
            
    except Exception as e:
        print(f"   -> ❌ Thất bại với HuggingFace: {e}")

if __name__ == "__main__":
    test_index()