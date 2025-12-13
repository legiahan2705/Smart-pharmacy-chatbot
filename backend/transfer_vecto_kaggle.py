# --- 1. CÀI ĐẶT THƯ VIỆN ---
# !pip install -q langchain langchain-community faiss-cpu sentence-transformers

import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("Bắt đầu quá trình vector hóa dữ liệu...")

# --- 2. CẤU HÌNH ---
JSONL_FILE_PATH = "/kaggle/input/longchau-drugs-jsonl/longchau_drugs.jsonl"
VECTOR_STORE_PATH = "/kaggle/working/faiss_index_longchau"

# --- 3. KHỞI TẠO MÔ HÌNH EMBEDDING CỦA HUGGINGFACE ---
print("Đang tải mô hình embedding local. Lần đầu có thể mất vài phút...")
# Sử dụng GPU sẽ tự động được ưu tiên trên Kaggle nếu có
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# --- 4. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ---
print(f"Đang đọc dữ liệu từ file: {JSONL_FILE_PATH}")
documents = []

with open(JSONL_FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        product = json.loads(line)

        # Lấy các giá trị ra biến để dễ đọc và xử lý
        category = product.get("category", "chưa rõ")
        indications = product.get("indications", "chưa rõ công dụng")
        product_name = product.get("product_name", "")
        active_ingredient = (
            product.get("active_ingredient", "")
            .replace("Thông tin thành phần Hàm lượng", "")
            .strip()
        )
        usage_instructions = product.get("usage_instructions", "")

        # 1. Tạo page_content tối ưu
        # Nguyên tắc: Công dụng lên đầu, phân loại rõ ràng, và dùng câu văn tự nhiên.
        page_content = f"""
        Sản phẩm này chủ yếu dùng để {indications}.
        Phân loại chính: {category}. Đây là một sản phẩm thuộc nhóm {category}, không phải là thuốc kháng sinh hay thuốc điều trị các bệnh lý chuyên khoa nặng.
        Tên đầy đủ của sản phẩm là {product_name}.
        Đối tượng và cách sử dụng: {usage_instructions}.
        Thành phần chính bao gồm: {active_ingredient}.
        """.strip()

        # 2. Tạo metadata
        # Nguyên tắc: Chứa các dữ liệu có cấu trúc để hiển thị hoặc lọc.
        metadata = {
            "product_id": product.get("product_id"),
            "product_url": product.get("product_url"),
            "product_name": product.get("product_name"),
            "image_url": product.get("image_url"),
            "price_VND": product.get("price_VND"),
            "unit": product.get("unit"),
            "packaging": product.get("packaging"),
            "form": product.get("form"),
            "manufacturer": product.get("manufacturer"),
            "brand_origin": product.get("brand_origin"),
        }
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

print(f"Đã đọc và xử lý thành công {len(documents)} sản phẩm.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)
print(f"Đã chia {len(documents)} tài liệu thành {len(split_docs)} đoạn nhỏ (chunks).")


# --- 5. VECTOR HÓA VÀ LƯU TRỮ (PHIÊN BẢN ĐƠN GIẢN VÀ HIỆU QUẢ) ---
import time  # Import time ở đây để đo thời gian

print("Bắt đầu vector hóa toàn bộ dữ liệu (không cần chia lô)...")
start_time = time.time()

# Chỉ cần một dòng lệnh duy nhất để xử lý tất cả
vector_db = FAISS.from_documents(split_docs, embeddings)

# Lưu lại CSDL
vector_db.save_local(VECTOR_STORE_PATH)

end_time = time.time()
print("-" * 50)
print(f"✅ HOÀN TẤT! ✅")
print(f"Thời gian thực thi: {((end_time - start_time) / 60):.2f} phút")
print(f"Cơ sở dữ liệu vector đã được lưu thành công vào thư mục: '{VECTOR_STORE_PATH}'")
print("-" * 50)
