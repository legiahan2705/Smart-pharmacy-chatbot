# --- 1. CÀI ĐẶT THƯ VIỆN ---
!pip install -q langchain langchain-community faiss-cpu langchain-google-genai

import json
import os
import time
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from kaggle_secrets import UserSecretsClient 

print("🚀 Bắt đầu quá trình vector hóa dữ liệu (FULL DETAIL VERSION)...")

# --- 2. CẤU HÌNH ---
# Kiểm tra lại đường dẫn file input của bạn
JSON_FILE_PATH = "/kaggle/input/data-longchau/longchau_selected.json" 
VECTOR_STORE_PATH = "/kaggle/working/faiss_index"

# --- 3. KHỞI TẠO API & MODEL ---
print("🔑 Đang lấy API Key...")
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
except Exception as e:
    print("❌ LỖI: Chưa cấu hình Secret 'GOOGLE_API_KEY'.")
    raise e

print("⏳ Đang tải mô hình Google Embeddings (text-embedding-004)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# --- 4. ĐỌC DỮ LIỆU & TẠO CONTENT CHI TIẾT ---
print(f"📂 Đang đọc dữ liệu từ file: {JSON_FILE_PATH}")
documents = []

try:
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data_array = json.load(f) 
        
    print(f"   -> Tìm thấy {len(data_array)} dòng dữ liệu thô.")
    
    for product in data_array:
        try:
            # 1. Lấy thông tin cơ bản
            name = product.get("Tên thuốc") or product.get("product_name") or ""
            if not name: continue 

            # 2. Lấy thông tin chi tiết (Ưu tiên tiếng Việt, fallback sang tiếng Anh)
            # Hàm get an toàn: lấy value, nếu k có trả về chuỗi rỗng
            def get_safe(key_vi, key_en):
                val = product.get(key_vi) or product.get(key_en) or ""
                return str(val).strip()

            danh_muc = get_safe("Danh mục", "category")
            thanh_phan = get_safe("Thành phần", "active_ingredient").replace("Thông tin thành phần Hàm lượng", "")
            cong_dung = get_safe("Công dụng", "indications")
            lieu_dung = get_safe("Liều dùng", "usage_instructions")
            
            # --- QUAN TRỌNG: CÁC TRƯỜNG "SÂU" MÀ BẠN CẦN ---
            tac_dung_phu = get_safe("Tác dụng phụ", "side_effects")
            luu_y = get_safe("Lưu ý", "precautions") # Chứa thông tin về gan, thận
            chong_chi_dinh = get_safe("Chống chỉ định", "contraindications") # Chứa thông tin về bà bầu, trẻ em
            bao_quan = get_safe("Bảo quản", "preservation")
            
            nha_san_xuat = get_safe("Nhà sản xuất", "manufacturer")
            nuoc_san_xuat = get_safe("Nước sản xuất", "country_of_origin")
            xuat_xu = get_safe("Xuất xứ thương hiệu", "brand_origin")
            dang_bao_che = get_safe("Dạng bào chế", "form")
            quy_cach = get_safe("Quy cách", "packaging")

            # 3. Xây dựng Page Content "Siêu đầy đủ"
            # AI sẽ đọc đoạn văn bản này để trả lời. Càng chi tiết càng tốt.
            page_content = f"""
            Tên sản phẩm: {name}
            Danh mục: {danh_muc}
            Dạng bào chế: {dang_bao_che}
            Quy cách đóng gói: {quy_cach}
            Xuất xứ: Thương hiệu {xuat_xu}, Sản xuất tại {nuoc_san_xuat} bởi {nha_san_xuat}.

            THÀNH PHẦN:
            {thanh_phan}

            CÔNG DỤNG & CHỈ ĐỊNH:
            {cong_dung}

            CÁCH DÙNG & LIỀU DÙNG:
            {lieu_dung}

            CHỐNG CHỈ ĐỊNH (Không dùng cho):
            {chong_chi_dinh}

            LƯU Ý & THẬN TRỌNG (Cảnh báo an toàn):
            {luu_y}

            TÁC DỤNG PHỤ CÓ THỂ GẶP:
            {tac_dung_phu}
            
            BẢO QUẢN:
            {bao_quan}
            """.strip()

            # 4. Metadata (Dùng để lọc nếu cần, hoặc hiển thị UI)
            metadata = {
                "source": name,
                "price": str(product.get("Giá bán") or product.get("price_VND") or "0"),
                "origin": xuat_xu
            }
            
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
            
        except Exception as e:
            continue 

except FileNotFoundError:
    print(f"❌ KHÔNG TÌM THẤY FILE TẠI: {JSON_FILE_PATH}")
    exit()

if len(documents) == 0:
    print("❌ CẢNH BÁO: Không xử lý được sản phẩm nào.")
    exit()

print(f"✅ Đã chuẩn hóa FULL DATA cho {len(documents)} sản phẩm.")

# Chia nhỏ văn bản
# Tăng chunk_size lên 1500 vì content bây giờ rất dài
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)
print(f"📦 Đã chia thành {len(split_docs)} chunks.")

# --- 5. TẠO VECTOR INDEX ---
print("⚡ Bắt đầu tạo Vector Index (Google Version)...")
start_time = time.time()

try:
    vector_db = FAISS.from_documents(split_docs, embeddings)
    vector_db.save_local(VECTOR_STORE_PATH)
    
    end_time = time.time()
    print("-" * 50)
    print(f"🎉 THÀNH CÔNG! FAISS Index (Full Detail) đã được tạo.")
    print(f"⏱️ Thời gian: {((end_time - start_time) / 60):.2f} phút")
    print("-" * 50)
    
    # Nén file lại
    !zip -r faiss_index.zip {VECTOR_STORE_PATH}
    print("✅ Đã nén xong: faiss_index.zip. Hãy tải về ngay!")
    
except Exception as e:
    print(f"❌ Lỗi tạo Vector: {e}")