# --- 1. CÀI ĐẶT THƯ VIỆN ---
!pip install -qU langchain langchain-core langchain-community langchain-google-genai google-generativeai faiss-cpu langchain-text-splitters
import json
import os
# --- THÊM 2 DÒNG NÀY ĐỂ CHỐNG TREO MÁY TRÊN KAGGLE ---
os.environ["USE_TF"] = "0"    # Cấm load TensorFlow

import time
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# try:
#     user_secrets = UserSecretsClient()
#     api_key = user_secrets.get_secret("GOOGLE_API_KEY")
#     os.environ["GOOGLE_API_KEY"] = api_key
# except Exception as e:
#     print("❌ LỖI: Chưa cấu hình Secret 'GOOGLE_API_KEY'.")
#     raise e

# DÁN TRỰC TIẾP API KEY MỚI VÀO ĐÂY (Nằm trong ngoặc kép)
api_key = "" 
os.environ["GOOGLE_API_KEY"] = api_key

print("⏳ Đang tải mô hình Google Embeddings (gemini-embedding-001)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# --- 4. ĐỌC DỮ LIỆU & TẠO CONTENT CHI TIẾT ---
print(f"📂 Đang đọc dữ liệu từ file: {JSON_FILE_PATH}")
documents = []

try:
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data_array = json.load(f) 
        
    # ... (code đọc file json ở Phần 4)
    print(f"   -> Tìm thấy {len(data_array)} dòng dữ liệu thô.")
    
    # CHIẾN THUẬT CHIA ĐỂ TRỊ: Chạy đợt 1 (Từ 501 đến 1000)
    data_array = data_array[0:500] 
    
    print(f"   -> Đang chạy ĐỢT 1: Xử lý {len(data_array)} sản phẩm.")
    
    for product in data_array:
        try:
            # 1. Lấy thông tin cơ bản
            name = product.get("Tên thuốc") or product.get("product_name") or ""
            if not name: continue 

            # 2. Lấy thông tin chi tiết (Ưu tiên tiếng Việt, fallback sang tiếng Anh)
            # Hàm get an toàn: lấy value, nếu k có trả về chuỗi rỗng
            # --- CODE MỚI: QUÉT KEY THÔNG MINH ---
            # Hàm tìm value dựa trên từ khóa bắt đầu (startswith)
            def get_dynamic_key(item_dict, prefix):
                for k, v in item_dict.items():
                    if str(k).startswith(prefix):
                        return str(v).strip()
                return ""

            # Dùng hàm mới để quét các key hay bị đổi tên
            danh_muc = product.get("Danh mục") or product.get("category") or ""
            
            # Quét "Thành phần của..."
            thanh_phan = get_dynamic_key(product, "Thành phần").replace("Thông tin thành phần Hàm lượng", "")
            
            # Quét "Công dụng của..."
            cong_dung = get_dynamic_key(product, "Công dụng")
            
            # Quét "Cách dùng..." hoặc "Liều dùng..."
            lieu_dung = get_dynamic_key(product, "Cách dùng") or get_dynamic_key(product, "Liều dùng")
            
            # Các key cố định thì dùng .get() bình thường
            tac_dung_phu = product.get("Tác dụng phụ", "")
            luu_y = product.get("Lưu ý", "")
            chong_chi_dinh = product.get("Chống chỉ định", "")
            bao_quan = product.get("Bảo quản", "")
            
            nha_san_xuat = product.get("Nhà sản xuất", "")
            nuoc_san_xuat = product.get("Nước sản xuất", "")
            xuat_xu = product.get("Xuất xứ thương hiệu", "")
            dang_bao_che = product.get("Dạng bào chế", "")
            quy_cach = product.get("Quy cách", "")
            # -------------------------------------

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

# --- 5. TẠO VECTOR INDEX (CƠ CHẾ AUTO-RETRY BẤT TỬ) ---
print("⚡ Bắt đầu tạo Vector Index (Google Version)...")
start_time = time.time()

try:
    batch_size = 50
    vector_db = None
    
    total_batches = (len(split_docs) + batch_size - 1) // batch_size
    print(f"📦 Dữ liệu được chia thành {total_batches} lô để xử lý an toàn.")

    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i : i + batch_size]
        current_batch = (i // batch_size) + 1
        
        print(f"   -> Đang nhúng (embedding) lô {current_batch}/{total_batches}...")
        
        # --- VÒNG LẶP RETRY: Kẻ thù của lỗi 429 ---
        # --- VÒNG LẶP RETRY: Kẻ thù của lỗi 429 ---
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if vector_db is None:
                    vector_db = FAISS.from_documents(batch, embeddings)
                else:
                    temp_db = FAISS.from_documents(batch, embeddings)
                    vector_db.merge_from(temp_db)
                
                # NẾU THÀNH CÔNG -> Thoát vòng lặp retry, đi tới lô tiếp theo
                break 
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"      ⏳ Quá tải (429) ở lô {current_batch}... (Lần {attempt + 1}/{max_retries})")
                    
                    # NẾU THỬ 5 LẦN VẪN CHẾT -> HẾT QUOTA NGÀY -> CỨU DỮ LIỆU & DỪNG HẲN
                    if attempt == max_retries - 1:
                        print("🚨 BÁO ĐỘNG ĐỎ: HẾT QUOTA NGÀY! DỪNG TOÀN BỘ CHƯƠNG TRÌNH!")
                        if vector_db is not None:
                            vector_db.save_local(VECTOR_STORE_PATH)
                            !zip -r faiss_index_partial.zip {VECTOR_STORE_PATH}
                            print("✅ Đã lưu khẩn cấp thành công: faiss_index_partial.zip")
                        raise Exception("Đã cạn kiệt API Key. Chương trình tự hủy để tránh treo máy vô ích.")
                        
                    time.sleep(60) 
                else:
                    print(f"      ❌ Lỗi lạ ở lô {current_batch}: {error_msg}")
                    break
        
        # Ngủ nhẹ 5 giây giữa các lô bình thường để không dồn dập
        time.sleep(5) 
        
    # LƯU FILE CUỐI CÙNG
    if vector_db is not None:
        vector_db.save_local(VECTOR_STORE_PATH)
        
        end_time = time.time()
        print("-" * 50)
        print(f"🎉 THÀNH CÔNG! FAISS Index đã được tạo xong.")
        print(f"⏱️ Thời gian: {((end_time - start_time) / 60):.2f} phút")
        print("-" * 50)
        
        # Nén file lại
        !zip -r faiss_index.zip {VECTOR_STORE_PATH}
        print("✅ Đã nén xong: faiss_index.zip. BẠN CÓ THỂ TẢI VỀ RỒI!")
    else:
        print("❌ Thất bại: Không có dữ liệu nào được lưu.")
        
except Exception as e:
    print(f"❌ Lỗi hệ thống: {e}")