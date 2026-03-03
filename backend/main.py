import os
import json
import gc
import requests
import base64
import io         
import edge_tts   # <--- Sửa lỗi vàng cho TTS
from dotenv import load_dotenv
from typing import TypedDict, Literal

# --- FastAPI Imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse 
from fastapi.middleware.cors import CORSMiddleware

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, HarmBlockThreshold, HarmCategory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.memory import MemorySaver

import pandas as pd
import re

# =======================================================
# 0. KHỞI TẠO & CẤU HÌNH
# =======================================================
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# 1. XỬ LÝ DỮ LIỆU (DÙNG PARQUET ĐỂ TĂNG TỐC KHỞI ĐỘNG)
# =======================================================
def load_and_clean_data() -> pd.DataFrame:
    parquet_path = "data/optimized_db.parquet"
    source_json_path = "data/longchau_selected.json"

    # 1. Load Cache (Ưu tiên)
    if os.path.exists(parquet_path):
        print("⚡ [Pandas] Tìm thấy Cache Parquet. Đang tải...")
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"⚠️ Cache lỗi ({e}), sẽ tự động xóa và xử lý lại từ đầu...")
            os.remove(parquet_path) # Xóa cache hỏng

    # 2. Xử lý lần đầu (Tối ưu bộ nhớ chống tràn RAM)
    print("🐢 [Pandas] Bắt đầu đọc file JSON...")
    try:
        with open(source_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   -> Đã đọc xong {len(data)} sản phẩm. Đang chuẩn hóa dữ liệu...")

        # BƯỚC B: CHỈ TRÍCH XUẤT CÁC CỘT CẦN THIẾT CHO PANDAS
        cleaned_data = []
        for item in data:
            # Hàm quét các key bị đổi tên (Thành phần của..., Công dụng của...)
            def get_dynamic(prefix):
                for k, v in item.items():
                    if str(k).startswith(prefix):
                        return str(v).strip()
                return ""

            # Tạo dictionary mới sạch sẽ, KHÔNG rác
            clean_item = {
                "Tên thuốc": item.get("Tên thuốc", item.get("product_name", "")),
                "Giá bán": item.get("Giá bán", item.get("price_VND", "0")),
                "Danh mục": item.get("Danh mục", item.get("category", "")),
                "Dạng bào chế": item.get("Dạng bào chế", item.get("form", "")),
                "Quy cách": item.get("Quy cách", item.get("packaging", "")),
                "Xuất xứ thương hiệu": item.get("Xuất xứ thương hiệu", item.get("brand_origin", "")),
                "Nước sản xuất": item.get("Nước sản xuất", item.get("country_of_origin", "")),
                "Nhà sản xuất": item.get("Nhà sản xuất", item.get("manufacturer", "")),
                "Thành phần": get_dynamic("Thành phần"),
                "Công dụng": get_dynamic("Công dụng")
            }
            cleaned_data.append(clean_item)

        print("   -> Đang chuyển sang DataFrame...")
        
        # BƯỚC C: Tạo DataFrame từ list đã làm sạch (Bảo đảm không bị nổ cột)
        df = pd.DataFrame(cleaned_data)
        
        # Xóa biến tạm để giải phóng RAM ngay lập tức
        del data 
        del cleaned_data
        gc.collect() 
        
        print("   -> Đang làm sạch cột giá và điền dữ liệu thiếu...")
        df['Giá bán'] = df['Giá bán'].astype(str)
        df['price_int'] = df['Giá bán'].str.replace(r'[^\d]', '', regex=True)
        df['price_int'] = pd.to_numeric(df['price_int'], errors='coerce').fillna(0).astype(int)

        df = df.fillna('')
        
        print("   -> Đang lưu Cache Parquet (Bước cuối)...")
        # Chuyển tất cả về string (trừ price_int) để tránh lỗi format của Parquet
        for col in df.columns:
            if col != 'price_int':
                df[col] = df[col].astype(str)
        
        df.to_parquet(parquet_path, index=False)
        print(f"✅ [Pandas] Xử lý xong và đã lưu Cache vào {parquet_path}.")
        
        return df

    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG KHI XỬ LÝ DATA: {e}")
        return pd.DataFrame()

_df_cache = None

def get_df_safe() -> pd.DataFrame: # Thêm -> pd.DataFrame để IDE hiểu và hết báo vàng
    global _df_cache
    if _df_cache is None: 
        _df_cache = load_and_clean_data()
    return _df_cache

# =======================================================
# 2. MODELS & VECTORSTORE
# =======================================================
# Dùng Flash để nhanh, temperature thấp để chính xác
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, # Giữ 0 để trả lời chính xác theo tài liệu
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
)
print("⏳ Đang tải mô hình Embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

def load_vectorstore():
    index_path = "faiss_index"
    if os.path.exists(index_path):
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Lỗi load FAISS: {e}")
            return None
    return None

vectorstore = load_vectorstore()
if vectorstore:
    # Tăng tốc bằng cách lọc bớt rác ngay từ đầu (score_threshold)
    # Sửa trong main.py
    retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 10, 
        # "score_threshold": 0.3  <-- XÓA HOẶC COMMENT DÒNG NÀY ĐI
    }
)
else:
    print("⚠️ Không có Vectorstore!")

# =======================================================
# 3. CORE LOGIC (TỐI ƯU HÓA: ONE-SHOT BRAIN)
# =======================================================

class AppState(TypedDict):
    question: str
    chat_history: list[str]
    language: str
    intent_data: dict # Chứa kết quả phân tích gộp (Safety + Route + Keyword)
    context: str | None
    answer: str | None

# --- MESSAGES ---
EMPATHETIC_SAFETY_MESSAGE = """Tôi không thể cung cấp thông tin về cách sử dụng thuốc để gây hại cho bản thân. Các loại thuốc chỉ an toàn khi được sử dụng đúng liều lượng theo chỉ dẫn của bác sĩ hoặc dược sĩ. Việc sử dụng quá liều có thể gây nguy hiểm nghiêm trọng đến sức khỏe và tính mạng.

Nếu bạn đang gặp khó khăn hoặc có ý định tự tử, xin hãy tìm kiếm sự giúp đỡ ngay lập tức. Có rất nhiều nguồn hỗ trợ sẵn sàng lắng nghe và giúp đỡ bạn. Bạn có thể liên hệ với các đường dây nóng hỗ trợ tâm lý hoặc nói chuyện với người thân, bạn bè, hoặc chuyên gia y tế.

Một số đường dây nóng hỗ trợ tâm lý tại Việt Nam mà bạn có thể liên hệ:
* Tổng đài quốc gia bảo vệ trẻ em 111
* Tổng đài tư vấn sức khỏe tâm thần 1900 561203
* Hoặc tìm kiếm sự hỗ trợ từ các bệnh viện, phòng khám chuyên khoa tâm thần gần nhất.

Hãy nhớ rằng bạn không đơn độc và có sự giúp đỡ dành cho bạn."""

# --- NODE 1: "THE BRAIN" (GỘP SAFETY + ROUTER + EXPANSION) ---
brain_prompt_template = """
Bạn là bộ não trung tâm của hệ thống AI y tế Long Châu. Nhiệm vụ của bạn là phân tích câu hỏi và trả về kết quả dưới dạng JSON.

HÃY THỰC HIỆN 3 BƯỚC PHÂN TÍCH SAU:

BƯỚC 1: KIỂM DUYỆT AN TOÀN (SAFETY CHECK)
Phân tích xem câu hỏi có chứa ý định nguy hiểm không.
- ĐẶT "is_unsafe": true KHI VÀ CHỈ KHI:
   + Người dùng có ý định TỰ TỬ, TỰ HẠI rõ ràng (muốn chết, tìm cách chết, hỏi liều gây tử vong).
   + Hỏi cách ĐẦU ĐỘC người khác.
   + Hỏi mua thuốc cấm/ma túy.

- ĐẶT "is_unsafe": false (AN TOÀN) KHI:
   + Người dùng hỏi về "lỡ uống quá liều", "uống nhầm", "quên liều".
   + Người dùng lo lắng về tác dụng phụ khi uống nhiều (Ví dụ: "Uống 2 viên có sao không?").
   -> TRƯỜNG HỢP NÀY CẦN TRẢ VỀ false ĐỂ HỆ THỐNG TÌM KIẾM THÔNG TIN TƯ VẤN CÁCH XỬ LÝ.

BƯỚC 2: ĐỊNH TUYẾN (ROUTING)
Xác định loại câu hỏi để chọn nguồn dữ liệu:
- Ưu tiên chọn "structured_analysis" (Pandas) nếu câu hỏi chứa các TIÊU CHÍ LỌC CỤ THỂ:
   + Hỏi về GIÁ CẢ (rẻ nhất, đắt nhất, bao nhiêu tiền).
   + Hỏi về DẠNG BÀO CHẾ (dạng gói, dạng viên, siro, thuốc bôi, thuốc nước, hỗn dịch).
   + Hỏi về XUẤT XỨ (của Mỹ, của Pháp, nước nào sản xuất).
   + Hỏi về QUY CÁCH (hộp bao nhiêu viên).
- Chỉ chọn "vector_search" khi hỏi thuần túy về kiến thức: Công dụng là gì? Cách dùng thế nào? Bệnh này uống thuốc gì (không yêu cầu dạng cụ thể)?
-> Gán giá trị vào trường "route".

BƯỚC 3: MỞ RỘNG CÂU HỎI (QUERY EXPANSION)
Chuyển đổi câu hỏi thành từ khóa tìm kiếm chuyên sâu:
1. Luôn thêm từ khóa "Thuốc", "Điều trị", "Dược phẩm".
2. Nếu mô tả triệu chứng, thêm tên các HOẠT CHẤT (Active Ingredients) phổ biến.
3. Đọc Lịch sử trò chuyện để giải quyết đại từ nhân xưng (nó, thuốc này) nếu cần.
4. QUAN TRỌNG NHẤT: Bất kể người dùng hỏi bằng ngôn ngữ nào, các từ khóa trong trường "keywords" BẮT BUỘC phải được dịch sang Tiếng Việt để tìm kiếm trong cơ sở dữ liệu.

INPUT DATA:
Language: {language}
History: {chat_history}
Question: {question}

OUTPUT JSON FORMAT (Không được phép trả về Markdown, chỉ JSON thuần):
{{
    "is_unsafe": boolean,
    "route": "vector_search" | "structured_analysis",
    "keywords": "string"
}}
"""
brain_chain = PromptTemplate.from_template(brain_prompt_template) | llm | JsonOutputParser()

async def brain_node(state: AppState):
    print("--- 🧠 THE BRAIN IS THINKING (One-Shot) ---")
    question = state["question"]
    # Chỉ lấy 4 câu gần nhất để prompt không quá dài nhưng vẫn đủ context
    history = "\n".join(state.get("chat_history", [])[-4:]) 
    
    # Keyword check nhanh (Lớp thủ công)
    danger_keywords = ["tự tử", "muốn chết", "tự sát", "liều chết", "tự vẫn", "quyên sinh", "đầu độc", "cắt cổ", "uống thuốc độc"]
    if any(k in question.lower() for k in danger_keywords):
        print("!!! SAFETY TRIGGERED (KEYWORD) !!!")
        return {"intent_data": {"is_unsafe": True, "route": "none", "keywords": ""}}

    # AI Check (Lớp thông minh)
    try:
        # Lấy language từ state (mặc định là 'vi' nếu không có)
        user_lang = state.get("language", "vi") 
        
        result = await brain_chain.ainvoke({
            "question": question, 
            "chat_history": history,
            "language": user_lang  # <--- Truyền vào biến {language} trong prompt
        })
        print(f"Brain Analysis: {result}")
        return {"intent_data": result}
    except Exception as e:
        print(f"Brain Error: {e}. Fallback to vector search.")
        # Fallback an toàn nếu lỗi JSON
        return {"intent_data": {"is_unsafe": False, "route": "vector_search", "keywords": question}}
    

# --- NODE 2: RETRIEVE ---
# --- SỬA LẠI HAM RETRIEVE_NODE ---
async def retrieve_node(state: AppState):
    print("--- 🔍 RETRIEVE ---")
    query = state["intent_data"].get("keywords", state["question"])
    print(f"Searching: {query}")
    
    docs = await retriever.ainvoke(query)
    
    # --- DEBUG LOG QUAN TRỌNG ---
    if not docs:
        print("❌ CẢNH BÁO: Không tìm thấy tài liệu nào (Docs rỗng)!")
        return {"context": ""}
        
    print(f"✅ Đã tìm thấy {len(docs)} tài liệu.")
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"   -> Tổng độ dài Context: {len(context)} ký tự")
    # ----------------------------

    return {"context": context}

# --- NODE 3: PANDAS (Prompt Đầy Đủ Cũ) ---
pandas_prompt_template = """
Bạn có một pandas DataFrame tên là `df` chứa dữ liệu thuốc.
Các cột quan trọng cần dùng: 
- 'Tên thuốc'
- 'price_int' (Giá bán dạng số nguyên. 0 nghĩa là "Liên hệ nhà thuốc").
- 'Giá bán' (Giá dạng chuỗi hiển thị, ví dụ: "570.000đ").
- 'Danh mục' (Ví dụ: "Dầu cá, Omega 3, DHA", "Thuốc giảm đau").
- 'Xuất xứ thương hiệu' (Ví dụ: "Hoa Kỳ", "Pháp").
- 'Nước sản xuất' (Ví dụ: "Ba Lan", "Việt Nam").
- 'Dạng bào chế' (Viên nén, Siro, Viên nang mềm...).
- 'Quy cách' (Ví dụ: "Hộp 6 Vỉ x 20 Viên").

Nhiệm vụ: Viết MỘT dòng code Python để lọc dữ liệu và trả lời câu hỏi.
Kết quả phải được gán vào biến `result`.

QUY TẮC LOGIC QUAN TRỌNG:
1. NGỮ CẢNH (CONTEXT):
   - Đọc kỹ "Lịch sử trò chuyện". Nếu câu hỏi hiện tại thiếu chủ ngữ (ví dụ: "tìm dạng viên", "loại nào rẻ hơn"), HÃY LẤY TÊN BỆNH HOẶC TÊN THUỐC TỪ LỊCH SỬ ÁP VÀO.
   - Ví dụ: History nói về "đau dạ dày". User hỏi "tìm dạng viên". -> Code phải tìm thuốc "đau dạ dày" VÀ "dạng viên".

2. Lọc theo "Dạng bào chế":
   - Nếu hỏi "Dạng gói": Tìm chứa 'Gói' OR 'Bột' OR 'Hỗn dịch' OR 'Gel' OR 'Dung dịch'.
   - Nếu hỏi "Dạng viên": Tìm chứa 'Viên' OR 'Nang'.
   
3. Lọc theo "Bệnh/Công dụng": Tìm trong CẢ 3 CỘT: `Danh mục` OR `Tên thuốc` OR `Công dụng`.
4. Luôn thêm điều kiện `df['price_int'] > 0`.
5. KẾT QUẢ: Luôn hiển thị cột `Tên thuốc`, `Giá bán`, `Dạng bào chế`.

Ví dụ 1:
Question: Tìm 3 loại thuốc Omega 3 rẻ nhất.
Python: result = df[(df['Danh mục'].str.contains('Omega 3', case=False, na=False)) & (df['price_int'] > 0)].nsmallest(3, 'price_int')[['Tên thuốc', 'Giá bán', 'Quy cách', 'Xuất xứ thương hiệu']].to_string()

Ví dụ 2:
Question: Có bao nhiêu loại thuốc của Mỹ?
Python: result = f"Có {{len(df[(df['Nước sản xuất'].str.contains('Mỹ|Hoa Kỳ|USA', case=False, na=False)) | (df['Xuất xứ thương hiệu'].str.contains('Mỹ|Hoa Kỳ|USA', case=False, na=False))])}} thuốc có xuất xứ hoặc thương hiệu Mỹ."

Ví dụ 3:
Question: Liệt kê các thuốc dạng Siro giá dưới 50000.
Python: result = df[(df['Dạng bào chế'].str.contains('Siro', case=False, na=False)) & (df['price_int'] > 0) & (df['price_int'] < 50000)][['Tên thuốc', 'Giá bán', 'Quy cách']].to_string()

QUY TẮC LOGIC QUAN TRỌNG:
1. NGỮ CẢNH (CONTEXT):
   - Đọc kỹ "Lịch sử trò chuyện". Nếu câu hỏi hiện tại thiếu chủ ngữ (ví dụ: "tìm dạng viên", "loại nào rẻ hơn"), HÃY LẤY TÊN BỆNH HOẶC TÊN THUỐC TỪ LỊCH SỬ ÁP VÀO.
   - Ví dụ: History nói về "đau dạ dày". User hỏi "tìm dạng viên". -> Code phải tìm thuốc "đau dạ dày" VÀ "dạng viên".

2. Lọc theo "Dạng bào chế":
   - Nếu hỏi "Dạng gói": Tìm chứa 'Gói' OR 'Bột' OR 'Hỗn dịch' OR 'Gel' OR 'Dung dịch'.
   - Nếu hỏi "Dạng viên": Tìm chứa 'Viên' OR 'Nang'.
   
3. Lọc theo "Bệnh/Công dụng": Tìm trong CẢ 3 CỘT: `Danh mục` OR `Tên thuốc` OR `Công dụng`.
4. Luôn thêm điều kiện `df['price_int'] > 0`.
5. KẾT QUẢ: Luôn hiển thị cột `Tên thuốc`, `Giá bán`, `Dạng bào chế`.

LƯU Ý QUAN TRỌNG: Khi kết quả là một DataFrame có nhiều dòng, bạn BẮT BUỘC phải luôn sử dụng .head(5) 
trước khi gọi .to_string() để chỉ lấy tối đa 5 sản phẩm tiêu biểu nhất, tránh làm tràn bộ nhớ. 
Ví dụ: result = df[...].head(5).to_string()

Lịch sử trò chuyện: {chat_history}
Question: {question}
Python:
"""
pandas_chain = PromptTemplate.from_template(pandas_prompt_template) | llm | StrOutputParser()

async def structured_analysis_node(state: AppState):
    print("--- 🐼 PANDAS ANALYSIS ---")
    question = state["question"]
    # 1. Lấy lịch sử chat để Pandas hiểu ngữ cảnh
    history = "\n".join(state.get("chat_history", [])[-4:]) # Lấy 4 câu gần nhất
    df = get_df_safe()
    
    # 2. Truyền thêm chat_history vào invoke
    code = await pandas_chain.ainvoke({
        "question": question, 
        "chat_history": history # <--- QUAN TRỌNG: Truyền history vào đây
    })
    
    clean_code = code.replace("```python", "").replace("```", "").strip()
    print(f"Generated Code: {clean_code}") # In ra để debug xem nó có lọc đúng 'dạ dày' không

    local_vars = {"df": df, "result": None}
    try:
        exec(clean_code, {}, local_vars)
        result = local_vars["result"]
        final = result.to_string() if hasattr(result, 'to_string') else str(result)
        final_answer = f"Số liệu tìm được:\n{final}"
    except Exception as e:
        final_answer = f"Lỗi tính toán: {e}"
        
    return {"answer": final_answer}

    
    
# --- NODE 4: GENERATE (Prompt Đầy Đủ Cũ) ---
# Prompt này giữ nguyên y hệt bản gốc của bạn
generate_prompt_template = """
Bạn là một trợ lý tư vấn thuốc thông minh của Long Châu.

NGUYÊN TẮC AN TOÀN TUYỆT ĐỐI (SAFETY GUARDRAILS):
1. TỰ TỬ & LÀM HẠI BẢN THÂN: Nếu người dùng hỏi về liều lượng gây chết người, cách tự tử... -> TỪ CHỐI TRẢ LỜI.
2. QUÁ LIỀU/UỐNG NHẦM: Cảnh báo đi khám bác sĩ, sau đó cung cấp thông tin tham khảo từ Context.
3. KHÔNG THAY THẾ BÁC SĨ: Với các triệu chứng nghiêm trọng, khuyên đi khám ngay.
4. KHÔNG BỊA ĐẶT: Chỉ trả lời dựa trên Context và Lịch sử.

Lịch sử hội thoại: {chat_history}

Context: {context}

Language: {lang_instruction}

Question: {question}
Answer:
"""
rag_generation_chain = PromptTemplate.from_template(generate_prompt_template) | llm | StrOutputParser()

async def generate_node(state: AppState):
    print("--- ✍️ GENERATE ---")
    question = state["question"]
    context = state.get("context", "")
    
    user_lang = state.get("language", "vi")
    if user_lang == "en":
        lang_instruction = "IMPORTANT: You MUST generate your final response entirely in ENGLISH. Translate the medical information accurately from the provided Vietnamese context."
    else:
        lang_instruction = "QUAN TRỌNG: Bạn phải trả lời hoàn toàn bằng Tiếng Việt."
    
    # Nếu không có context thì báo ngay
    if not context:
        print("⚠️ Context rỗng, bỏ qua bước gọi LLM.")
        answer = "Xin lỗi, tôi không tìm thấy thông tin về loại thuốc này trong cơ sở dữ liệu."
    else:
        history = "\n".join(state.get("chat_history", []))
        try:
            answer = await rag_generation_chain.ainvoke({
                "question": question, 
                "context": context, 
                "chat_history": history,
                "lang_instruction": lang_instruction
            })
            # DEBUG: In câu trả lời ra terminal xem nó có bị rỗng không
            print(f"🤖 AI Answer: {answer}") 
        except Exception as e:
            print(f"Error Generate: {e}")
            answer = "Xin lỗi, tôi gặp lỗi khi tạo câu trả lời."
    
    new_history = state.get("chat_history", []) + [f"User: {question}", f"AI: {answer}"]
    return {"answer": answer, "chat_history": new_history}

async def update_history_pandas(state: AppState):
    # Node phụ để cập nhật lịch sử cho nhánh Pandas (vì Pandas Node return answer trực tiếp)
    question = state["question"]
    answer = state["answer"]
    new_history = state.get("chat_history", []) + [f"User: {question}", f"AI: {answer}"]
    return {"chat_history": new_history}

# =======================================================
# 4. XÂY DỰNG GRAPH
# =======================================================
def build_rag_agent():
    workflow = StateGraph(AppState)

    # Add Nodes
    workflow.add_node("brain", brain_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("structured_analysis", structured_analysis_node)
    workflow.add_node("update_history_pandas", update_history_pandas) # Node phụ để lưu history

    # Entry Point
    workflow.set_entry_point("brain")

    # Routing Logic
    def route_decision(state):
        intent = state["intent_data"]
        
        # 1. Nếu không an toàn -> End ngay (trả lời ở API handler)
        if intent.get("is_unsafe"):
            print("--- ROUTING: UNSAFE -> END ---")
            return "unsafe"
            
        # 2. Định tuyến bình thường
        route = intent.get("route")
        print(f"--- ROUTING TO: {route} ---")
        if route == "structured_analysis":
            return "structured_analysis"
        else:
            return "vector_search"

    workflow.add_conditional_edges(
        "brain",
        route_decision,
        {
            "structured_analysis": "structured_analysis",
            "vector_search": "retrieve",
            "unsafe": END
        }
    )

    # Edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Nhánh Pandas: Tính toán -> Lưu history -> End
    workflow.add_edge("structured_analysis", "update_history_pandas")
    workflow.add_edge("update_history_pandas", END)

    return workflow.compile(checkpointer=MemorySaver())

rag_agent = build_rag_agent()
print("🚀 TURBO BACKEND (FULL PROMPTS) READY!")

# =======================================================
# 5. API ENDPOINT
# =======================================================
class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default_user"
    language: str = "vi"

@app.post("/chat")
async def chat_handler(request: ChatRequest):
    print(f"--> User: {request.question}")
    config = {"configurable": {"thread_id": request.thread_id}}
    
    result = await rag_agent.ainvoke({"question": request.question, "language": request.language}, config=config)
    
    # Kiểm tra Safety từ kết quả Brain
    intent = result.get("intent_data", {})
    if intent.get("is_unsafe"):
        final_answer = EMPATHETIC_SAFETY_MESSAGE
    else:
        final_answer = result.get("answer", "Lỗi: Không có câu trả lời.")
    
    return {"answer": final_answer}

# =======================================================
# 6. AUDIO ENDPOINTS 
# =======================================================

# --- A. Text-to-Speech (TTS) - Dùng Edge-TTS (Miễn phí, Giọng hay) ---
@app.post("/tts")
async def text_to_speech(request: ChatRequest):
    """
    Nhận text -> Trả về file âm thanh MP3 (Streaming)
    """
    text = request.question # Lấy đoạn văn bản cần đọc
    
    if request.language == "en":
        voice = "en-US-AriaNeural"  # Giọng nữ tiếng Anh chuẩn Mỹ (rất hay)
    else:
        voice = "vi-VN-HoaiMyNeural" # Giọng nữ tiếng Việt (rất tự nhiên)

    # Tạo giao tiếp với Edge TTS
    communicate = edge_tts.Communicate(text, voice)
    
    # Tạo bộ nhớ đệm để chứa âm thanh (không cần lưu file rác vào ổ cứng)
    audio_stream = io.BytesIO()
    
    # Ghi dữ liệu vào stream
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    
    audio_stream.seek(0) # Tua lại đầu băng
    
    # Trả về dạng stream để Frontend phát được ngay
    return StreamingResponse(audio_stream, media_type="audio/mpeg")

# --- B. Speech-to-Text (STT) - Dùng Gemini qua REST API (Không xung đột) ---
@app.post("/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    language: str = Form("vi")  # <-- Đưa vào đây mới đúng chuẩn FastAPI
):
    """
    Nhận file âm thanh -> Gửi trực tiếp qua HTTP Request tới Gemini
    """
    # 1. Đọc dữ liệu file
    file_bytes = await file.read()
    
    # 2. Mã hóa sang Base64
    base64_audio = base64.b64encode(file_bytes).decode('utf-8')
    
    # 🌟 Chọn prompt theo ngôn ngữ
    if language == "en":
        stt_prompt = "Please listen to this audio and transcribe it exactly in English. Return only the transcribed text without any introductory words."
    else:
        stt_prompt = "Hãy nghe đoạn âm thanh này và chép lại nguyên văn nội dung bằng tiếng Việt. Chỉ trả về nội dung văn bản, không thêm lời dẫn."
    
    # 3. Xác định Mime Type
    mime_type = "audio/mp3" 
    if file.filename.endswith(".wav"): mime_type = "audio/wav"
    elif file.filename.endswith(".webm"): mime_type = "audio/webm"
    
    # 4. Cấu hình Key & URL
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"text": "Lỗi: Chưa cấu hình GOOGLE_API_KEY"}
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    # 5. Tạo Payload 
    payload = {
        "contents": [{
            "parts": [
                {"text": stt_prompt},                
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_audio
                    }
                }
            ]
        }]
    }

    try:
        print("📤 Đang gửi Inline Audio tới Gemini (REST API)...")
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            result_json = response.json()
            try:
                text_result = result_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                print(f"🎙️ Gemini nghe được: {text_result}")
                return {"text": text_result}
            except KeyError:
                return {"text": ""}
        else:
            return {"text": ""}

    except Exception as e:
        print(f"❌ Lỗi kết nối STT: {e}")
        return {"text": ""}
    
# =======================================================
# 7. UNIFIED ENDPOINT (ALL-IN-ONE)
# =======================================================
@app.post("/chat-voice-flow")
async def chat_voice_flow(
    file: UploadFile = File(...), 
    thread_id: str = Form("default_user"), 
    language: str = Form("vi")
):
    """
    Quy trình Full: Nhận Audio -> STT -> Agent xử lý -> TTS -> Trả về JSON (Text + Audio Base64)
    """
    print(f"🎤 Nhận yêu cầu Voice Chat từ user: {thread_id} (Ngôn ngữ: {language})")

    # --- BƯỚC 1: STT (Speech to Text) ---
    file_bytes = await file.read()
    base64_audio = base64.b64encode(file_bytes).decode('utf-8')
    
    mime_type = "audio/mp3"
    if file.filename.endswith(".wav"): mime_type = "audio/wav"
    elif file.filename.endswith(".webm"): mime_type = "audio/webm"

    api_key = os.environ.get("GOOGLE_API_KEY")
    stt_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    # 🌟 Chọn prompt STT
    stt_prompt = "Please listen to this audio and transcribe it exactly in English. Return only the transcribed text without any introductory words." if language == "en" else "Hãy nghe đoạn âm thanh này và chép lại nguyên văn nội dung bằng tiếng Việt. Chỉ trả về nội dung văn bản, không thêm lời dẫn."
    
    stt_payload = {
        "contents": [{
            "parts": [
                {"text": stt_prompt},
                {"inline_data": {"mime_type": mime_type, "data": base64_audio}}
            ]
        }]
    }

    user_text = ""
    try:
        resp = requests.post(stt_url, json=stt_payload, headers={"Content-Type": "application/json"})
        if resp.status_code == 200:
            user_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"   -> Nghe được: {user_text}")
        else:
            return {"error": "Không nghe rõ giọng nói"}
    except Exception as e:
        return {"error": f"Lỗi kết nối STT: {str(e)}"}

    if not user_text:
        return {"answer": "Tôi không nghe thấy gì cả.", "audio_base64": None}

    # --- BƯỚC 2: AGENT THINKING (LangGraph) ---
    print(f"   -> Agent đang suy nghĩ...")
    config = {"configurable": {"thread_id": thread_id}}
    
    # 🌟 TRUYỀN NGÔN NGỮ VÀO CHO THE BRAIN DỊCH THUẬT
    result = await rag_agent.ainvoke({
        "question": user_text,
        "language": language
    }, config=config)
    
    intent = result.get("intent_data", {})
    if intent.get("is_unsafe"):
        bot_answer = EMPATHETIC_SAFETY_MESSAGE
    else:
        bot_answer = result.get("answer", "Xin lỗi, tôi gặp lỗi khi xử lý thông tin.")
    
    print(f"   -> Bot trả lời: {bot_answer}")

    # --- BƯỚC 3: TTS (Text to Speech) ---
    print(f"   -> Đang chuyển văn bản sang giọng nói...")
    
    # 🌟 CHỌN GIỌNG ĐỌC THEO NGÔN NGỮ
    voice = "en-US-AriaNeural" if language == "en" else "vi-VN-HoaiMyNeural"
    communicate = edge_tts.Communicate(bot_answer, voice)
    
    audio_stream = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    
    audio_base64 = base64.b64encode(audio_stream.getvalue()).decode('utf-8')

    # --- KẾT QUẢ TRẢ VỀ ---
    return {
        "user_text": user_text,   
        "bot_answer": bot_answer, 
        "audio_base64": audio_base64 
    }