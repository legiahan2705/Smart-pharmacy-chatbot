import os
import json
import gc
from dotenv import load_dotenv
from typing import TypedDict, Literal

# --- FastAPI Imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, HarmBlockThreshold, HarmCategory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
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
import gc # Thư viện dọn dẹp rác bộ nhớ

def load_and_clean_data():
    parquet_path = "data/optimized_db.parquet"
    source_json_path = "data/longchau_selected.json"

    # 1. Load Cache (Ưu tiên)
    if os.path.exists(parquet_path):
        print("⚡ [Pandas] Tìm thấy Cache Parquet. Đang tải...")
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"⚠️ Cache lỗi ({e}), sẽ xử lý lại từ đầu...")

    # 2. Xử lý lần đầu (Tối ưu bộ nhớ)
    print("🐢 [Pandas] Bắt đầu đọc file JSON...")
    try:
        # Bước A: Đọc file
        with open(source_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   -> Đã đọc xong {len(data)} sản phẩm. Đang chuẩn hóa dữ liệu...")

        # Bước B: Xử lý trực tiếp trên biến 'data' (Không tạo bản sao normalized_data)
        for item in data:
            # Lấy danh sách key để tránh lỗi runtime khi dictionary thay đổi size
            keys = list(item.keys()) 
            for key in keys:
                if key.startswith("Thành phần"):
                    item["Thành phần"] = item.pop(key) # Đổi tên key cũ thành mới và xóa key cũ ngay
                elif key.startswith("Công dụng"):
                    item["Công dụng"] = item.pop(key)

        print("   -> Đang chuyển sang DataFrame...")
        
        # Bước C: Tạo DataFrame và XÓA NGAY biến data để giải phóng RAM
        df = pd.DataFrame(data)
        del data # Xóa biến data
        gc.collect() # Ép buộc dọn dẹp bộ nhớ ngay lập tức
        
        print("   -> Đang làm sạch cột giá và điền dữ liệu thiếu...")
        
        # Bước D: Xử lý cột giá (Dùng vectorized operation nhanh hơn apply)
        # Chuyển về string trước để tránh lỗi
        df['Giá bán'] = df['Giá bán'].astype(str)
        # Dùng Regex trích xuất số trực tiếp (nhanh hơn loop)
        df['price_int'] = df['Giá bán'].str.replace(r'[^\d]', '', regex=True)
        df['price_int'] = pd.to_numeric(df['price_int'], errors='coerce').fillna(0).astype(int)

        # Bước E: Điền dữ liệu trống
        cols_to_fill = ['Nhà sản xuất', 'Nước sản xuất', 'Xuất xứ thương hiệu', 'Danh mục', 'Dạng bào chế', 'Quy cách', 'Thành phần', 'Lưu ý', 'Bảo quản', 'Công dụng', 'Đơn vị']
        
        # Chỉ điền những cột thực sự tồn tại trong df
        existing_cols = [c for c in cols_to_fill if c in df.columns]
        df[existing_cols] = df[existing_cols].fillna('')
        
        # Bước F: Chuyển đổi kiểu dữ liệu để lưu Parquet an toàn
        print("   -> Đang lưu Cache Parquet (Bước cuối)...")
        
        # Chuyển tất cả về string (trừ price_int) để tránh lỗi format của Parquet
        for col in df.columns:
            if col != 'price_int':
                df[col] = df[col].astype(str)
        
        # Lưu file
        df.to_parquet(parquet_path, index=False)
        print(f"✅ [Pandas] Xử lý xong và đã lưu Cache vào {parquet_path}.")
        
        return df

    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG KHI XỬ LÝ DATA: {e}")
        # Trả về DataFrame rỗng để server không bị crash hẳn
        return pd.DataFrame()

global_df = load_and_clean_data()

# =======================================================
# 2. MODELS & VECTORSTORE
# =======================================================
# Dùng Flash để nhanh, temperature thấp để chính xác
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE} 
)

print("⏳ Đang tải mô hình Embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ.get("GOOGLE_API_KEY"))

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15, "score_threshold": 0.4})
else:
    print("⚠️ Không có Vectorstore!")

# =======================================================
# 3. CORE LOGIC (TỐI ƯU HÓA: ONE-SHOT BRAIN)
# =======================================================

class AppState(TypedDict):
    question: str
    chat_history: list[str]
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
# Prompt này chứa ĐẦY ĐỦ các ý từ code cũ của bạn, nhưng gộp lại để chạy 1 lần.
brain_prompt_template = """
Bạn là bộ não trung tâm của hệ thống AI y tế Long Châu. Nhiệm vụ của bạn là phân tích câu hỏi và trả về kết quả dưới dạng JSON.

HÃY THỰC HIỆN 3 BƯỚC PHÂN TÍCH SAU:

BƯỚC 1: KIỂM DUYỆT AN TOÀN (SAFETY CHECK)
Phân tích xem câu hỏi có chứa ý định nguy hiểm không dựa trên các tiêu chí:
1. Tự tử, tự hại (Self-harm): Muốn chết, tìm cách kết thúc cuộc sống, ngủ mãi mãi.
2. Đầu độc, Giết người (Violence): Tìm thuốc độc, thuốc không màu không mùi, cách hại người.
3. Sử dụng sai mục đích nghiêm trọng: Dùng thuốc quá liều để "phê", gây mê.
-> Nếu vi phạm: Đặt "is_unsafe": true.

BƯỚC 2: ĐỊNH TUYẾN (ROUTING)
Xác định loại câu hỏi để chọn nguồn dữ liệu:
- Nếu hỏi thông tin mô tả, công dụng, cách dùng, tác dụng phụ, thành phần -> Chọn "vector_search".
- Nếu hỏi GIÁ CẢ (rẻ nhất, đắt nhất), SỐ LƯỢNG (bao nhiêu loại), SO SÁNH giá, hoặc LỌC theo tiêu chí -> Chọn "structured_analysis".
-> Gán giá trị vào trường "route".

BƯỚC 3: MỞ RỘNG CÂU HỎI (QUERY EXPANSION)
Chuyển đổi câu hỏi thành từ khóa tìm kiếm chuyên sâu:
1. Luôn thêm từ khóa "Thuốc", "Điều trị", "Dược phẩm".
2. Nếu mô tả triệu chứng, thêm tên các HOẠT CHẤT (Active Ingredients) phổ biến.
3. Đọc Lịch sử trò chuyện để giải quyết đại từ nhân xưng (nó, thuốc này) nếu cần.
-> Gán kết quả vào trường "keywords".

INPUT DATA:
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
        result = await brain_chain.ainvoke({"question": question, "chat_history": history})
        print(f"Brain Analysis: {result}")
        return {"intent_data": result}
    except Exception as e:
        print(f"Brain Error: {e}. Fallback to vector search.")
        # Fallback an toàn nếu lỗi JSON
        return {"intent_data": {"is_unsafe": False, "route": "vector_search", "keywords": question}}

# --- NODE 2: RETRIEVE ---
async def retrieve_node(state: AppState):
    print("--- 🔍 RETRIEVE ---")
    query = state["intent_data"].get("keywords", state["question"])
    print(f"Searching: {query}")
    docs = await retriever.ainvoke(query)
    
    # Format docs
    context = "\n\n".join([doc.page_content for doc in docs])
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

QUY TẮC QUAN TRỌNG:
1. Khi tìm "Rẻ nhất" (nsmallest), PHẢI loại bỏ giá bằng 0: `df[df['price_int'] > 0]`.
2. Khi tìm theo "Xuất xứ" (Ví dụ: Thuốc Mỹ), hãy tìm trong CẢ 2 CỘT: `Xuất xứ thương hiệu` HOẶC `Nước sản xuất`.
3. Khi tìm theo tên bệnh/triệu chứng (Ví dụ: đau đầu, bổ não), PHẢI tìm trong CẢ 3 CỘT: `Danh mục` HOẶC `Tên thuốc` HOẶC `Công dụng`.
4. Luôn hiển thị cột `Quy cách` trong kết quả.

Ví dụ 1:
Question: Tìm 3 loại thuốc Omega 3 rẻ nhất.
Python: result = df[(df['Danh mục'].str.contains('Omega 3', case=False, na=False)) & (df['price_int'] > 0)].nsmallest(3, 'price_int')[['Tên thuốc', 'Giá bán', 'Quy cách', 'Xuất xứ thương hiệu']].to_string()

Ví dụ 2:
Question: Có bao nhiêu loại thuốc của Mỹ?
Python: result = f"Có {{len(df[(df['Nước sản xuất'].str.contains('Mỹ|Hoa Kỳ|USA', case=False, na=False)) | (df['Xuất xứ thương hiệu'].str.contains('Mỹ|Hoa Kỳ|USA', case=False, na=False))])}} thuốc có xuất xứ hoặc thương hiệu Mỹ."

Ví dụ 3:
Question: Liệt kê các thuốc dạng Siro giá dưới 50000.
Python: result = df[(df['Dạng bào chế'].str.contains('Siro', case=False, na=False)) & (df['price_int'] > 0) & (df['price_int'] < 50000)][['Tên thuốc', 'Giá bán', 'Quy cách']].to_string()

Question: {question}
Python:
"""
pandas_chain = PromptTemplate.from_template(pandas_prompt_template) | llm | StrOutputParser()

async def structured_analysis_node(state: AppState):
    print("--- 🐼 PANDAS ANALYSIS ---")
    question = state["question"]
    code = await pandas_chain.ainvoke({"question": question})
    clean_code = code.replace("```python", "").replace("```", "").strip()
    
    local_vars = {"df": global_df, "result": None}
    try:
        exec(clean_code, {}, local_vars)
        result = local_vars["result"]
        # Convert result to string safely
        if hasattr(result, 'to_string'): final = result.to_string()
        else: final = str(result)
        final_answer = f"Dựa trên số liệu phân tích được:\n{final}"
    except Exception as e:
        final_answer = f"Xin lỗi, tôi gặp lỗi khi tính toán số liệu: {str(e)}"
    
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

Lịch sử hội thoại:
{chat_history}

Context:
{context}

Question: {question}
Answer:
"""
rag_generation_chain = PromptTemplate.from_template(generate_prompt_template) | llm | StrOutputParser()

async def generate_node(state: AppState):
    print("--- ✍️ GENERATE ---")
    question = state["question"]
    context = state.get("context", "")
    history = "\n".join(state.get("chat_history", []))
    
    try:
        answer = await rag_generation_chain.ainvoke({
            "question": question, 
            "context": context, 
            "chat_history": history
        })
    except Exception as e:
        print(f"Error: {e}")
        answer = "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này."
    
    # Cập nhật lịch sử
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

@app.post("/chat")
async def chat_handler(request: ChatRequest):
    print(f"--> User: {request.question}")
    config = {"configurable": {"thread_id": request.thread_id}}
    
    result = await rag_agent.ainvoke({"question": request.question}, config=config)
    
    # Kiểm tra Safety từ kết quả Brain
    intent = result.get("intent_data", {})
    if intent.get("is_unsafe"):
        final_answer = EMPATHETIC_SAFETY_MESSAGE
    else:
        final_answer = result.get("answer", "Lỗi: Không có câu trả lời.")
    
    return {"answer": final_answer}