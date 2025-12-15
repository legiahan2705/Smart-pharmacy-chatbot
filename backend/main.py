import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Literal

# --- 1. Imports cho FastAPI ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 

# --- 2. Imports cho LangChain / LangGraph ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import HarmBlockThreshold, HarmCategory
# --- Imports cho phần Memory ---
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import re
# =======================================================
# 0. KHỞI TẠO VÀ CÀI ĐẶT
# =======================================================

# Tải biến môi trường (GOOGLE_API_KEY)
load_dotenv()

# Khởi tạo app FastAPI
app = FastAPI()

# --- CẤU HÌNH CORS ---
# Cho phép Frontend (localhost:3000) gọi vào Backend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Cho phép các nguồn trên
    allow_credentials=True,
    allow_methods=["*"],         # Cho phép tất cả các method (GET, POST...)
    allow_headers=["*"],         # Cho phép tất cả header
)

# =======================================================
# 1. PHẦN LANGGRAPH VÀ RAG AGENT
# =======================================================

# --- ROUTER CHAIN ---
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vector_search", "structured_analysis"]

# Prompt để LLM chọn đường đi
router_system_prompt = """
Bạn là chuyên gia định tuyến câu hỏi.
- Nếu người dùng hỏi về thông tin mô tả, công dụng, cách dùng, tác dụng phụ, thành phần -> Chọn 'vector_search'.
- Nếu người dùng hỏi về GIÁ CẢ (rẻ nhất, đắt nhất, bao nhiêu tiền), SỐ LƯỢNG (có bao nhiêu loại), SO SÁNH giá, 
hoặc LỌC theo tiêu chí (nước sản xuất, dạng bào chế) -> Chọn 'structured_analysis'.
"""
router_prompt = PromptTemplate.from_template(
    f"{router_system_prompt}\nQuestion: {{question}}"
)

# Sử dụng structured output (function calling) để đảm bảo kết quả chuẩn
llm_router = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
structured_llm_router = llm_router.with_structured_output(RouteQuery)
router_chain = router_prompt | structured_llm_router

# I. Sử dụng khi người dùng hỏi về GIÁ CẢ, SỐ LƯỢNG, SO SÁNH
# --- HÀM TẢI VÀ LÀM SẠCH DỮ LIỆU CHO PANDAS ---
def load_and_clean_data():
    source_json_path = "data/longchau_selected.json"
    try:
        with open(source_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1. Chuẩn hóa Key trước khi tạo DataFrame
        # Vì key "Thành phần..." thay đổi theo tên thuốc, ta cần đưa về tên chuẩn
        normalized_data = []
        for item in data:
            new_item = item.copy()
            for key in item.keys():
                # Chuẩn hóa cột Thành phần
                if key.startswith("Thành phần"):
                    new_item["Thành phần"] = item[key]
                # Chuẩn hóa cột Công dụng
                elif key.startswith("Công dụng"):
                    new_item["Công dụng"] = item[key]
            normalized_data.append(new_item)

        df = pd.DataFrame(normalized_data)
        
        # 2. Làm sạch cột Giá bán (Tạo cột price_int)
        def clean_price(price_str):
            if not price_str: return 0
            # Giữ lại số, loại bỏ 'đ', '.', ',', 'Hộp'...
            digits = re.sub(r'[^\d]', '', str(price_str))
            if digits:
                return int(digits)
            return 0 # Giá là "Liên hệ" hoặc rỗng -> coi là 0

        df['price_int'] = df['Giá bán'].apply(clean_price)
        
        # 3. Điền giá trị trống cho các cột quan trọng để tránh lỗi code Python
        columns_to_fill = ['Nước sản xuất', 'Xuất xứ thương hiệu', 'Danh mục', 'Dạng bào chế', 'Quy cách', 'Thành phần', 'Công dụng']
        for col in columns_to_fill:
             if col in df.columns:
                 df[col] = df[col].fillna('')
             else:
                 df[col] = '' # Tạo cột rỗng nếu data thiếu hẳn trường này
        
        print(f"Đã tải DataFrame thành công! Số lượng sản phẩm: {len(df)}")
        return df

    except Exception as e:
        print(f"Lỗi tải dữ liệu Pandas: {e}")
        return pd.DataFrame()

# Khởi tạo DataFrame toàn cục
global_df = load_and_clean_data()

# II. Sử dụng khi người dùng hỏi về THÔNG TIN THUỐC
# 1a. Định nghĩa State 
class AppState(TypedDict):
    question: str
    chat_history: list[str]
    expanded_question: str | None
    context: str | None
    answer: str | None
    contextRelevance: Literal["yes", "no"] | None
    is_unsafe: bool | None # Giữ lại, sẽ được reset qua mỗi lần chạy

# 1b. Khởi tạo LLM, Embeddings, và Retriever
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Lưu ý: Dùng tên model chuẩn 1.5
    temperature=0.3,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, # Chặn ngay cả nguy cơ thấp
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }
)
print("Đang tải mô hình Google Embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", # Hoặc "models/embedding-001"
    google_api_key=os.environ.get("GOOGLE_API_KEY") # Đảm bảo đã load_dotenv()
)
print("Tải mô hình Google Embeddings thành công!")

# --- 2. HÀM TẢI HOẶC TẠO VECTORSTORE TỪ JSON ---
def load_or_create_vectorstore(embeddings):
    """
    Hàm này sẽ tải index FAISS nếu đã tồn tại,
    hoặc tạo mới từ file JSON nếu chưa có.
    """
    source_json_path = "data/longchau_selected.json"
    index_path = "faiss_index" 

    # Nếu index đã tồn tại, tải nó lên cho nhanh
    if os.path.exists(index_path):
        print(f"Đang tải index FAISS đã có")
        try:
            vectorstore = FAISS.load_local(
                index_path, 
                embeddings,
                allow_dangerous_deserialization=True 
            )
            print("Tải index thành công!")
            return vectorstore
        except Exception as e:
            print(f"Lỗi khi tải index: {e}. Sẽ tạo lại từ đầu.")

    print(f"Không tìm thấy index. Đang tạo mới từ {source_json_path}...")
    
    # Các key trong JSON mà chúng ta muốn bỏ qua
    keys_to_ignore = [
        "URL",
        "Ảnh sản phẩm",
        "Số đăng ký",
    ]
    
    docs = [] # Mảng chứa các Documents (1 doc / 1 thuốc)
    
    try:
        with open(source_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        print(f"Đã đọc {len(json_data)} sản phẩm từ JSON.")

        # Lặp qua từng sản phẩm trong file JSON
        for item in json_data:
            # 1. Lấy Tên thuốc làm tiêu đề chính
            ten_thuoc = item.get("Tên thuốc", "Sản phẩm không tên")
            
            # Metadata giữ nguyên để lọc nếu cần
            metadata = {
                "source": source_json_path,
                "ten_thuoc": ten_thuoc,
                "url": item.get("URL"),
            }
            
            # 2. Xây dựng nội dung (page_content) ĐỘNG
            content_pieces = []
            
            # LUÔN LUÔN đưa Tên thuốc lên dòng đầu tiên (Tốt cho tìm kiếm)
            content_pieces.append(f"Tên thuốc: {ten_thuoc}")
            
            # Vòng lặp "Vét cạn" tất cả các trường còn lại
            for key, value in item.items():
                # Bỏ qua các key không cần thiết hoặc giá trị rỗng
                if key in keys_to_ignore or not value:
                    continue
                
                # --- LOGIC LÀM SẠCH KEY ---
                # Ví dụ: Biến "Công dụng của Panadol" -> thành "Công dụng"
                clean_key = key
                if isinstance(value, str):
                    if f"của {ten_thuoc}" in key:
                        clean_key = key.replace(f"của {ten_thuoc}", "").strip()
                    elif f"của thuốc {ten_thuoc}" in key:
                        clean_key = key.replace(f"của thuốc {ten_thuoc}", "").strip()
                    
                # Thêm vào danh sách nội dung
                content_pieces.append(f"{clean_key}: {value}")
                
            # 3. Nối lại thành một đoạn văn bản hoàn chỉnh
            page_content = "\n".join(content_pieces)
            
            # Tạo Document
            docs.append(Document(page_content=page_content, metadata=metadata))

        print(f"Đã xử lý {len(docs)} tài liệu.")

        # --- Chia nhỏ tài liệu (Text Splitting) ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, # GIẢM XUỐNG CÒN 512 HOẶC 400
            chunk_overlap=100, # GIẢM THEO TỶ LỆ
            separators=["\n\n", "\n", " ", ""] 
        )
        
        split_docs = text_splitter.split_documents(docs)
        print(f"Đã chia tài liệu thành {len(split_docs)} mảnh (chunks).")
        
        # --- Tạo VectorStore (FAISS) ---
        print("Đang tạo index FAISS (có thể mất vài phút)...")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # --- Lưu index lại cho lần sau ---
        vectorstore.save_local(index_path)
        print(f"Đã tạo và lưu index mới vào: {index_path}")
        
        return vectorstore

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {source_json_path}.")
        print("Hãy chắc chắn bạn đã đặt file JSON đúng chỗ và tạo thư mục 'data'.")
        return None
    except Exception as e:
        print(f"Lỗi khi xử lý JSON: {e}")
        return None
    
# Gọi hàm mới để tải hoặc tạo vectorstore
vectorstore = load_or_create_vectorstore(embeddings)

# Kiểm tra xem vectorstore có được tạo thành công không
if vectorstore:
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
        "k": 20,          # Tăng từ 4 lên 20
        "score_threshold": 0.3 # Chỉ lấy những thứ có độ liên quan nhất định
        }
    ) 
    print("Retriever đã sẵn sàng!")
else:
    print("LỖI NGHIÊM TRỌNG: Không thể khởi tạo VectorStore. Server sẽ thoát.")
    exit() # Thoát server nếu không có vectorstore
    

# 1c. Chuẩn bị RAG Chain (phần generate)
prompt_template = """
Bạn là một trợ lý tư vấn thuốc thông minh của Long Châu.

NGUYÊN TẮC AN TOÀN TUYỆT ĐỐI (SAFETY GUARDRAILS):
1. TỰ TỬ & LÀM HẠI BẢN THÂN: Nếu người dùng hỏi về liều lượng gây chết người, cách tự tử, uống bao nhiêu thì chết, hoặc cách sử dụng thuốc để gây hại -> BẠN PHẢI TỪ CHỐI TRẢ LỜI.
    - Câu trả lời bắt buộc: "Tôi không thể cung cấp thông tin này. Nếu bạn hoặc ai đó đang gặp nguy hiểm, vui lòng gọi ngay cấp cứu 115 hoặc đến cơ sở y tế gần nhất."
2. QUÁ LIỀU/UỐNG NHẦM (Overdose): 
    - Nếu người dùng lỡ uống quá liều, hãy dựa vào Context để xem thuốc đó có độc tính cao không.
    - Luôn bắt đầu bằng câu: "⚠️ Lưu ý: Thông tin chỉ mang tính tham khảo. Bạn nên liên hệ ngay với bác sĩ hoặc dược sĩ để được đánh giá chính xác."
    - Sau đó cung cấp thông tin về triệu chứng quá liều và cách xử trí (nếu có trong Context).
3. KHÔNG THAY THẾ BÁC SĨ: Với các triệu chứng nghiêm trọng (khó thở, đau ngực dữ dội, nôn ra máu...), hãy khuyên người dùng đi khám bác sĩ ngay lập tức thay vì chỉ gợi ý thuốc.
4. KHÔNG BỊA ĐẶT: Chỉ trả lời dựa trên Context và Lịch sử. Nếu không có thông tin -> Trả lời không biết một cách lịch sự.
5. Đồng thời, bạn cần dựa vào lịch sử hội thoại để hiểu rõ hơn về ngữ cảnh.

Lịch sử hội thoại:
{chat_history}

Context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)
rag_generation_chain = prompt | llm | StrOutputParser()

# 1d. TẠO MỘT CHAIN MỚI ĐỂ "GRADE" 
# Chain này sẽ dùng LLM để quyết định context có liên quan không
grade_prompt_template = """
Bạn là một người kiểm duyệt tài liệu. Nhiệm vụ của bạn là kiểm tra xem các tài liệu (Context)
có chứa thông tin liên quan để trả lời câu hỏi (Question) hay không.
Chỉ trả lời "yes" (nếu liên quan) hoặc "no" (nếu không liên quan).

QUAN TRỌNG: Chỉ cần Context chứa *một phần* thông tin liên quan đến Question
(ví dụ: Question hỏi về "Paracetamol" và Context nói về "Paracetamol và Tramadol"),
bạn VẪN PHẢI trả lời "yes".

Context:
{context}

Question:
{question}

Answer (yes/no):
"""
grade_prompt = PromptTemplate.from_template(grade_prompt_template)
grading_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_grader_chain = grade_prompt | grading_llm | StrOutputParser()

# 1e. TẠO CHAIN MỚI ĐỂ "MỞ RỘNG CÂU HỎI" 
expansion_prompt_template = """
Bạn là một trợ lý dược sĩ chuyên nghiệp. Nhiệm vụ của bạn là chuyển đổi câu hỏi triệu chứng của người dùng 
thành các từ khóa tìm kiếm chuyên sâu để tra cứu trong cơ sở dữ liệu.

Lịch sử trò chuyện (Context):
{chat_history}

Câu hỏi hiện tại: {question}

QUAN TRỌNG:
1. Luôn thêm từ khóa "Thuốc", "Điều trị", "Dược phẩm" để tránh tìm ra dụng cụ y tế.
2. Nếu người dùng mô tả triệu chứng, hãy thêm tên các HOẠT CHẤT (Active Ingredients) phổ biến điều trị triệu chứng đó.
3. Đọc kỹ "Lịch sử trò chuyện" để hiểu người dùng đang nói về loại thuốc nào (nếu họ dùng từ "nó", "thuốc này", "thuốc đó").
4. Kết hợp tên thuốc từ lịch sử với câu hỏi hiện tại để tạo từ khóa tìm kiếm.
5. Nếu câu hỏi hiện tại là chủ đề mới hoàn toàn, hãy bỏ qua lịch sử.

Ví dụ 1:
History: [User: Panadol công dụng gì?, AI: Giảm đau hạ sốt...]
Question: Thuốc này giá bao nhiêu?
Search Query: Giá bán Panadol, Panadol giá bao nhiêu, mua thuốc Panadol

Ví dụ 2:
Question: Con tôi bị rôm sảy.
Search Query: Thuốc bôi rôm sảy, kem trị ngứa, viêm da, an toàn cho trẻ em

Search Query:
"""
expansion_prompt = PromptTemplate.from_template(expansion_prompt_template)
expansion_chain = expansion_prompt | grading_llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# =======================================================
# CÁC NODES ĐÃ SỬA LỖI
# =======================================================

# --- NODE MỚI: SAFETY CHECK ---
async def safety_check_node(state: AppState):
    print("--- NODE: SAFETY CHECK ---")
    question = state["question"]
    
    # Danh sách các từ khóa cấm kỵ (Đã được chuyển từ query_expansion_node sang đây)
    danger_keywords = [
        "tự tử", "muốn chết", "tự sát", "uống bao nhiêu thì chết", "liều chết", "cách chết", "tự vẫn", "quyên sinh", "chết người", "đầu đo", "cắt cổ", "uống thuốc độc", "uống thuốc ngủ để chết", "tự làm hại bản thân"
    ]
    
    # Nếu câu hỏi chứa từ khóa nguy hiểm
    if any(keyword in question.lower() for keyword in danger_keywords):
        print("!!! SAFETY ALERT: Phát hiện ý định nguy hiểm !!!")
        answer = "Tôi là AI tư vấn thuốc, tôi không thể cung cấp thông tin về liều gây hại hoặc tự tử. Vui lòng gọi ngay 115 hoặc đến cơ sở y tế gần nhất để được hỗ trợ."
        
        # Trả về cờ True và câu trả lời. Route Decision sẽ chuyển thẳng đến END.
        return {
            "is_unsafe": True, 
            "answer": answer,
            "expanded_question": None # Xóa bỏ để không ảnh hưởng lượt sau nếu có lỗi
        }
    
    # TRẢ VỀ: Nếu an toàn, đặt cờ False. Route Decision sẽ tiếp tục định tuyến.
    return {"is_unsafe": False}


# NODE MỚI ĐỂ CHẠY VIỆC MỞ RỘNG (ĐÃ SỬA: Loại bỏ Safety Guardrail)
async def query_expansion_node(state: AppState):
    print("--- NODE: QUERY EXPANSION ---")
    question = state["question"]
    
    history_list = state.get("chat_history", [])
    print(f"DEBUG HISTORY (Last 2 turns): {history_list[-2:] if history_list else 'Empty'}")
    history_str = "\n".join(history_list) if history_list else "Chưa có lịch sử."
    
    expanded_question_raw = await expansion_chain.ainvoke({
        "question": question,
        "chat_history": history_str 
    })     
    
    # Xóa bỏ tiền tố "Search Query:" mà LLM có thể thêm vào
    if "Search Query:" in expanded_question_raw:
        expanded_question = expanded_question_raw.split("Search Query:", 1)[-1].strip()
    else:
        expanded_question = expanded_question_raw.strip()

    print(f"Original question: {question}")
    print(f"Expanded search query (Cleaned): {expanded_question}")
    
    return {"expanded_question": expanded_question} 


async def retrieve_node(state: AppState):
    print("--- NODE: RETRIEVE ---")
    # Lấy câu hỏi MỚI từ State
    question_to_search = state["expanded_question"]
    
    # --- XỬ LÝ TRƯỜNG HỢP expanded_question BỊ NONE KHI CHẠY LỖI ---
    if not question_to_search:
        print("Bỏ qua tìm kiếm do câu hỏi mở rộng rỗng.")
        return {"context": ""} # Context rỗng sẽ dẫn đến handleIrrelevant

    print(f"Searching database for: {question_to_search}") # In ra để debug
    relevant_docs = await retriever.ainvoke(question_to_search)
    
    print("\n--- DEBUG: TÀI LIỆU TRUY XUẤT ĐƯỢC ---")
    if not relevant_docs:
        print("!!!!!!!!!!! KHÔNG TÌM THẤY TÀI LIỆU NÀO. !!!!!!!!!!!")
    else:
        for i, doc in enumerate(relevant_docs):
            print('Nội dung tài liệu đã được comment')
    print("----------------------------------\n")

    context = format_docs(relevant_docs)
    print("Context retrieved.")
    return {"context": context}

async def llm_grade_documents_node(state: AppState):
    print("--- NODE: LLM GRADE DOCUMENTS ---")
    question = state["question"]
    context = state.get("context")
    
    # Xử lý trường hợp context rỗng (để tránh gọi LLM vô ích)
    if not context or context.strip() == "":
        print("Grading: Context is EMPTY. Returning 'no'.")
        return {"contextRelevance": "no"}

    # Gọi LLM Grader
    grade_result = await llm_grader_chain.ainvoke({
        "question": question,
        "context": context
    })
    
    # Chuẩn hóa kết quả (loại bỏ khoảng trắng, viết thường)
    grade = grade_result.strip().lower()
    
    if "yes" in grade:
        print("Grading: Context IS relevant (LLM approved).")
        return {"contextRelevance": "yes"}
    else:
        print("Grading: Context is NOT relevant (LLM rejected).")
        return {"contextRelevance": "no"}

# NODE GENERATE (ĐÃ SỬA: Xử lý lỗi Gemini chặn và không cập nhật lịch sử)
async def generate_node(state: AppState):
    print("--- NODE: GENERATE ---")
    
    current_history = state.get("chat_history", [])
    print(f"LOG: Độ dài lịch sử hiện tại: {len(current_history)} tin nhắn.")
    
    question = state["question"]
    context = state.get("context", "")
    answer = "" # Khởi tạo biến answer

    # 2. Cố gắng sinh câu trả lời từ LLM
    try:
        # Chuẩn bị input cho chain
        history_list = state.get("chat_history", [])
        history_str = "\n".join(history_list) if history_list else "Chưa có lịch sử."
        
        # Gọi LLM (Có thể bị Google chặn tại đây nếu nội dung nguy hiểm)
        answer = await rag_generation_chain.ainvoke({
            "question": question,
            "context": context,
            "chat_history": history_str
        })
        
    except Exception as e:
        # 3. BẮT LỖI KHI GEMINI CHẶN (SAFETY BLOCK)
        print(f"!!! LỖI SINH CÂU TRẢ LỜI (Có thể do Safety Filter): {e}")
        # Gán giá trị mặc định cho answer
        answer = "Xin lỗi, tôi không thể trả lời câu hỏi này do vi phạm tiêu chuẩn an toàn về y tế và sức khỏe của Google. Vui lòng tham khảo ý kiến bác sĩ trực tiếp."
        
        # Trả về câu trả lời đã gán. Lịch sử KHÔNG được cập nhật tại đây.
        return {"answer": answer}

    # 4. Cập nhật lịch sử và trả về kết quả CHỈ KHI SINH CÂU TRẢ LỜI THÀNH CÔNG
    new_history_entry = [
        f"User: {question}",
        f"AI: {answer}"
    ]
    updated_history = current_history + new_history_entry
    
    print("Answer generated")
    return {"answer": answer, "chat_history": updated_history}


async def handle_irrelevant_node(state: AppState):
    print("--- NODE: HANDLE IRRELEVANT ---")
    answer = "Xin lỗi. Tôi không tìm thấy thông tin về loại thuốc này trong cơ sở dữ liệu."
    
    # --- Cập nhật lịch sử dù không tìm thấy ---
    history_list = state.get("chat_history", [])
    updated_history = history_list + [
        f"User: {state['question']}",
        f"AI: {answer}"
    ]
    return {"answer": answer, "chat_history": updated_history}

# --- NODE STRUCTURED ANALYSIS (Giữ nguyên logic cập nhật lịch sử) ---
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

Ví dụ 2 (SỬA LỖI TẠI ĐÂY: Dùng {{ và }} ):
Question: Có bao nhiêu loại thuốc của Mỹ?
Python: result = f"Có {{len(df[(df['Nước sản xuất'].str.contains('Mỹ|Hoa Kỳ|USA', case=False, na=False)) | (df['Xuất xứ thương hiệu'].str.contains('Mỹ|Hoa Kỳ|USA', case=False, na=False))])}} thuốc có xuất xứ hoặc thương hiệu Mỹ."

Ví dụ 3:
Question: Liệt kê các thuốc dạng Siro giá dưới 50000.
Python: result = df[(df['Dạng bào chế'].str.contains('Siro', case=False, na=False)) & (df['price_int'] > 0) & (df['price_int'] < 50000)][['Tên thuốc', 'Giá bán', 'Quy cách']].to_string()

Question: {question}
Python:
"""

pandas_prompt = PromptTemplate.from_template(pandas_prompt_template)
pandas_chain = pandas_prompt | llm | StrOutputParser()

async def structured_analysis_node(state: AppState):
    print("--- NODE: STRUCTURED ANALYSIS (PANDAS) ---")
    question = state["question"]
    
    # 1. Nhờ LLM sinh code Python
    generated_code = await pandas_chain.ainvoke({"question": question})
    
    # Làm sạch code (bỏ markdown ```python ... ```)
    clean_code = generated_code.replace("```python", "").replace("```", "").strip()
    print(f"Generated Python Code: {clean_code}")
    
    # 2. Thực thi code (EXEC) - Lưu ý: Chỉ dùng cho demo/nội bộ
    # Cần biến result để hứng kết quả
    local_vars = {"df": global_df, "result": None}
    
    try:
        exec(clean_code, {"df": global_df, "pd": pd, "re": re}, local_vars) # Thêm pd và re vào exec
        result = local_vars["result"]
        
        # Nếu result là DataFrame hoặc Series, chuyển thành string
        if isinstance(result, (pd.DataFrame, pd.Series)):
            final_answer = result.to_string()
        else:
            final_answer = str(result)
            
        # Thêm lời dẫn cho tự nhiên
        final_answer = f"Dựa trên phân tích số liệu:\n{final_answer}"
        
    except Exception as e:
        final_answer = f"Xin lỗi, tôi gặp lỗi khi tính toán số liệu: {str(e)}. Vui lòng thử lại với câu hỏi rõ ràng hơn."
        print(f"Pandas Error: {e}")

    # Cập nhật lịch sử (quan trọng để giữ mạch hội thoại)
    history_list = state.get("chat_history", [])
    updated_history = history_list + [
        f"User: {question}",
        f"AI: {final_answer}"
    ]
    
    return {"answer": final_answer, "chat_history": updated_history}


# 1g. Xây dựng Graph (Agent) (ĐÃ SỬA: Thay đổi cấu trúc và logic định tuyến)
def build_rag_agent():
    workflow = StateGraph(AppState)

    # 1. THÊM NODE SAFETY CHECK (MỚI)
    workflow.add_node("safety_check", safety_check_node)
    
    # 2. Các node khác
    workflow.add_node("expand_query", query_expansion_node) 
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("gradeDocuments", llm_grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("handleIrrelevant", handle_irrelevant_node)
    workflow.add_node("structured_analysis", structured_analysis_node)

    # 3. Đặt điểm bắt đầu MỚI
    workflow.set_entry_point("safety_check") 
    
    # --- LOGIC ROUTING ĐÃ SỬA ---
    def route_decision(state):
        # 1. KIỂM TRA AN TOÀN TRƯỚC TIÊN (Từ safety_check node)
        if state.get("is_unsafe"):
            print("--- ROUTING: UNSAFE DETECTED -> END ---")
            # Nếu nguy hiểm, kết thúc luồng. Câu trả lời đã có trong state.answer.
            return END
        
        # 2. SỬ DỤNG ROUTER CHAIN ĐỂ CHỌN ĐƯỜNG ĐI (Cho câu hỏi an toàn)
        question = state["question"]
        print("--- ROUTING ---")
        decision = router_chain.invoke({"question": question})
        print(f"Destination: {decision.datasource}")
        
        if decision.datasource == "structured_analysis":
            return "structured_analysis"
        else:
            # vector_search/mô tả -> chuyển sang mở rộng câu hỏi
            return "expand_query" 

    # Cạnh có điều kiện đầu tiên (Sau Safety Check)
    workflow.add_conditional_edges(
        "safety_check",
        route_decision,
        {
            "structured_analysis": "structured_analysis", # Chuyển đến Pandas
            "expand_query": "expand_query",               # Chuyển đến RAG
            END: END                                      # Chuyển đến kết thúc (nếu is_unsafe=True)
        }
    )
    
    # Đấu nối luồng RAG
    workflow.add_edge("expand_query", "retrieve") # 1. Mở rộng câu hỏi -> Tìm kiếm
    workflow.add_edge("retrieve", "gradeDocuments") # 2. Tìm kiếm -> Đánh giá Context

    # Cạnh có điều kiện (giữ nguyên)
    workflow.add_conditional_edges(
        "gradeDocuments",
        lambda state: state["contextRelevance"],
        {
            "yes": "generate",
            "no": "handleIrrelevant",
        }
    )

    # Các cạnh kết thúc
    workflow.add_edge("generate", END)
    workflow.add_edge("handleIrrelevant", END)
    workflow.add_edge("structured_analysis", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    print("Đã xây dựng RAG Agent (Graph) VỚI SAFETY NODE ĐỘC LẬP. Backend sẵn sàng!")
    return app

# Biên dịch agent khi server khởi động
rag_agent = build_rag_agent()

# =======================================================
# 2. PHẦN API (FASTAPI)
# =======================================================

# 1. Định nghĩa kiểu dữ liệu cho request body (thay thế cho zod.object)
class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default_user"

# 2. Tạo endpoint /chat (thay thế cho app.post('/chat', ...))
@app.post("/chat")
async def chat_handler(request: ChatRequest):
    print(f"Đang xử lý câu hỏi: {request.question}")
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Chạy agent với input
    # Agent sẽ chạy từ safety_check, nếu bị chặn sẽ trả về câu trả lời an toàn ngay
    result = await rag_agent.ainvoke(
        {"question": request.question}, 
        config=config
    )
    
    # Trả về câu trả lời
    final_answer = result.get("answer", "Lỗi: Không có câu trả lời.")
    print(f"Câu trả lời từ AI: {final_answer}")
    
    return {"answer": final_answer}

# 3. Endpoint "Hello World" để kiểm tra server
@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với RAG Agent! Gửi request POST đến /chat"}