import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Literal

# --- 1. Imports cho FastAPI ---
from fastapi import FastAPI
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
# --- Imports cho phần Memory ---
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
# =======================================================
# 0. KHỞI TẠO VÀ CÀI ĐẶT
# =======================================================

# Tải biến môi trường (GOOGLE_API_KEY)
load_dotenv()

# Khởi tạo app FastAPI
app = FastAPI()

# =======================================================
# 1. PHẦN LANGGRAPH VÀ RAG AGENT
# =======================================================

# 1a. Định nghĩa State 
class AppState(TypedDict):
    question: str
    chat_history: list[str]
    expanded_question: str | None
    context: str | None
    answer: str | None
    contextRelevance: Literal["yes", "no"] | None

# 1b. Khởi tạo LLM, Embeddings, và Retriever
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
print("Đang tải mô hình Google Embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", # Hoặc "models/embedding-001"
    google_api_key=os.environ.get("GOOGLE_API_KEY") # Đảm bảo đã load_dotenv()
)
print("Tải mô hình Google Embeddings thành công!")

# --- 2. HÀM TẢI HOẶC TẠO VECTORSTORE TỪ JSON ---
# Không cần thiết, vì nếu chạy hàm này trên laptop sẽ mất tgian rất lâu (nhưng cứ để đi)
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

        print(f"Đã xử lý {len(docs)} tài liệu với đầy đủ 18 trường thông tin.")
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
        "k": 20,             # Tăng từ 4 lên 20
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
Chỉ sử dụng thông tin được cung cấp trong phần "Context" để trả lời câu hỏi.
Đồng thời, bạn cần dựa vào lịch sử hội thoại để hiểu rõ hơn về ngữ cảnh.
Tuyệt đối không tự bịa ra thông tin.

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
# 1e. TẠO CHAIN MỚI ĐỂ "TRÍCH XUẤT TỪ KHÓA" (THAY VÌ MỞ RỘNG)
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

# NODE MỚI ĐỂ CHẠY VIỆC MỞ RỘNG
async def query_expansion_node(state: AppState):
    print("--- NODE: QUERY EXPANSION ---")
    question = state["question"] # Lấy câu hỏi GỐC
    
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

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 1f. Định nghĩa các Nodes
async def retrieve_node(state: AppState):
    print("--- NODE: RETRIEVE ---")
    # Lấy câu hỏi MỚI từ State
    question_to_search = state["expanded_question"]
    
    print(f"Searching database for: {question_to_search}") # In ra để debug
    relevant_docs = await retriever.ainvoke(question_to_search)
    
    print("\n--- DEBUG: TÀI LIỆU TRUY XUẤT ĐƯỢC ---")
    if not relevant_docs:
        print("!!!!!!!!!!! KHÔNG TÌM THẤY TÀI LIỆU NÀO. !!!!!!!!!!!")
    else:
        for i, doc in enumerate(relevant_docs):
            # print(f"--- Tài liệu {i+1} ---")
            # print(doc.page_content[:500] + "...") 
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

async def generate_node(state: AppState):
    print("--- NODE: GENERATE ---")
    
    current_history = state.get("chat_history", [])
    print(f"LOG: Độ dài lịch sử hiện tại: {len(current_history)} tin nhắn.")
    
    question = state["question"]
    context = state["context"]
    
    history_list = state.get("chat_history", [])
    history_str = "\n".join(history_list) if history_list else "Chưa có lịch sử."
    
    answer = await rag_generation_chain.ainvoke({
        "question": question,
        "context": context,
        "chat_history": history_str
    })
    
    new_history_entry = [
        f"User: {question}",
        f"AI: {answer}"
    ]
    updated_history = history_list + new_history_entry
    
    print("Answer generated.")
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

# 1g. Xây dựng Graph (Agent)
def build_rag_agent():
    workflow = StateGraph(AppState)

    workflow.add_node("expand_query", query_expansion_node) 
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("gradeDocuments", llm_grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("handleIrrelevant", handle_irrelevant_node)

    # Đặt điểm bắt đầu MỚI
    workflow.set_entry_point("expand_query") 
    
    # Đấu nối luồng mới
    workflow.add_edge("expand_query", "retrieve") # 1. Mở rộng câu hỏi
    workflow.add_edge("retrieve", "gradeDocuments") # 2. Tìm kiếm

    # Cạnh có điều kiện (giữ nguyên)
    workflow.add_conditional_edges(
        "gradeDocuments",
        lambda state: state["contextRelevance"],
        {
            "yes": "generate",
            "no": "handleIrrelevant",
        }
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("handleIrrelevant", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    print("Đã xây dựng RAG Agent (Graph) VỚI QUERY EXPANSION. Backend sẵn sàng!")
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