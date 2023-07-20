import customtkinter
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, weaviate, FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import customtkinter as ctk

os.environ["OPENAI_API_KEY"] = "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"
reader = PdfReader("combine.pdf")

raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2500,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

doc_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")


def break_text(text, max_length):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_length:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def answer():
    user_input = entry.get()
    docs = doc_search.similarity_search(user_input)
    ans = chain.run(input_documents=docs, question=user_input)
    ans = break_text(ans, 60)
    output_label.configure(text=ans)


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.geometry("750x750")
root.title("ChatBot")
exmo_label = ctk.CTkLabel(root, text="EXMO 23'", font=ctk.CTkFont(size=30, weight="bold"), text_color="#2C74B3")
exmo_label.pack(padx=10, pady=(40, 20))

welcome_label_frame = ctk.CTkFrame(root, corner_radius=10)
welcome_label_frame.pack(pady=20)

welcome_label = ctk.CTkLabel(welcome_label_frame, text="Welcome!\nDepartment of Town and Country Planning,\nUniversity of Moratuwa", font=ctk.CTkFont(size=20, weight="bold"))
welcome_label.pack(pady=20, padx=20)

ask_label = ctk.CTkLabel(root, text="What do you like to know?", font=ctk.CTkFont(size=20, weight="bold"), text_color="#43919B")
ask_label.pack(padx=10, pady=(40, 20))

entry = ctk.CTkEntry(root, placeholder_text="Type here..", width=520, height=35)
entry.pack()

ask_btn = ctk.CTkButton(root, text="Ask me!", width=520, command=answer)
ask_btn.pack(pady=20)

scrollable_frame = ctk.CTkScrollableFrame(root, width=500, height=200)
scrollable_frame.pack()

output_label = ctk.CTkLabel(scrollable_frame, text="", font=ctk.CTkFont(size=15))
output_label.pack(padx=20)

root.mainloop()