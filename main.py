from dotenv import load_dotenv
from loader.load_neo4j import load_data_neo4j
from loader.upload_data import UploadData
from query.index import GraphRetriever
from langchain_community.document_loaders import PyPDFLoader
import pytesseract
import re

load_dotenv()

def upload():
    load_dotenv()
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    input_pdf = "hanoisapa.pdf"
    output_md = "sample_output.md"

    loader = PyPDFLoader(input_pdf)
    pages = loader.load()
    py_content = "\n".join([page.page_content for page in pages])
    split_words = re.findall(r'\S+', py_content)
    py_text = "\n".join(split_words)


    uploader = UploadData()
    with open(input_pdf, "rb") as f:
        ocr_text = uploader.parse_pdf_with_ocr_images(f, lang="vie")

    full_text = "\n".join([py_text, ocr_text])

    with open(output_md, "w", encoding="utf-8") as md_f:
        md_f.write(full_text)
    print("Đã ghi nội dung ra:", output_md)

    date_neo4j = load_data_neo4j(full_text)
    print("Dữ liệu upload thành công:", date_neo4j)

if __name__ == "__main__":  
    # Sử dụng hàm upload() để tạo graph
    # upload()
    retriever = GraphRetriever().run_chain("Ngày 1 xe đón tại đâu ở và Ngày 2 sẽ đi đâu")
    print("Response: ", retriever)
    
    