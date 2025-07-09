from dotenv import load_dotenv
from loader.load_neo4j import load_data_neo4j
from loader.upload_data import UploadData
from query.index import GraphRetriever
import pytesseract
import os

load_dotenv()

def upload():
    load_dotenv()
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    input_pdf = "test_convert_pdf_image.pdf"
    output_md = "sample_output.md"

    uploader = UploadData()
    with open(input_pdf, "rb") as f:
        ocr_text = uploader.parse_pdf_with_ocr_images(f, lang="vie")

    full_text = "".join(ocr_text)

    date_neo4j = load_data_neo4j(full_text)
    print("Dữ liệu upload thành công:", date_neo4j)

    with open(output_md, "w", encoding="utf-8") as md_f:
        md_f.write(full_text)
    print("Đã ghi nội dung ra:", output_md)


if __name__ == "__main__":  
    # Sử dụng hàm upload() để tạo graph
    # upload()
    retriever = GraphRetriever().run_chain("Ngày 1 xe đón tại đâu,khởi hành đi đâu và ngày 2 khám phá cái gì?")
    print("Response: ", retriever)
    
    