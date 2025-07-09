import pytesseract
import io
import fitz
from PIL import Image

class UploadData:
    def parse_pdf_with_ocr_images(self, file_obj, lang):
            file_obj.seek(0)
            doc = fitz.open(stream=file_obj.read(), filetype="pdf")
            full_text = []

            for i, page in enumerate(doc):
                page_text = page.get_text()
                full_text.append(f"## Page {i+1}\n")
                full_text.append(page_text.strip())

                # Trích xuất ảnh từ trang
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # OCR ảnh
                    image = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(image, lang=lang).strip()
                    if ocr_text:
                        full_text.append(f"\n**image {img_index+1} on page {i+1}:**\n")
                        full_text.append(ocr_text)

            return "\n\n".join(full_text)