### Hướng dẫn chạy project

1. **Cài đặt Python 3.11**\

2. **Tạo và kích hoạt môi trường ảo (venv)**\
   Mở terminal/cmd tại thư mục gốc của project và chạy:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Cài Tesseract**\
   - Tải Tesseract từ link git và lưu vào ổ C:
   https://github.com/UB-Mannheim/tesseract/wiki
   - Lấy file "vie.traineddata" trong folder tessdata để trong Tesseract vừa tải về 
   Ví dụ lưu vào ổ C -> "C:\Program Files\Tesseract-OCR\tessdata"
4. **Cài đặt thư viện từ**\

   ```bash
   pip install -r requirements.txt
   ```

5. **Chạy ứng dụng**

   ```bash
   python main.py
   ```
