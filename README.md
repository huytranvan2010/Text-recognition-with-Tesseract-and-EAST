# Text-recognition-with-Tesseract + EAST algorithm
Trong bài này chúng ta đi giải quyết bài toán OCR với sự kết hợp của EAST và Tesseract.
* EAST sẽ đảm nhận vai trò text detector
* Tessseract sẽ thực hiện text recognition

Trong project này có file `frozen_east_text_detection.pb` - EAST text detector, đây là pre-trained model CNN cho text detection. 
Tóm tắt một số bước chính như sau:
* Load EAST text detector
* Load ảnh, chuyển kích thước ảnh về số chia hết cho 32
* Chuyển ảnh về blob, đưa qua network để nhận được predictions: geometry và scores
* Thực hiện NMS để nhận được boxes (so với kích thước ảnh đầu vào)
* Rescale lại boxes so với kích thước ảnh gốc, trích xuất text ROI 
* Đưa text ROI qua Tesseract, sau đó in kết quả ra (text ROI được đưa trực tiếp vào Tesseract chưa qua xử lý, các bạn có thể thử pre-processing)

Thử test với một số ảnh trong thư mục `images` nhận thấy nhiều ảnh phải thêm padding vào mới nhận biết được. Có một số chữ không nhận dạng được. Một số nguyên nhân có thể là:
* Chữ bị nghiêng, xoay
* Phông chữ của text không được trained trong Tesseract

Có lẽ chúng ta cần thực hiện [a perspective transform to correct the view](https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/). Nên nhớ rằng trong bài này chúng ta cũng không đưa ra **rotated bounding boxes**. Như các bạn đã thấy bounding boxes khi in ra có dạng nằm ngang. Bản thân nó có góc xoay, ở đây mình chưa tính đến điều này.

Một số lựa chọn thay thế cho Tesseract OCR:
* Google Vision API OCR Engine
* Amazon Rekognition
* Microsoft Cognitive Services

Ngoài ra có thể xem xét thên EasyOCR package. EasyOCR có thể:
* Vừa thực hiện text detection và text recognition (Tesseract cũng có thể làm vậy, ở đây mình sử dụng thêm EAST để crop text ROI)
* EasyOCR có thể hỗ trợ nhiều ngôn ngữ
* Pythonic API (dễ dàng làm việc)
* Sử dụng state-of-the-art model
* Vẫn tiếp tục được phát triển, sớm hỗ trợ chữ viết tay

