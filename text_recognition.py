# Hướng dẫn chạy
# python text_recognition.py --image images/example_01.jpg
import pytesseract
import argparse
import cv2
from hammiu import decode_prediction
import numpy as np
from imutils.object_detection import non_max_suppression

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the input image")
parser.add_argument("-c", "--min_confidence", type=float, default=0.5, help="min confidence score for bbox to consider")
parser.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
parser.add_argument("-w", "--width", type=int, default=320, help="nearest multiple of 32 for resized width")
parser.add_argument("-e", "--height", type=int, default=320, help="nearest multiple of 32 for resized height")
args = vars(parser.parse_args())

print("[INFO] loading EAST detector...")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# Chuẩn bị ảnh, chuyển về kích thước // 32
image = cv2.imread(args["image"])
# copy một ảnh để xử lý trên đó, tí so sánh với ảnh gốc
orig = image.copy()
# lưu lại kích thước ảnh gốc ban đầu
(orig_h, orig_w) = image.shape[:2]

# Nếu dùng cách này hoặc chọn height, width không thích hợp sẽ cho kết quả không tốt
# Đã thử và xác nhận điều này
# new_h = (h // 32) * 32
# new_w = (w // 32) * 32

new_h = args["height"]
new_w = args["width"]
# chủ yếu để chú ý sẽ làm việc với image mới được resize rồi, điều này thực chất ko cần do bên dưới đã chuyển thành blob có kích thước đó
image = cv2.resize(image, (new_w, new_h))   

# lưu lại ratio để còn rescale lên so với ảnh ban đầu, do ảnh đưa vào mạng phải resize lại cho cạnh chia hết cho 32
ratio_h = orig_h / new_h
ratio_w = orig_w / new_w

# chuyển về 4D blob (batch, channels, H, W), convert BRG => RGB
# substratc mean from image, input của EAST là bội của 32, nếu ko sẽ không nối được ở phần feature-merging branch
blob = cv2.dnn.blobFromImage(image, 1, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
# cho blob qua network
net.setInput(blob)

output_layer_names = net.getUnconnectedOutLayersNames()
(geometry, scores) = net.forward(output_layer_names)

# lấy final boxes sau khi đã loại bỏ box có score nhỏ và áp dụng NMS
final_boxes = decode_prediction(geometry, scores, min_score=args["min_confidence"])     # để overThreshold mặc định nhé

# Tạo list để lưu kết quả
results = []

# Phải nhân với hệ số scale ban đầu mới khớp được với ảnh ban đầu, do kích thước ảnh ban đầu có thể không chia hết cho 32
# nên cần resize để chia hết cho 32
# duyệt qua các boxes

for(xmin, ymin, xmax, ymax) in final_boxes:
    """ Lấy tọa độ của bounding box trên ảnh gốc (do đã nhân với hệ số tỉ lệ). Từ đây chuyển về hết kích thước ảnh gốc"""
    xmin = int(xmin * ratio_w)
    ymin = int(ymin * ratio_h)
    xmax = int(xmax * ratio_w)
    ymax = int(ymax * ratio_h)

    # áp dụng padding vào bounding box nhằm mở rộng bounding box
    # nhiều khi bounding box ăn sâu vào bên trong text, cần mở rộng ra
    """ 
    Ở đây để padding tỉ lệ theo chiều cao và chiều rộng (ví dụ 5%) của bounding box
    Điều này có nghĩa rằng chiều nào lớn hơn sẽ mở rộng hơn.
    Tuy nhiên mình nghĩ ko cần thiết, có thể mở rộng 2 chiều như nhau, truyền vào số pixel cũng được
    Ở đây truyên tỉ lệ
    """
    dx = int((xmax - xmin) * args["padding"])
    dy = int((ymax - ymin) * args["padding"])

    # sau khi áp dụng padding cần dùng min, max để tránh tràn ra khỏi kích thước ảnh
    xmin = max(0, xmin - dx)
    ymin = max(0, ymin - dy)
    xmax = min(orig_w, xmax + 2 * dx)
    ymax = min(orig_h, ymax + 2 * dy)

    # trích xuất text ROI
    roi = orig[ymin:ymax, xmin:xmax]
    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)     # nên nhớ orig vẫn đang ở BGR

    # set config for Tesseract
    config = "-l eng --oem 1 --psm 7"
    text = pytesseract.image_to_string(roi, config=config)

    # lưu bounding box và text tương ứng vào list
    results.append(((xmin, ymin, xmax, ymax), text))

""" Sắp xếp results theo tọa độ của bounding boxes từ trên xuống """
results = sorted(results, key=lambda x: x[0][1])    # sắp xếp theo ymin

# duyệt qua kết quả
for ((xmin, ymin, xmax, ymax), text) in results:
    print("{}\n".format(text))
    
    # strip out non-ASCII text so we can draw the text on the image using OpenCV
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = orig.copy()

    cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.putText(output, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, ), 3)

    cv2.imshow("Output", output)
    cv2.waitKey(0)


