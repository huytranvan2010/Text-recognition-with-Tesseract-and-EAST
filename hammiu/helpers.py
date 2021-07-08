from imutils.object_detection import non_max_suppression
import numpy as np

def decode_prediction(geometry, scores, min_score):
    """
        Trả về list of bounding boxes

        geometry - geometry prediction từ EAST text detector

        scores - scores prediction từ EAST text detector
    """
    # tạo lists chứa rects và confidences
    rects = []
    confidence_scores = []
    # duyệt qua từng pixels của feature maps (có thể nói vậy do đều dùng Conv layer `1x1` nên width và height ko đổi, chỉ số channels thay đổi)
    for i in range(geometry.shape[2]):      # duyệt theo height - row
        for j in range(geometry.shape[3]):  # duyệt theo width - column
            if scores[0][0][i][j] < min_score:    # score của pixel, shape đầu tiên của scores là batch_size (ở đây có 1 ảnh lấy [0]), cái sau lưu score cũng chỉ có 1 giá trị lấy [0]
                continue
            # ở đây phải nhân thêm 4 cho tất cả tọa độ vì output size chỉ bằng 1/4 input size, nhìn vào kiến trúc mạng sẽ thấy
            # mình đã kiểm tra new_h, new_w và future map hay shape của scrores và geometry rồi
            top_x = int(4*j - geometry[0][3][i][j])
            top_y = int(4*i - geometry[0][0][i][j])

            bottom_x = int(4*j + geometry[0][1][i][j])
            bottom_y = int(4*i + geometry[0][2][i][j])

            """ Cách bên dưới tính đến góc nhưng rõ ràng không triệt để, ở đây text nằm ngang góc gần = 0, ko có sự khác biệt"""
            # angle = geometry[0][4][i][j]
            # sin = np.sin(angle)
            # cos = np.cos(angle)

            # h = geometry[0][0][i][j] + geometry[0][2][i][j]
            # w = geometry[0][1][i][j] + geometry[0][3][i][j]

            # bottom_x = int(4*j + cos * geometry[0][1][i][j] + sin * geometry[0][2][i][j])
            # bottom_y = int(4*i - sin * geometry[0][1][i][j] + cos * geometry[0][2][i][j])

            # top_x = int(bottom_x - w)
            # top_y = int(bottom_y - h)

            rects.append((top_x, top_y, bottom_x, bottom_y))

            # Mỗi box sẽ tương ứng với confidence 
            confidence_scores.append(float(scores[0][0][i][j]))

    # Thực hiện NMS để loại bỏ bớt các box chồng chập 
    final_boxes = non_max_suppression(np.array(rects), probs=confidence_scores)

    return final_boxes