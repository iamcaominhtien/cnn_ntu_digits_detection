from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
from module.detect_table import DetectTable


def remove_horizontal_line(IMG: np.ndarray) -> np.ndarray:
    h, w = IMG.shape
    PARAM_ROW_WHITENESS_THRES = 150
    erode_kernel = np.ones((3, 3), np.uint8)
    im = IMG.copy()
    is_eroded = 0
    for _idx in range(3):
        row_whiteness = im.mean(axis=1)
        _row_whiteness_needed = np.where(row_whiteness > PARAM_ROW_WHITENESS_THRES)[0]
        if len(_row_whiteness_needed) == 0:
            break
        _run = 0
        if _idx == 2:
            break
        while _run < len(_row_whiteness_needed) - 1:
            h_line_top_row = h_line_bottom_row = _row_whiteness_needed[_run]
            for _run in range(_run, len(_row_whiteness_needed)):
                if _row_whiteness_needed[_run] - h_line_top_row > 20:
                    break
                else:
                    h_line_bottom_row = _row_whiteness_needed[_run]

            # print(h_line_top_row, h_line_bottom_row)
            h_line_mid_row = int((h_line_bottom_row + h_line_top_row) / 2)
            if h_line_mid_row < 0:
                break

            # if is_eroded == 0:
            im = cv2.morphologyEx(im, cv2.MORPH_ERODE, iterations=2, kernel=erode_kernel)
            is_eroded = is_eroded + 1

            rules = [
                {'top_kernel': 10, 'bot_kernel': 0, 'lateral_kernel': 2, 'threshold': 250, 'punish': 1},
                {'top_kernel': 0, 'bot_kernel': 10, 'lateral_kernel': 2, 'threshold': 250, 'punish': 1},
                {'top_kernel': 0, 'bot_kernel': 5, 'lateral_kernel': 0, 'threshold': 250, 'punish': 1.5},
                {'top_kernel': 5, 'bot_kernel': 0, 'lateral_kernel': 0, 'threshold': 250, 'punish': 1.5},
                {'top_kernel': 2, 'bot_kernel': 2, 'lateral_kernel': 2, 'threshold': 250, 'punish': 1.5}
            ]
            MAX_LATERAL_KERNEL = max([rule['lateral_kernel'] for rule in rules])
            PARAM_VIOLATING_THRESHOLD = 2 if _idx == 0 else 2  # ! MUST BE UPDATED ALONG WITH 'rules'
            PARAM_N_LOCAL_DEFENDERS = 3
            PARAM_DEFENDING_SUCCESS_THRESHOLD = 2
            PARAM_ERASE_KERNEL = {'top': 5, 'bot': 5, 'left': 0, 'right': 0}

            im_tmp = im.copy()
            violate_count_dict = defaultdict(int)
            for i in range(MAX_LATERAL_KERNEL, w - MAX_LATERAL_KERNEL):
                violate_count_dict[i] = 0
                temp = []
                for rule in rules:
                    lat_k = rule['lateral_kernel']
                    top_k = rule['top_kernel']
                    bot_k = rule['bot_kernel']
                    threshold = rule['threshold']
                    punish = rule['punish']
                    window_mean = im_tmp[h_line_mid_row - top_k:h_line_mid_row + bot_k + 1,
                                  i - lat_k:i + lat_k + 1].mean()
                    temp.append(window_mean)
                    if window_mean < threshold:
                        violate_count_dict[i] += punish
                # print(i, temp)

            defender_low_bound = MAX_LATERAL_KERNEL
            defender_up_bound = w - 1 - MAX_LATERAL_KERNEL
            for i, cnt in violate_count_dict.items():
                if cnt >= PARAM_VIOLATING_THRESHOLD:
                    defense_count = 0
                    leftmost_defender = max(defender_low_bound, i - PARAM_N_LOCAL_DEFENDERS)
                    rightmost_defender = min(defender_up_bound, i + PARAM_N_LOCAL_DEFENDERS)
                    for j in range(leftmost_defender, rightmost_defender + 1):
                        if violate_count_dict[j] < PARAM_VIOLATING_THRESHOLD:
                            defense_count += 1
                    if defense_count < PARAM_DEFENDING_SUCCESS_THRESHOLD:
                        # pass
                        im[h_line_mid_row - PARAM_ERASE_KERNEL['top']:h_line_mid_row + PARAM_ERASE_KERNEL['bot'] + 1,
                        i - PARAM_ERASE_KERNEL['left']:i + PARAM_ERASE_KERNEL['right'] + 1] = 0
            im = cv2.morphologyEx(im, cv2.MORPH_DILATE, iterations=2, kernel=erode_kernel)

    # if is_eroded > 0:
    #     for _ in range(is_eroded):
    #         im = cv2.morphologyEx(im, cv2.MORPH_DILATE, iterations=2, kernel=erode_kernel)
    return im


def remove_vertical_line(im: np.ndarray) -> np.ndarray:
    # PARAM_LEFTMOST = 40
    # im[:, :PARAM_LEFTMOST] = 0
    # # return im[:, PARAM_LEFTMOST:]
    # im = ~ im
    _find_vertical_lines = np.where(np.mean(im, axis=0) >= 0.8 * 255)[0]
    _index_left_s = np.where(_find_vertical_lines < 60)[0]
    _index_left = None
    if len(_index_left_s):
        _index_left = _find_vertical_lines[_index_left_s[-1]]

    _index_right_s = np.where(_find_vertical_lines > 300)[0]
    _index_right = None
    if len(_index_right_s):
        _index_right = _find_vertical_lines[_index_right_s[0]]

    _left = 7
    _right = 353
    if _index_left is not None:
        _left = _index_left + 7
    if _index_right is not None:
        _right = _index_right - 7
    im[:, :_left] = 0
    im[:, _right:] = 0
    return im


def remove_background(image: np.ndarray) -> np.ndarray:
    image = ~image  # Make black background
    image = cv2.GaussianBlur(image, (3, 3), 0)
    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image = remove_vertical_line(image)
    image = remove_horizontal_line(image)
    return image


def noise_filter(IMAGE, show_result=False):
    """"
        Remove noise in image
        Args:
            IMAGE: image
            show_result: show result or not
        Return:
            image after remove noise
    """
    matrix = IMAGE.copy()
    image = IMAGE.copy()
    h_img, w_img = IMAGE.shape[:2]
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    # DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    _dx = []
    _dy = []

    for idx_x in range(h_img):
        for idx_y in range(w_img):
            if matrix[idx_x, idx_y] == 0:
                continue

            # Khởi tạo hàng đợi và đánh dấu điểm bắt đầu
            queue = [(idx_x, idx_y)]
            matrix[idx_x][idx_y] = 0
            path_x, path_y = [idx_x], [idx_y]

            # Duyệt hàng đợi
            while queue:
                # Lấy phần tử đầu tiên trong hàng đợi
                x, y = queue.pop(0)

                # Duyệt các điểm lân cận (bao gồm cả đường chéo)
                for dx, dy in DIRECTIONS:
                    nx, ny = x + dx, y + dy

                    # Kiểm tra điểm có nằm trong ma trận và có giá trị != 0 hay không
                    # if h_img > nx >= 0 != matrix[nx][ny] and 0 <= ny < w_img:
                    if 0 <= nx < h_img and 0 <= ny < w_img and matrix[nx][ny] != 0:
                        # Đánh dấu điểm và thêm vào hàng đợi
                        matrix[nx][ny] = 0
                        queue.append((nx, ny))

                        # Lưu vết vị trí mới vào mảng
                        path_x.append(nx)
                        path_y.append(ny)

            if len(path_x) < 30:
                image[path_x, path_y] = 0
            else:
                _dx.append(path_x)
                _dy.append(path_y)

    # zoom image
    space = 2
    rows, cols = np.nonzero(image)
    if len(rows) == 0:
        rows = np.array([0, h_img])
    if len(cols) == 0:
        cols = np.array([0, w_img])
    y_min = max(np.min(rows) - space, 0)
    y_max = min(np.max(rows) + space, h_img)
    x_min = max(np.min(cols) - space, 0)
    x_max = min(np.max(cols) + space, w_img)

    image = image[y_min:y_max, x_min:x_max]

    if show_result:
        plt.imshow(image)

    return image


def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


# def it_is_no_point_symbol(IMG, debug=False):
#     _image = IMG[50:-50, 50:-50]
#     try:
#         if np.mean(_image) < 10:
#             return True
#         # _image = cv2.resize(_image, (361, 361), interpolation=cv2.INTER_CUBIC)
#         # _, _thresh = cv2.threshold(_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         #
#         # _N = _thresh.shape[0] - (50 - 1)
#         # _number_of_each_triangle = _N * (_N - 1) // 2
#         #
#         # _upper_triangle = np.triu(_thresh[:, ::-1], k=50)
#         # _upper_mean = np.sum(_upper_triangle) / _number_of_each_triangle
#         #
#         # _lower_triangle = np.tril(_thresh[:, ::-1], k=-50)
#         # _lower_mean = np.sum(_lower_triangle) / _number_of_each_triangle
#         #
#         # if debug:
#         #     print(_upper_mean, _lower_mean)
#         #
#         # if _upper_mean < 25 and _lower_mean < 25:
#         #     return True
#         # else:
#         #     if _upper_mean < 6 or _lower_mean < 6:
#         #         return True
#
#         gray_img = cv2.resize(IMG, (200, 200), interpolation=cv2.INTER_CUBIC)
#         _, gray_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
#         gray_img = rotate_image(gray_img, -40)
#         if len(np.where(np.mean(gray_img, axis=1) > 245)[0]) > 0:
#             gray_shrink = shrink_image(gray_img)
#             h = gray_shrink.shape[0]
#             if h < 90:
#                 return True
#     except:
#         print(_image.shape)
#
#     return False

def it_is_no_point_symbol(IMG, debug=False):
    try:
        _gray_img = cv2.resize(IMG, (200, 200), interpolation=cv2.INTER_CUBIC)
        _, _gray_img = cv2.threshold(_gray_img, 127, 255, cv2.THRESH_BINARY)
        _gray_img = rotate_image(_gray_img, -40)
        _average = np.mean(_gray_img, axis=1)
        if max(_average) < 10:
            return True
        if len(np.where(_average > 245)[0]) > 0:
            _gray_shrink = shrink_image(_gray_img)
            h = _gray_shrink.shape[0]
            if h < 90:
                return True
    except Exception as _:
        pass
    return False


def find_numbers(data):
    try:
        H = 220
        W = 361
        _thresh = data.copy()
        _thresh = cv2.resize(_thresh, (W, H), interpolation=cv2.INTER_CUBIC)
        _thresh = cv2.dilate(_thresh, np.ones((3, 3), np.uint8), iterations=8)
        _, _thresh = cv2.threshold(_thresh, 127, 255, cv2.THRESH_BINARY_INV)
        _thresh = 255 - _thresh
        _ctrs, _ = cv2.findContours(_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _sorted_ctrs = sorted(_ctrs, key=lambda _ctr: cv2.boundingRect(_ctr)[0])
        _thresh = cv2.erode(_thresh, np.ones((3, 3), np.uint8), iterations=8)
        number_of_numbers = 0

        def _process_result(crs):
            original_height, original_width = data.shape
            scale_x = original_width / W
            scale_y = original_height / H
            # print("scale_x: ", scale_x)
            # print("scale_y: ", scale_y)

            for _idx in range(len(crs)):
                # print("before: ", crs)
                _x_min, _x_max, _y_min, _y_max = crs[_idx]
                _x_min, _x_max, _y_min, _y_max = int(_x_min * scale_x), int(_x_max * scale_x), int(
                    _y_min * scale_y), int(_y_max * scale_y)
                crs[_idx] = [_x_min, _x_max, _y_min, _y_max]
                # print("after: ", crs)
            return crs

        def _find_num_process_for_1(_sorted_ctrs):
            if len(_sorted_ctrs) == 0:
                if it_is_no_point_symbol(_thresh):
                    return []

            x, y, w, h = _sorted_ctrs[0]
            _x_min, _y_min = x, y
            _x_max, _y_max = x + w, y + h

            if it_is_no_point_symbol(_thresh[_y_min: _y_max, _x_min: _x_max]):
                return []

            _middle = (_x_min + _x_max) // 2
            _pos = np.argmin(_thresh.mean(axis=0)[_middle - 20: _middle + 20])
            crs = [[_x_min, _middle - 20 + _pos, _y_min, _y_max], [_middle - 20 + _pos, _x_max, _y_min, _y_max]]
            return _process_result(crs)

        def _find_num_process_for_2(_sorted_ctrs):
            # global _sorted_ctrs
            x1, y1, w1, h1 = _sorted_ctrs[0]
            x2, y2, w2, h2 = _sorted_ctrs[1]
            if w1 > 0.8 * W and h1 > 0.4 * H:
                _sorted_ctrs = [_sorted_ctrs[0]]
                return _find_num_process_for_1(_sorted_ctrs)
            if w2 > 0.8 * W and h2 > 0.4 * H:
                _sorted_ctrs = [_sorted_ctrs[1]]
                return _find_num_process_for_1(_sorted_ctrs)
                # Kiểm tra chồng lấn
            if x1 < x2 < x1 + w1 < x2 + w2:
                _sorted_ctrs = [[x1, y1, x2 - x1, h1], [x2, y2, w2, h2]]

            crs = [[x, x + w, y, y + h] for x, y, w, h in _sorted_ctrs]
            return _process_result(crs)

        if len(_sorted_ctrs) == 1:
            _sorted_ctrs = [cv2.boundingRect(_ctr) for _ctr in _sorted_ctrs]
            return _find_num_process_for_1(_sorted_ctrs)
        elif len(_sorted_ctrs) == 2:
            _sorted_ctrs = [cv2.boundingRect(_ctr) for _ctr in _sorted_ctrs]
            return _find_num_process_for_2(_sorted_ctrs)
        elif len(_sorted_ctrs) > 2:
            _sorted_ctrs = sorted(_sorted_ctrs, key=lambda _ctr: cv2.boundingRect(_ctr)[0])[:3]
            _ctr_first = _sorted_ctrs[0]
            x1, y1, w1, h1 = cv2.boundingRect(_ctr_first)
            _ctr_second = _sorted_ctrs[1]
            x2, y2, w2, h2 = cv2.boundingRect(_ctr_second)
            _ctr_third = _sorted_ctrs[2]
            x3, y3, w3, h3 = cv2.boundingRect(_ctr_third)

            # TH1: 3 khung có chiều cao gần bằng nhau
            _min_h = min(h1, h2, h3)
            if abs(h1 - _min_h) < 20 and abs(h2 - _min_h) < 20 and abs(h3 - _min_h) < 20:
                # Nếu có khung nào có chiều rộng < 20 -> loại bỏ
                if w3 < 30:
                    _sorted_ctrs = [cv2.boundingRect(_ctr_first), cv2.boundingRect(_ctr_second)]
                    return _find_num_process_for_2(_sorted_ctrs)
                if w2 < 30:
                    _sorted_ctrs = [cv2.boundingRect(_ctr_first), cv2.boundingRect(_ctr_third)]
                    return _find_num_process_for_2(_sorted_ctrs)
                if w1 < 30:
                    _sorted_ctrs = [cv2.boundingRect(_ctr_second), cv2.boundingRect(_ctr_third)]
                    return _find_num_process_for_2(_sorted_ctrs)

                # print("TH1: 3 khung có chiều cao gần bằng nhau")
                for _ctr in _sorted_ctrs:
                    x, y, w, h = cv2.boundingRect(_ctr)
                    cv2.rectangle(_thresh, (x, y), (x + w, y + h), (255, 255, 255), 5)
                crs = [[x, x + w, y, y + h] for x, y, w, h in _sorted_ctrs]
                return _process_result(crs)
            # TH2: 2 khung sau, 1 trên 1 dưới và tổng chiều cao ~ bằng chiều cao khung trên cộng trừ 20 -> gom 2 khung sau lại làm 1
            if ((y2 < y3 and y2 + h2 < y3 + h3) or (y2 > y3 and y2 + h2 > y3 + h3)) and abs(h1 - (h2 + h3)) < 20:
                # print("TH2: 2 khung sau, 1 trên 1 dưới và tổng chiều cao ~ bằng chiều cao khung trên cộng trừ 20 -> gom 2 khung sau lại làm 1")
                _x2_min, _x2_max, _y2_min, _y2_max = x2, x2 + w2, y2, y2 + h2
                _x3_min, _x3_max, _y3_min, _y3_max = x3, x3 + w3, y3, y3 + h3
                _x_min, _x_max = min(_x2_min, _x3_min), max(_x2_max, _x3_max)
                _y_min, _y_max = min(_y2_min, _y3_min), max(_y2_max, _y3_max)
                _merger_2_vs_3 = [_x_min, _y_min, _x_max - _x_min, _y_max - _y_min]
                _sorted_ctrs = [cv2.boundingRect(_ctr_first), np.array(_merger_2_vs_3)]
                return _find_num_process_for_2(_sorted_ctrs)

            # TH3: lấy ra 2 khung có diện tích phù hợp nhất hoặc lớn nhất
            cords = []
            for _idx, _ctr in enumerate(_sorted_ctrs):
                # Get bounding box
                x, y, w, h = cv2.boundingRect(_ctr)
                if h > 70 and w > 75:
                    cords.append(_idx)
                    number_of_numbers += 1
            if len(cords) == 2:
                _sorted_ctrs = [cv2.boundingRect(_sorted_ctrs[cords[0]]), cv2.boundingRect(_sorted_ctrs[cords[1]])]
                # print("TH3: lấy ra 2 khung có diện tích phù hợp nhất")
                return _find_num_process_for_2(_sorted_ctrs)

            # print("TH3: lấy ra 2 khung có diện tích lớn nhất")
            _sorted_ctrs = [cv2.boundingRect(_ctr) for _ctr in _sorted_ctrs]
            areas = [w * h for _, _, w, h in _sorted_ctrs]
            _min_area_index = np.argmin(areas)
            _x_min, _y_min, _w_min, _h_min = _sorted_ctrs[_min_area_index]
            _thresh[_y_min:_y_min + _h_min, _x_min:_x_min + _w_min] = 0
            _sorted_ctrs = [_item for _idx, _item in enumerate(_sorted_ctrs) if _idx != _min_area_index]
            return _find_num_process_for_2(_sorted_ctrs)

        # cv2.drawContours(thresh, sorted_ctrs, -1, (255, 255, 255), 2)
        #     _thresh = cv2.resize(_thresh, (data.shape[1], data.shape[0]), interpolation=cv2.INTER_CUBIC)
        crs = [[x, x + w, y, y + h] for x, y, w, h in _sorted_ctrs]
        return _process_result(crs)
    except:
        return []


def shrink_image(data):
    rows = np.any(data, axis=1)
    cols = np.any(data, axis=0)

    first_row = np.argmax(rows)
    last_row = data.shape[0] - np.argmax(np.flip(rows))
    first_col = np.argmax(cols)
    last_col = data.shape[1] - np.argmax(np.flip(cols))

    return data[first_row:last_row, first_col:last_col]


def extend_image(image, space=3):
    """
    This function extends an image by adding a border of zeros around it.

    Args:
      image (numpy.ndarray): The image to be extended.

    Returns:
      extended_image (numpy.ndarray): The extended image.

    """
    IMAGE = image.copy()
    h, w = IMAGE.shape[:2]

    if h > w:
        IMAGE = cv2.copyMakeBorder(IMAGE, space, space, 0, 0, cv2.BORDER_CONSTANT, value=0)
        subtract = h - w
        left = right = subtract // 2
        if left < 3:
            left = right = 3

        IMAGE = cv2.copyMakeBorder(IMAGE, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
    elif h < w:
        IMAGE = cv2.copyMakeBorder(IMAGE, 0, 0, space, space, cv2.BORDER_CONSTANT, value=0)
        subtract = w - h
        top = bottom = subtract // 2
        if top < 3:
            top = bottom = 3

        IMAGE = cv2.copyMakeBorder(IMAGE, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        IMAGE = cv2.copyMakeBorder(IMAGE, space, space, space, space, cv2.BORDER_CONSTANT, value=0)

    return IMAGE if (IMAGE is not None) else image


def get_nearest_contour(contour, nearest_contour, test=False):
    # return 0: loại bỏ
    # return 1: gộp 2 contour lại làm 1
    # return 2: giữ nguyên, nearest_contour là 1 số mới
    x, y, w, h = contour
    x1, y1, w1, h1 = nearest_contour

    # nếu nearest_contour nằm trong contour, trả về 0
    if x <= x1 and y <= y1 and x + w >= x1 + w1 and y + h >= y1 + h1:
        if test: print('nearest_contour nằm trong contour')
        return 0
    # nếu nearest_contour nằm bên phải
    if x1 > x + w:
        # nếu khoảng cách lớn hơn 20 -> 0
        if x1 - (x + w) > 20:
            return 0
        # nếu ở trên bên phải
        if y1 < y:
            # nếu khoảng cách lớn hơn 20 -> 0
            if y - y1 > 20:
                return 0
            # nếu chiều cao nhỏ hơn 35 và chiều rộng lớn hơn 50 -> 1
            if h1 < 35 and w1 > 50:
                if test: print(
                    'nearest_contour nằm ở trên bên phải, chiều cao nhỏ hơn 35 và chiều rộng lớn hơn 50, return 1')
                return 1
            return 0
        # nếu ở dưới bên phải -> 0
        if y1 > y + h:
            return 0
        # còn lại: bên phải, không trên cũng không dưới
        # nếu chiều cao > 70 và chiều rộng > 20 -> 2, đây là một số
        if h1 > 70 and w1 > 20:
            if test: print('nearest_contour nằm ở bên phải, chiều cao > 70 và chiều rộng > 20, return 2')
            return 2
        return 0
    # tương tự cho bên trái
    if x1 + w1 < x:
        # nếu khoảng cách lớn hơn 20 -> 0
        if x - (x1 + w1) > 20:
            return 0
        # nếu ở trên bên trái
        if y1 < y:
            return 0
        # nếu ở dưới bên trái
        if y1 > y + h:
            return 0
        if x - (x1 + w1) > 20:
            return 0
        if h1 > 70 and w1 > 20:
            if test: print('nearest_contour nằm ở bên trái, chiều cao > 70 và chiều rộng > 20, return 2')
            return 2
        return 0
    # nếu nearest_contour nằm bên trên
    if y1 < y:
        # nếu khoảng cách lớn hơn 20 -> 0
        if y - (y1 + h1) > 20:
            return 0
        # nếu ở trên bên trái
        if x1 < x:
            return 0
        # nếu ở trên bên phải
        if x1 > x + w:
            # nếu cách bên phải quá 20 -> 0
            if x1 - (x + w) > 20:
                return 0
            # nếu chiều cao nhỏ hơn 35 và chiều rộng lớn hơn 70 -> 1
            if h1 < 35 and w1 > 70:
                if test: print(
                    'nearest_contour nằm ở trên bên phải, chiều cao nhỏ hơn 35 và chiều rộng lớn hơn 70, return 1')
                return 1
            if w1 > 50 and y - (y1 + h1) < 10 and x1 - (x + w) < 10 and x1 - x > 25:
                return 1
            return 0
        # còn lại: bên trên, không trái cũng không phải
        if w1 > 50 and y - (y1 + h1) < 10 and x1 - (x + w) < 10 and x1 - x > 25:
            if test: print('nearest_contour nằm ở trên, chiều cao nhỏ hơn 35 và chiều rộng lớn hơn 50, return 1')
            return 1
        # if w1 > 50 and y - (y1 + h1) < 10 and x1 - (x + w) < 10 and x1 - x > 25:
        #     return 1
        return 0
    # nếu nearest_contour nằm bên dưới
    return 0


def process_contour_len_one(img, contour):
    # only_one = True: chỉ có duy nhất 1 contour được phát hiện trong toàn bộ ảnh
    x, y, w, h = contour[0]
    _frame = img[y: y + h, x: x + w]
    try:
        if it_is_no_point_symbol(_frame):
            return []
    except:
        return contour
    if w / h > 1.2 and w / img.shape[1] > 0.55:
        # split into 2 contours
        return [(x, y, w // 2, h), (x + w // 2, y, w - w // 2, h)]
    return contour


def merge_two_contour_into_one(c1, c2):
    x1, y1, w1, h1 = c1
    x2, y2, w2, h2 = c2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)


def filter_contours(img, _ctrs_cord):
    '''
    Hàm lọc các contour không cần thiết, lọc dựa trên chiều cao của contour so với contour lớn nhất, và vị trí của contour so với contour lớn nhất
    :param _ctrs_cord:
    :return:
    '''
    # Loại bỏ các contour cóc chiều cao quá lớn (>90%)
    for _ctr in _ctrs_cord:
        if _ctr[3] > img.shape[0] * 0.9 and _ctr[2] < 50:
            _ctrs_cord.remove(_ctr)

    if len(_ctrs_cord) == 1:
        return _ctrs_cord
    if len(_ctrs_cord) == 2:
        # get the biggest contour (w*h)
        _ctrs_cord_temp = sorted(_ctrs_cord, key=lambda _ctr: _ctr[2] * _ctr[3])
        _biggest_contour = _ctrs_cord_temp[-1]
        _contour_maybe_removed = _ctrs_cord_temp[0]
        # Nếu chiều cao < 50% chiều cao của contour lớn nhất thì xóa contour đó
        if _contour_maybe_removed[3] < _biggest_contour[3] * 0.5:
            _ctrs_cord.remove(_contour_maybe_removed)
        # Nếu y_max (y+h) của contour nhỏ nhất < 40% y_max của contour lớn nhất thì xóa contour đó
        elif _contour_maybe_removed[1] + _contour_maybe_removed[3] < (_biggest_contour[1] + _biggest_contour[3]) * 0.5:
            _ctrs_cord.remove(_contour_maybe_removed)
    elif len(_ctrs_cord) > 2:
        # get the biggest contour (w*h)
        _ctrs_cord_temp = sorted(_ctrs_cord, key=lambda _ctr: _ctr[2] * _ctr[3])
        _biggest_contour = _ctrs_cord_temp[-1]
        _contour_maybe_removed = _ctrs_cord_temp[0:-1]

        for _ctr in _contour_maybe_removed:
            # Nếu chiều cao < 50% chiều cao của contour lớn nhất thì xóa contour đó
            if _ctr[3] < _biggest_contour[3] * 0.5:
                _ctrs_cord.remove(_ctr)
            # Nếu y_max (y+h) của contour đang xét < 50% y_max của contour lớn nhất thì xóa contour đó
            # elif _ctr[1] + _ctr[3] < (_biggest_contour[1] + _biggest_contour[3]) * 0.5:
            #     _ctrs_cord.remove(_ctr)
            elif _ctr[1] + _ctr[3] < (_biggest_contour[1] + _biggest_contour[3] * 0.6):
                _ctrs_cord.remove(_ctr)

    # Trường hợp 2 contour, kiểm tra xem 2 contour có cách nhau quá xa
    if len(_ctrs_cord) == 2:
        _first_contour = _ctrs_cord[0]
        _xfc, _yfc, _wfc, _hfc = _first_contour
        _last_contour = _ctrs_cord[1]
        _xlc, _ylc, _wlc, _hlc = _last_contour
        if _xlc - (_xfc + _wfc) > 100:
            _ctrs_cord.remove(_last_contour)
        elif _xfc - (_xlc + _wlc) > 100:
            _ctrs_cord.remove(_first_contour)

    # Trong trường hợp có 3 contour, contour ở giữa rất có khả năng là contour của dấu phẩy
    if len(_ctrs_cord) == 3:
        _maybe_comma = _ctrs_cord[1]
        _xmc, _ymc, _wmc, _hmc = _maybe_comma
        _first_contour = _ctrs_cord[0]
        _xfc, _yfc, _wfc, _hfc = _first_contour
        _last_contour = _ctrs_cord[2]
        _xlc, _ylc, _wlc, _hlc = _last_contour
        if _ymc > _yfc + _hfc * 0.25 or _ymc > _ylc + _hlc * 0.25:
            _ctrs_cord.remove(_maybe_comma)
        elif (_hmc < _hfc * 0.7 and _wmc < 40) or (_hmc < _hlc * 0.7 and _wmc < 40):
            _ctrs_cord.remove(_maybe_comma)

    _ctrs_cord = sorted(_ctrs_cord, key=lambda _ctr: _ctr[0])
    if len(_ctrs_cord) > 2:
        for _i, _ctr_i in enumerate(_ctrs_cord):
            if _i == 0: continue
            _x1, _y1, _w1, _h1 = _ctrs_cord[_i - 1]
            _x2, _y2, _w2, _h2 = _ctr_i
            if _x2 - (_x1 + _w1) < 15:
                # merge 2 contour
                _merge_temp = merge_two_contour_into_one(_ctrs_cord[_i - 1], _ctr_i)
                # if w/h < 1.2: can merge
                if _merge_temp[2] / _merge_temp[3] < 1.2:
                    _ctrs_cord[_i - 1] = _merge_temp
                    _ctrs_cord.remove(_ctr_i)
                    break
    return _ctrs_cord


def cell_456_process(IMG):
    _img = IMG.copy()
    if _img.ndim > 2:
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('raw', _img)
    _img_resized = cv2.resize(_img, (361, 220), interpolation=cv2.INTER_CUBIC)
    # _img_resized = ~_img_resized  # Make black background
    _img_resized = cv2.GaussianBlur(_img_resized, (3, 3), 0)
    _, _thresh = cv2.threshold(_img_resized, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _result = remove_background(_thresh)
    _vertical_length = int(_result.shape[0] / 40)
    _vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _vertical_length))
    _vertical_detect = cv2.morphologyEx(_result, cv2.MORPH_OPEN, _vertical_kernel, iterations=2)
    _blurred_image = cv2.GaussianBlur(_vertical_detect, (3, 3), 0)
    _median_filter = cv2.medianBlur(_blurred_image, 15)
    _gray_img = _median_filter.copy()

    _ctrs, _ = cv2.findContours(_gray_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _sorted_ctrs = sorted(_ctrs, key=lambda _ctr: cv2.boundingRect(_ctr)[0])
    _ctrs_cord = [cv2.boundingRect(_ctr) for _ctr in _sorted_ctrs]
    if len(_sorted_ctrs) == 1:
        for _idx in range(len(_ctrs_cord) - 1, -1, -1):
            if _ctrs_cord[_idx][2] < 30 or _ctrs_cord[_idx][3] < 50:
                _ctrs_cord.remove(_ctrs_cord[_idx])
        _number_maybe = [_ctr for _ctr in _ctrs_cord if _ctr[3] > 70 and _ctr[2] > 30]
        _ctrs_cord.clear()
        _ctrs_cord = _number_maybe.copy()
    elif len(_sorted_ctrs) == 2:
        # continue
        for _idx in range(len(_ctrs_cord) - 1, -1, -1):
            if _ctrs_cord[_idx][2] < 30 or _ctrs_cord[_idx][3] < 50:
                _ctrs_cord.remove(_ctrs_cord[_idx])
        _ctrs_cord = sorted(_ctrs_cord, key=lambda _ctr: _ctr[3])
        _number_maybe = [_ctr for _ctr in _ctrs_cord if _ctr[3] > 70 and _ctr[2] > 30]

        _number_maybe = sorted(_number_maybe, key=lambda _ctr: _ctr[0])
        _ctrs_cord.clear()
        _ctrs_cord = _number_maybe.copy()
    elif len(_sorted_ctrs) > 2:
        # continue
        for _idx in range(len(_ctrs_cord) - 1, -1, -1):
            if _ctrs_cord[_idx][2] < 30 or _ctrs_cord[_idx][3] < 50:
                _ctrs_cord.remove(_ctrs_cord[_idx])
        _ctrs_cord = sorted(_ctrs_cord, key=lambda _ctr: _ctr[3])
        _number_maybe = []
        for _ctr in _ctrs_cord:
            if _ctr[3] > 70 and _ctr[2] > 30:
                _number_maybe.append(_ctr)

        # xóa các khung trong _number_maybe khỏi _ctrs_cord
        for _ctr in _number_maybe:
            if _ctr in _ctrs_cord:
                _ctrs_cord.remove(_ctr)

        # duyệt qua các khung còn lại, kiểm tra có khung nào có khả năng làm 1 số mới hoặc gộp được hay không
        _newest_maybe_numbers = []
        for _ctr in _ctrs_cord:
            for _idx, _maybe in enumerate(_number_maybe):
                _check = get_nearest_contour(_maybe, _ctr)
                if _check == 1:
                    _number_maybe[_idx] = merge_two_contour_into_one(_maybe, _ctr)
                elif _check == 2:
                    _newest_maybe_numbers.append(_ctr)
        _number_maybe.extend(_newest_maybe_numbers)

        # sắp xếp lại _number_maybe theo thứ tự x tăng dần
        _number_maybe = sorted(_number_maybe, key=lambda _ctr: _ctr[0])
        _ctrs_cord.clear()
        _ctrs_cord = _number_maybe.copy()

    if len(_ctrs_cord) > 1:
        _newest_maybe_numbers = []
        for _ctr in _ctrs_cord:
            _temp = process_contour_len_one(_gray_img, [_ctr])
            if len(_temp) > 1:
                _newest_maybe_numbers.extend(_temp)
            else:
                _newest_maybe_numbers.append(_ctr)
        _newest_maybe_numbers = sorted(_newest_maybe_numbers, key=lambda _ctr: _ctr[0])
        _ctrs_cord = _newest_maybe_numbers.copy()

    if len(_ctrs_cord) == 1:
        _ctrs_cord = process_contour_len_one(_gray_img, _ctrs_cord)

    # print(_ctrs_cord)
    _ctrs_cord = set(_ctrs_cord)
    _ctrs_cord = sorted(_ctrs_cord, key=lambda _ctr: _ctr[0])
    _ctrs_cord = filter_contours(_gray_img, _ctrs_cord)

    _numbers = []
    for _idx, _ctr in enumerate(_ctrs_cord):
        x, y, w, h = _ctr
        frame = extend_image(_gray_img[y:y + h, x:x + w], max(w // 8, 3))

        # find contour on frame
        _ctrs, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _sum_frame = np.sum(frame)
        for _ctr_temp in _ctrs:
            x, y, w, h = cv2.boundingRect(_ctr_temp)
            _frame_temp = frame[y:y + h, x:x + w]
            _sum_frame_temp = np.sum(_frame_temp)
            if _sum_frame_temp / _sum_frame < 0.1:
                frame[y:y + h, x:x + w] = 0

        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        frame = cv2.resize(frame, (28, 28))
        _numbers.append(frame)

    return _numbers
