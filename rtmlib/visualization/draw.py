import math

import cv2
import numpy as np

from .skeleton import *  # noqa
import random

def draw_bbox(img, bboxes, color=(0, 255, 0)):
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), color, 2)
    return img


def draw_skeleton(img,
                  keypoints,
                  scores,
                  openpose_skeleton=False,
                  kpt_thr=0.5,
                  radius=2,
                  line_width=2):
    num_keypoints = keypoints.shape[1]

    if openpose_skeleton:
        if num_keypoints == 18:
            skeleton = 'openpose18'
        elif num_keypoints == 134:
            skeleton = 'openpose134'
        else:
            raise NotImplementedError
    else:
        if num_keypoints == 17:
            skeleton = 'coco17'
        elif num_keypoints == 133:
            skeleton = 'coco133'
        elif num_keypoints == 21:
            skeleton = 'hand21'
        else:
            raise NotImplementedError

    skeleton_dict = eval(f'{skeleton}')
    keypoint_info = skeleton_dict['keypoint_info']
    skeleton_info = skeleton_dict['skeleton_info']

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]


    num_instance = keypoints.shape[0]
    if skeleton in ['coco17', 'coco133', 'hand21']:
        for i in range(num_instance):
            img = draw_mmpose(img, keypoints[i], scores[i], keypoint_info,
                              skeleton_info, kpt_thr, radius, line_width)
    elif skeleton in ['openpose18', 'openpose134']:
        for i in range(num_instance):
            img = draw_openpose(img,
                                keypoints[i],
                                scores[i],
                                keypoint_info,
                                skeleton_info,
                                kpt_thr,
                                radius * 2,
                                alpha=0.6,
                                line_width=line_width * 2)
    else:
        raise NotImplementedError
    return img


def merge_image(images):
    widths = [img.shape[1] for img in images]
    max_height = max(img.shape[0] for img in images)

    # 计算拼接图像的总宽度
    total_width = sum(widths)

    # 创建一个足够容纳所有图像的空白图像
    stitched_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # 水平拼接所有图像
    current_width = 0
    for img in images:
        # 计算图像的宽度和高度
        height, width, _ = img.shape

        # 将图像复制到拼接图像的适当位置
        stitched_image[0:height, current_width:current_width+width] = img

        # 更新当前宽度位置
        current_width += width
    return stitched_image

def draw_skeleton_fixColor(img,
                  all_color,
                  skel_width,
                  skel_height,
                  keypoints,
                  scores,
                  openpose_skeleton=False,
                  kpt_thr=0.5,
                  radius=2,
                  line_width=2):
    num_keypoints = keypoints.shape[1]

    if openpose_skeleton:
        if num_keypoints == 18:
            skeleton = 'openpose18'
        elif num_keypoints == 134:
            skeleton = 'openpose134'
        else:
            raise NotImplementedError
    else:
        if num_keypoints == 17:
            skeleton = 'coco17'
        elif num_keypoints == 133:
            skeleton = 'coco133'
        elif num_keypoints == 21:
            skeleton = 'hand21'
        else:
            raise NotImplementedError

    skeleton_dict = eval(f'{skeleton}')
    keypoint_info = skeleton_dict['keypoint_info']
    skeleton_info = skeleton_dict['skeleton_info']

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]
    num_instance = keypoints.shape[0]

    min_x_list = []
    origin_id = []
    for id in range(num_instance):
        vis_kpt = [s >= kpt_thr for s in scores[id]]
        # x_values = []
        # for item in range(len(keypoints[id])):
            # if vis_kpt[item]:
            #     x_values.append(keypoints[id][item][0])
            # else:
            #     x_values.append(0)
        x_values = [x for x, _ in keypoints[id]]
        min_x = max(x_values)
        min_x_list.append(min_x)
        origin_id.append(id)

    if len(min_x_list):
        sorted_pair = sorted(zip(min_x_list, origin_id), key=lambda pair: pair[0])
        sorted_data, sorted_ids = zip(*sorted_pair)

    new_img = np.zeros((img.shape[0] + skel_height, img.shape[1], img.shape[2]), dtype=np.uint8)
    if skeleton in ['coco17', 'coco133', 'hand21']:
        vec_skel_img = []
        for i in range(4):
            skel_img = np.zeros((skel_height, skel_width, 3), dtype=np.uint8)
            if i < num_instance and len(min_x_list):
                img = draw_mmpose(img, keypoints[sorted_ids[i]], scores[sorted_ids[i]], keypoint_info,
                              skeleton_info, kpt_thr, radius, line_width, all_color[i])
                skel_img = draw_skel_only(skel_img, img.shape, keypoints[sorted_ids[i]], scores[sorted_ids[i]], keypoint_info,
                              skeleton_info, kpt_thr, radius, line_width, all_color[i])
            vec_skel_img.append(skel_img)

        all_skel_img = merge_image(vec_skel_img)
        cv2.vconcat([img, all_skel_img], new_img)

    elif skeleton in ['openpose18', 'openpose134']:
        for i in range(num_instance):
            img = draw_openpose(img,
                                keypoints[i],
                                scores[i],
                                keypoint_info,
                                skeleton_info,
                                kpt_thr,
                                radius * 2,
                                alpha=0.6,
                                line_width=line_width * 2)
    else:
        raise NotImplementedError
    return new_img

def findAngle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm <= 1e-6 or v2_norm <= 1e-6 or (v1_norm * v2_norm <= 1e-6):
        return 0
    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    angle = np.rad2deg(np.arccos(cos_angle))
    angle = np.nan_to_num(angle)
    return angle




def draw_angle_info(skel_img, keypoint):
    bar_width = 10
    left_shoulder = keypoint[5]
    left_elbow = keypoint[7]
    left_wrist = keypoint[9]
    left_hip = keypoint[11]

    right_shoulder = keypoint[6]
    right_elbow = keypoint[8]
    right_wrist = keypoint[10]
    right_hip = keypoint[12]

    left_angle = findAngle(left_shoulder, left_elbow, left_wrist)
    right_angle = findAngle(right_shoulder, right_elbow, right_wrist)
    left_draw =  np.interp(180 - left_angle, (0, 180),
              (skel_img.shape[0] - bar_width, bar_width))
    right_draw =  np.interp(180 - right_angle, (0, 180),
              (skel_img.shape[0] - bar_width, bar_width))

    draw_left_s = bar_width * 3
    draw_right_s = bar_width
    cv2.rectangle(skel_img, (draw_left_s, bar_width), (draw_left_s + bar_width, skel_img.shape[0] - bar_width),
                  (0, 0, 255), 1)
    cv2.rectangle(skel_img, (draw_left_s, int(left_draw)), (draw_left_s + bar_width, skel_img.shape[0] - bar_width),
                  (0, 0, 255), cv2.FILLED)

    cv2.rectangle(skel_img, (draw_right_s, bar_width), (draw_right_s + bar_width, skel_img.shape[0] - bar_width),
                  (0, 255,  0), 1)
    cv2.rectangle(skel_img, (draw_right_s, int(right_draw)), (draw_right_s + bar_width, skel_img.shape[0] - bar_width),
                  (0, 255, 0), cv2.FILLED)


    left_shoulder_angle = findAngle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = findAngle(right_elbow, right_shoulder, right_hip)
    left_shoulder_draw =  np.interp(left_shoulder_angle, (0, 180),
              (skel_img.shape[0] - bar_width, bar_width))
    right_shoulder_draw =  np.interp(right_shoulder_angle, (0, 180),
              (skel_img.shape[0] - bar_width, bar_width))

    draw_left_sholder = bar_width * 7
    draw_right_shoulder = bar_width * 5
    cv2.rectangle(skel_img, (draw_left_sholder, bar_width), (draw_left_sholder + bar_width, skel_img.shape[0] - bar_width),
                  (255, 128, 128), 1)
    cv2.rectangle(skel_img, (draw_left_sholder, int(left_shoulder_draw)), (draw_left_sholder + bar_width, skel_img.shape[0] - bar_width),
                  (255, 128, 128), cv2.FILLED)

    cv2.rectangle(skel_img, (draw_right_shoulder, bar_width), (draw_right_shoulder + bar_width, skel_img.shape[0] - bar_width),
                  (0, 128,  128), 1)
    cv2.rectangle(skel_img, (draw_right_shoulder, int(right_shoulder_draw)), (draw_right_shoulder + bar_width, skel_img.shape[0] - bar_width),
                  (0, 128, 128), cv2.FILLED)

    return skel_img

def draw_skel_only(skel_img, img_shape, keypoints,
                scores,
                keypoint_info,
                skeleton_info,
                kpt_thr=0.5,
                radius=2,
                line_width=2, skel_color=[0, 0, 0]):

        vis_kpt = [s >= kpt_thr for s in scores]
        link_dict = {}
        for i, kpt_info in keypoint_info.items():
            link_dict[kpt_info['name']] = kpt_info['id']

        for i, ske_info in skeleton_info.items():
            link = ske_info['link']
            pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

            if vis_kpt[pt0] and vis_kpt[pt1]:
                if skel_color == [0, 0, 0]:
                    skel_color = ske_info['color']
                kpt0 = ((keypoints[pt0] / img_shape[1]) * skel_img.shape[1])
                kpt1 = ((keypoints[pt1] / img_shape[0]) * skel_img.shape[0])

                skel_img = cv2.line(skel_img, (int(kpt0[0]), int(kpt0[1])),
                               (int(kpt1[0]), int(kpt1[1])),
                               skel_color,
                               thickness=line_width)
                skel_img = draw_angle_info(skel_img, keypoints)

        return skel_img


def draw_mmpose(img,
                keypoints,
                scores,
                keypoint_info,
                skeleton_info,
                kpt_thr=0.5,
                radius=2,
                line_width=2, skel_color=(0, 0, 0)):
    assert len(keypoints.shape) == 2

    vis_kpt = [s >= kpt_thr for s in scores]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']
        kpt = keypoints[i]
        if vis_kpt[i]:
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius),
                             kpt_color, -1)

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        if vis_kpt[pt0] and vis_kpt[pt1]:
            if skel_color == (0, 0, 0):
                skel_color = ske_info['color']
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]
            img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])),
                           (int(kpt1[0]), int(kpt1[1])),
                           skel_color,
                           thickness=line_width)

    return img


def draw_openpose(img,
                  keypoints,
                  scores,
                  keypoint_info,
                  skeleton_info,
                  kpt_thr=0.4,
                  radius=4,
                  alpha=1.0,
                  line_width=2):
    h, w = img.shape[:2]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        link_color = ske_info['color']
        kpt0, kpt1 = keypoints[pt0], keypoints[pt1]
        s0, s1 = scores[pt0], scores[pt1]

        if (kpt0[0] <= 0 or kpt0[0] >= w or kpt0[1] <= 0 or kpt0[1] >= h
                or kpt1[0] <= 0 or kpt1[0] >= w or kpt1[1] <= 0 or kpt1[1] >= h
                or s0 < kpt_thr or s1 < kpt_thr or link_color is None):
            continue

        X = np.array([kpt0[0], kpt1[0]])
        Y = np.array([kpt0[1], kpt1[1]])

        if i <= 16:
            # body part
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
            transparency = 0.6
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygons = cv2.ellipse2Poly((int(mX), int(mY)),
                                        (int(length / 2), int(line_width)),
                                        int(angle), 0, 360, 1)
            img = draw_polygons(img,
                                polygons,
                                edge_colors=link_color,
                                alpha=transparency)
        else:
            img = cv2.line(img, (int(X[0]), int(Y[0])), (int(X[1]), int(Y[1])),
                           link_color,
                           thickness=2)

    for j, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'][::-1])
        kpt = keypoints[j]

        if scores[j] < kpt_thr or sum(kpt_color) == 0:
            continue

        transparency = alpha
        if 24 <= j <= 91:
            j_radius = 3
        else:
            j_radius = 4
        # j_radius = radius // 2 if j > 17 else radius

        img = draw_circles(img,
                           kpt,
                           radius=np.array([j_radius]),
                           face_colors=kpt_color,
                           alpha=transparency)

    return img


def draw_polygons(img, polygons, edge_colors, alpha=1.0):
    if alpha == 1.0:
        img = cv2.fillConvexPoly(img, polygons, edge_colors)
    else:
        img = cv2.fillConvexPoly(img.copy(), polygons, edge_colors)
        img = cv2.addWeighted(img, 1 - alpha, img, alpha, 0)
    return img


def draw_circles(img, center, radius, face_colors, alpha=1.0):
    if alpha == 1.0:
        img = cv2.circle(img, (int(center[0]), int(center[1])), int(radius),
                         face_colors, -1)
    else:
        img = cv2.circle(img.copy(), (int(center[0]), int(center[1])),
                         int(radius), face_colors, -1)
        img = cv2.addWeighted(img, 1 - alpha, img, alpha, 0)
    return img
