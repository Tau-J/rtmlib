# Copyright (c) OpenMMLab. All rights reserved.
import cv2

from .visualization.skeleton import *  # noqa


def draw_bbox(img, bboxes, color=(0, 255, 0)):
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), color, 2)
    return img


def draw_skeleton(img,
                  keypoints,
                  scores,
                  skeleton='coco17',
                  style='mmpose',
                  kpt_thr=0.5,
                  radius=2,
                  line_width=2):
    assert style in {'mmpose', 'openpose'
                     }, ('`style` must be either `mmpose` or `openpose`')

    skeleton_dict = eval(f'{skeleton}')
    keypoint_info = skeleton_dict['keypoint_info']
    skeleton_info = skeleton_dict['skeleton_info']

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]

    if style == 'mmpose':
        num_instance = keypoints.shape[0]
        for i in range(num_instance):
            img = draw_mmpose(img, keypoints[i], scores[i], keypoint_info,
                              skeleton_info, kpt_thr, radius, line_width)
    return img


def draw_mmpose(img,
                keypoints,
                scores,
                keypoint_info,
                skeleton_info,
                kpt_thr=0.5,
                radius=2,
                line_width=2):
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
            link_color = ske_info['color']
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])),
                           (int(kpt1[0]), int(kpt1[1])),
                           link_color,
                           thickness=line_width)

    return img
