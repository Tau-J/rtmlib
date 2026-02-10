from .draw import draw_bbox, draw_skeleton
from .skeleton import (animal17, coco17, coco25, coco133, halpe26, hand21,
                       openpose18, openpose134)

__all__ = [
    'draw_skeleton', 'draw_bbox', 'coco17', 'coco133', 'hand21', 'openpose18',
    'openpose134', 'halpe26', 'coco25', 'animal17'
]
