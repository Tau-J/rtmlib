animal17 = dict(name='animal17',
                keypoint_info={
                    0:
                    dict(name='left_eye',
                         id=0,
                         color=[51, 153, 255],
                         swap='right_eye'),
                    1:
                    dict(name='right_eye',
                         id=1,
                         color=[51, 153, 255],
                         swap='left_eye'),
                    2:
                    dict(name='nose', id=2, color=[51, 153, 255], swap=''),
                    3:
                    dict(name='neck', id=3, color=[51, 153, 255], swap=''),
                    4:
                    dict(name='root_of_tail',
                         id=4,
                         color=[51, 153, 255],
                         swap=''),
                    5:
                    dict(name='left_shoulder',
                         id=5,
                         color=[0, 255, 0],
                         swap='right_shoulder'),
                    6:
                    dict(name='left_elbow',
                         id=6,
                         color=[0, 255, 0],
                         swap='right_elbow'),
                    7:
                    dict(name='left_front_paw',
                         id=7,
                         color=[0, 255, 0],
                         swap='right_front_paw'),
                    8:
                    dict(name='right_shoulder',
                         id=8,
                         color=[255, 128, 0],
                         swap='left_shoulder'),
                    9:
                    dict(name='right_elbow',
                         id=9,
                         color=[255, 128, 0],
                         swap='left_elbow'),
                    10:
                    dict(name='right_front_paw',
                         id=10,
                         color=[255, 128, 0],
                         swap='left_front_paw'),
                    11:
                    dict(name='left_hip',
                         id=11,
                         color=[0, 255, 0],
                         swap='right_hip'),
                    12:
                    dict(name='left_knee',
                         id=12,
                         color=[0, 255, 0],
                         swap='right_knee'),
                    13:
                    dict(name='left_back_paw',
                         id=13,
                         color=[0, 255, 0],
                         swap='right_back_paw'),
                    14:
                    dict(name='right_hip',
                         id=14,
                         color=[255, 128, 0],
                         swap='left_hip'),
                    15:
                    dict(name='right_knee',
                         id=15,
                         color=[255, 128, 0],
                         swap='left_knee'),
                    16:
                    dict(name='right_back_paw',
                         id=16,
                         color=[255, 128, 0],
                         swap='left_back_paw')
                },
                skeleton_info={
                    0:
                    dict(link=('left_back_paw', 'left_knee'),
                         id=0,
                         color=[0, 255, 0]),
                    1:
                    dict(link=('left_knee', 'left_hip'),
                         id=1,
                         color=[0, 255, 0]),
                    3:
                    dict(link=('left_hip', 'root_of_tail'),
                         id=3,
                         color=[0, 255, 0]),
                    4:
                    dict(link=('right_back_paw', 'right_knee'),
                         id=4,
                         color=[255, 128, 0]),
                    5:
                    dict(link=('right_knee', 'right_hip'),
                         id=5,
                         color=[255, 128, 0]),
                    6:
                    dict(link=('right_hip', 'root_of_tail'),
                         id=6,
                         color=[255, 128, 0]),
                    7:
                    dict(link=('root_of_tail', 'neck'),
                         id=7,
                         color=[51, 153, 255]),
                    8:
                    dict(link=('neck', 'left_shoulder'),
                         id=8,
                         color=[0, 255, 0]),
                    9:
                    dict(link=('left_shoulder', 'left_elbow'),
                         id=9,
                         color=[0, 255, 0]),
                    10:
                    dict(link=('left_elbow', 'left_front_paw'),
                         id=10,
                         color=[0, 255, 0]),
                    11:
                    dict(link=('neck', 'right_shoulder'),
                         id=11,
                         color=[255, 128, 0]),
                    12:
                    dict(link=('right_shoulder', 'right_elbow'),
                         id=12,
                         color=[255, 128, 0]),
                    13:
                    dict(link=('right_elbow', 'right_front_paw'),
                         id=13,
                         color=[255, 128, 0]),
                    14:
                    dict(link=('neck', 'nose'), id=14, color=[51, 153, 255]),
                    15:
                    dict(link=('left_eye', 'right_eye'),
                         id=15,
                         color=[51, 153, 255]),
                    16:
                    dict(link=('nose', 'left_eye'),
                         id=16,
                         color=[51, 153, 255]),
                    17:
                    dict(link=('nose', 'right_eye'),
                         id=17,
                         color=[51, 153, 255])
                })
