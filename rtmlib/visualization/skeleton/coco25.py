coco25 = dict(name='coco25',
              keypoint_info={
                  0:
                  dict(name='nose', id=0, color=[51, 153, 255], swap=''),
                  1:
                  dict(name='left_eye',
                       id=1,
                       color=[51, 153, 255],
                       type='upper',
                       swap='right_eye'),
                  2:
                  dict(name='right_eye',
                       id=2,
                       color=[51, 153, 255],
                       type='upper',
                       swap='left_eye'),
                  3:
                  dict(name='left_ear',
                       id=3,
                       color=[51, 153, 255],
                       type='upper',
                       swap='right_ear'),
                  4:
                  dict(name='right_ear',
                       id=4,
                       color=[51, 153, 255],
                       type='upper',
                       swap='left_ear'),
                  5:
                  dict(name='neck',
                       id=5,
                       color=[51, 153, 255],
                       type='upper',
                       swap=''),
                  6:
                  dict(name='left_shoulder',
                       id=6,
                       color=[0, 255, 0],
                       type='upper',
                       swap='right_shoulder'),
                  7:
                  dict(name='right_shoulder',
                       id=7,
                       color=[255, 128, 0],
                       type='upper',
                       swap='left_shoulder'),
                  8:
                  dict(name='left_elbow',
                       id=8,
                       color=[0, 255, 0],
                       type='upper',
                       swap='right_elbow'),
                  9:
                  dict(name='right_elbow',
                       id=9,
                       color=[255, 128, 0],
                       type='upper',
                       swap='left_elbow'),
                  10:
                  dict(name='left_wrist',
                       id=10,
                       color=[0, 255, 0],
                       type='upper',
                       swap='right_wrist'),
                  11:
                  dict(name='right_wrist',
                       id=11,
                       color=[255, 128, 0],
                       type='upper',
                       swap='left_wrist'),
                  12:
                  dict(name='left_hip',
                       id=12,
                       color=[0, 255, 0],
                       type='lower',
                       swap='right_hip'),
                  13:
                  dict(name='right_hip',
                       id=13,
                       color=[255, 128, 0],
                       type='lower',
                       swap='left_hip'),
                  14:
                  dict(name='hip',
                       id=14,
                       color=[51, 153, 255],
                       type='lower',
                       swap=''),
                  15:
                  dict(name='left_knee',
                       id=15,
                       color=[0, 255, 0],
                       type='lower',
                       swap='right_knee'),
                  16:
                  dict(name='right_knee',
                       id=16,
                       color=[255, 128, 0],
                       type='lower',
                       swap='left_knee'),
                  17:
                  dict(name='left_ankle',
                       id=17,
                       color=[0, 255, 0],
                       type='lower',
                       swap='right_ankle'),
                  18:
                  dict(name='right_ankle',
                       id=18,
                       color=[255, 128, 0],
                       type='lower',
                       swap='left_ankle'),
                  19:
                  dict(name='left_big_toe',
                       id=19,
                       color=[255, 128, 0],
                       type='lower',
                       swap='right_big_toe'),
                  20:
                  dict(name='left_small_toe',
                       id=20,
                       color=[255, 128, 0],
                       type='lower',
                       swap='right_small_toe'),
                  21:
                  dict(name='left_heel',
                       id=21,
                       color=[255, 128, 0],
                       type='lower',
                       swap='right_heel'),
                  22:
                  dict(name='right_big_toe',
                       id=22,
                       color=[255, 128, 0],
                       type='lower',
                       swap='left_big_toe'),
                  23:
                  dict(name='right_small_toe',
                       id=23,
                       color=[255, 128, 0],
                       type='lower',
                       swap='left_small_toe'),
                  24:
                  dict(name='right_heel',
                       id=24,
                       color=[255, 128, 0],
                       type='lower',
                       swap='left_heel')
              },
              skeleton_info={
                  0:
                  dict(link=('left_ankle', 'left_knee'),
                       id=0,
                       color=[0, 255, 0]),
                  1:
                  dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255,
                                                                    0]),
                  2:
                  dict(link=('left_hip', 'hip'), id=2, color=[0, 255, 0]),
                  3:
                  dict(link=('right_ankle', 'right_knee'),
                       id=3,
                       color=[255, 128, 0]),
                  4:
                  dict(link=('right_knee', 'right_hip'),
                       id=4,
                       color=[255, 128, 0]),
                  5:
                  dict(link=('right_hip', 'hip'), id=5, color=[51, 153, 255]),
                  6:
                  dict(link=('hip', 'neck'), id=6, color=[51, 153, 255]),
                  7:
                  dict(link=('neck', 'nose'), id=7, color=[51, 153, 255]),
                  8:
                  dict(link=('nose', 'left_eye'), id=8, color=[51, 153, 255]),
                  9:
                  dict(link=('left_eye', 'left_ear'),
                       id=9,
                       color=[51, 153, 255]),
                  10:
                  dict(link=('nose', 'right_eye'), id=10, color=[51, 153,
                                                                 255]),
                  11:
                  dict(link=('right_eye', 'right_ear'),
                       id=11,
                       color=[51, 153, 255]),
                  12:
                  dict(link=('neck', 'left_shoulder'),
                       id=12,
                       color=[0, 255, 0]),
                  13:
                  dict(link=('left_shoulder', 'left_elbow'),
                       id=13,
                       color=[0, 255, 0]),
                  14:
                  dict(link=('left_elbow', 'left_wrist'),
                       id=14,
                       color=[0, 255, 0]),
                  15:
                  dict(link=('neck', 'right_shoulder'),
                       id=15,
                       color=[255, 128, 0]),
                  16:
                  dict(link=('right_shoulder', 'right_elbow'),
                       id=16,
                       color=[255, 128, 0]),
                  17:
                  dict(link=('right_elbow', 'right_wrist'),
                       id=17,
                       color=[255, 128, 0]),
                  18:
                  dict(link=('left_ankle', 'left_big_toe'),
                       id=18,
                       color=[0, 255, 0]),
                  19:
                  dict(link=('left_ankle', 'left_small_toe'),
                       id=19,
                       color=[0, 255, 0]),
                  20:
                  dict(link=('left_ankle', 'left_heel'),
                       id=20,
                       color=[0, 255, 0]),
                  21:
                  dict(link=('right_ankle', 'right_big_toe'),
                       id=21,
                       color=[255, 128, 0]),
                  22:
                  dict(link=('right_ankle', 'right_small_toe'),
                       id=22,
                       color=[255, 128, 0]),
                  23:
                  dict(link=('right_ankle', 'right_heel'),
                       id=23,
                       color=[255, 128, 0])
              })
