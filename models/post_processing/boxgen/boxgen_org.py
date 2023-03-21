bboxes = []
scores = []
for i in range(1, label_num):
    ind = label == i
    points = np.array(np.where(ind)).transpose((1, 0))
    score_i = np.mean(score[ind])
    if score_i < cfg.test_cfg.min_score:
        label[ind] = 0
        continue
    if cfg.test_cfg.bbox_type == 'rect':
        pos, length, deg = cv2.minAreaRect(points[:, ::-1])
        pos, length = np.array(pos), np.array(length)
        pos += pos_const
        length += len_const
        pos, length = pos*scale, length*scale
        bbox = cv2.boxPoints((pos, length, deg))
    bbox = bbox.astype('int32')
    bboxes.append(bbox.reshape(-1))
    
'''
label: np.int32_t, ndim=2
score: np.float32_t, ndim=2
scale: float32_t, ndim=1
label_num: int
min_score: float
'''