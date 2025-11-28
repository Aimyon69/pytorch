import numpy as np
def IoU(box,boxes,isMin=False):
    box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
    area=(boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    xx1=np.maximum(box[0],boxes[:,0])
    yy1=np.maximum(box[1],boxes[:,1])
    xx2=np.minimum(box[2],boxes[:,2])
    yy2=np.minimum(box[3],boxes[:,3])
    w=np.maximum(0,xx2-xx1+1)
    h=np.maximum(0,yy2-yy1+1)
    inter=w*h
    if isMin:
        ovr=inter/np.minimum(box_area,area)
    else:
        ovr=inter/(box_area+area-inter)
    return ovr
def NMS(boxes,threshold=0.3,isMin=False):
    if boxes.shape[0]==0:
        return np.array([])
    _boxes=boxes[(-boxes[:,4]).argsort()]
    r_boxes=[]
    while _boxes.shape[0]>1:
        a_box=_boxes[0]
        b_boxes=_boxes[1:]
        r_boxes.append(a_box)
        idx=np.where(IoU(a_box,b_boxes,isMin)<threshold)
        _boxes=b_boxes[idx]
    if _boxes.shape[0]>0:
        r_boxes.append(_boxes[0])
    return np.stack(r_boxes)
def convert_to_square(bboxes):
    square_bboxes=np.zeros_like(bboxes)
    x1,y1,x2,y2=bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]
    h=y2-y1
    w=x2-x1
    max_side=np.maximum(h,w)
    cx=x1+w*0.5
    cy=y1+h*0.5
    square_bboxes[:,0]=cx-max_side*0.5
    square_bboxes[:,1]=cy-max_side*0.5
    square_bboxes[:,2]=square_bboxes[:,0]+max_side
    square_bboxes[:,3]=square_bboxes[:,1]+max_side
    square_bboxes[:,4]=bboxes[:,4]
    return square_bboxes
def NMS_Indices(boxes, threshold=0.3, isMin=False):
    if boxes.shape[0]==0:
        return np.array([])
    sort_index = (-boxes[:, 4]).argsort()
    keep_indices = []
    while sort_index.shape[0] > 0:
        idx_self = sort_index[0]
        keep_indices.append(idx_self)
        if sort_index.shape[0] == 1:
            break
        a_box = boxes[idx_self]
        idx_others = sort_index[1:]
        b_boxes = boxes[idx_others]
        idx = np.where(IoU(a_box, b_boxes, isMin) < threshold)
        sort_index = idx_others[idx]
    return np.array(keep_indices)

