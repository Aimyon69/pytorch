import sys
sys.path.append('D:/Code/pytorch/Design/mtcnn')
sys.path.append('D:/Code/pytorch/Design/ResNet')
from consolidate import MTCNNDetector
import cv2
from model import ResNet18
import torch
import numpy as np
def align_face(img, landmarks):
    left_eye=(landmarks[0], landmarks[1])
    right_eye=(landmarks[2], landmarks[3])
    dy=right_eye[1]-left_eye[1]
    dx=right_eye[0]-left_eye[0]
    angle=np.degrees(np.arctan2(dy, dx)) 
    center=((left_eye[0]+right_eye[0])//2,(left_eye[1]+right_eye[1])//2)
    M=cv2.getRotationMatrix2D(center,angle,1.0)
    aligned=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    return aligned
classes=['angry','disgust','fear','happy','neutral','sad','surprise']
def video():
    device=torch.device('cuda')
    mtcnn=MTCNNDetector()
    model=ResNet18(num_classes=7,in_channels=1)
    model.load_state_dict(torch.load('D:/Code/pytorch/Design/resnet.pth'))
    model.to(device)
    model.eval()
    index=None
    for i in range(11):
        cap=cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            index=i
            print('successfully find')
            cap.release()
            break
        cap.release()
    if index==None:
        exit()
    cap=cv2.VideoCapture(index)
    while(cap.isOpened()):
        ret,frame=cap.read()
        if ret==True:
            boxes,lm=mtcnn.detect_face(frame)
            for step,box in enumerate(boxes):
                box[0]=int(max(0, box[0]))
                box[1]=int(max(0, box[1]))
                box[2]=int(min(frame.shape[1], box[2]))
                box[3]=int(min(frame.shape[0], box[3]))
                if box[2]<=box[0] or box[3]<=box[1]:
                    continue
                cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
                current_lm = lm[step]
                relative_lm = []
                for k in range(5):
                    relative_lm.append(current_lm[2*k] - int(box[0])) 
                    relative_lm.append(current_lm[2*k+1] - int(box[1]))
                face_gray=cv2.cvtColor(align_face(frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])],relative_lm),cv2.COLOR_BGR2GRAY)
                face_gray=cv2.resize(face_gray,(48,48))
                face_gray=face_gray/255
                face_tensor=torch.tensor(face_gray,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    cls_out=model(face_tensor)
                preb=torch.argmax(cls_out,dim=1)
                cv2.putText(frame,classes[preb],(int(box[2]),int(box[3])),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2,cv2.LINE_AA)
            for land in lm:
                for k in range(5):
                    cv2.circle(frame,(int(land[k*2]),int(land[k*2+1])),1,(0,0,255))
            
            cv2.imshow('video',frame)
        if cv2.waitKey(int(1000/30))&0xff==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def picture(img_path):
    device=torch.device('cuda')
    mtcnn=MTCNNDetector()
    model=ResNet18(num_classes=7,in_channels=1)
    model.load_state_dict(torch.load('D:/Code/pytorch/Design/resnet.pth'))
    model.to(device)
    model.eval()
    frame=cv2.imread(img_path)
    boxes,lm=mtcnn.detect_face(frame)
    for step,box in enumerate(boxes):
        box[0]=int(max(0, box[0]))
        box[1]=int(max(0, box[1]))
        box[2]=int(min(frame.shape[1], box[2]))
        box[3]=int(min(frame.shape[0], box[3]))
        if box[2]<=box[0] or box[3]<=box[1]:
            continue
        cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
        current_lm = lm[step]
        relative_lm = []
        for k in range(5):
            relative_lm.append(current_lm[2*k] - int(box[0])) 
            relative_lm.append(current_lm[2*k+1] - int(box[1]))
        face_gray=cv2.cvtColor(align_face(frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])],relative_lm),cv2.COLOR_BGR2GRAY)
        face_gray=cv2.resize(face_gray,(48,48))
        face_gray=face_gray/255
        face_tensor=torch.tensor(face_gray,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_out=model(face_tensor)
        preb=torch.argmax(cls_out,dim=1)
        cv2.putText(frame,classes[preb],(int(box[2]),int(box[3])),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2,cv2.LINE_AA)
        for land in lm:
            for k in range(5):
                cv2.circle(frame,(int(land[k*2]),int(land[k*2+1])),1,(0,0,255))
    cv2.imshow('frame',frame)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__=='__main__':
    picture('D:/image/disgust.jpg')