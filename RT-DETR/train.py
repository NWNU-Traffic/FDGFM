import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
if __name__ == '__main__':
    model = RTDETR(r'F:\RT-DETR\ultralytics\cfg\models\Change\rtdetr-fgdfm.yaml')
    model.train(data=r'F:\RT-DETR\dataset\GTSDB\gtsdb.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=2,
                workers=0,
                device='cpu',
                project='runs/train',
                name='rtdetr-r18.exp',
                )