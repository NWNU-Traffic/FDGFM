import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'F:\RT-DETR\Weight\rtdetr-r18.exp\weights\best.pt')
    model.predict(source= r'F:\RT-DETR\images',
                  conf=0.25,
                  project='F:/RT-DETR/runs',
                  name='rtdetr-r18.exp',
                  save=True,
                  )