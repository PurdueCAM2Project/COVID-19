import argparse

from .models import *  # set ONNX_EXPORT in models.py
from .utils.datasets_for_hyperlink import *
from .utils.utils import *
from matplotlib.pyplot import imshow


class Vehicle_Detector():
    def __init__(self, weights='weights/yolov3-spp-ultralytics.pt', cfg='cfg/yolov3-spp.cfg', names='data/coco.names', iou_thres=0.3, conf_thres=0.15, imgsz=512, half=True, device_id='0'):
        self.weights = weights
        self.cfg = cfg
        self.names = load_classes(names)
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        self.half = half
        self.device = torch_utils.select_device(device=device_id)
        self.initiate_model()

    def initiate_model(self):
        self.model = Darknet(self.cfg, self.imgsz)
        self.load_weights()
        self.model.to(self.device).eval()
        half_ = self.half and self.device.type != 'cpu'
        if half_:
            self.model.half()

    def load_weights(self):
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(
                self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)

    def detect(self, image=cv2.imread('images/3.png'), view_img=False):

        im0s_ = image

        img_ = letterbox(im0s_, new_shape=self.imgsz)[0]
        # Convert
        img_ = img_[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_ = np.ascontiguousarray(img_)

        img_ = torch.from_numpy(img_).to(self.device)
        img_ = img_.half() if self.half else img_.float()  # uint8 to fp16/32
        img_ /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_.ndimension() == 3:
            img_ = img_.unsqueeze(0)

        # Inference
        pred = self.model(img_)[0]

        # to float
        if self.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        detections = dict()
        predictions = pred[0]
        valid_labels = [1, 2, 3, 4, 7]
        # labels = predictions[:,5]
        # # consider only vehicles
        # # labels
        #     # 1 -> bicycle
        #     # 2 -> car
        #     # 3 -> motorcycle
        #     # 5 -> bus
        #     # 7 -> truck
        at_least_one_useful_object_detected = predictions != None
        if at_least_one_useful_object_detected:
            for i in range(len(predictions)):
                confidence = float(predictions[i][4].cpu().detach().numpy())
                label = int(predictions[i][5])
                coordinates = predictions[i][:4].reshape(1, -1)
                object_ = self.names[label]
                coordinates = list(scale_coords(
                    img_.shape[2:], coordinates, im0s_.shape).round().cpu().detach().numpy()[0])
                if label in valid_labels:
                    detections[(object_, confidence)] = coordinates

        if view_img:
            color = (0, 255, 0)
            thickness = 3
            for key in detections.keys():
                b = detections[key]
                start = tuple(b[:2])
                end = tuple(b[2:])
                image = cv2.rectangle(
                    image, start, end, color, thickness=thickness)
            cv2.imshow("sample", image)
        return detections
