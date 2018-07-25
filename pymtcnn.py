# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import caffe
import time
import cv2

caffe.set_device(0)
caffe.set_mode_gpu()

class Net(object):
    def __init__(self, stage):
        self.caffemodel = './model/det{}.caffemodel'.format(stage)
        self.prototxt = './model/det{}.prototxt'.format(stage)
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
        self.input_size = None
        self.detect_thresh_value = None

    def bbreg(self, boundingbox, reg):
        #reg now is (numbox, 4), gemfield
        reg = reg.T

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1

        bb0 = boundingbox[:, 0] + reg[:, 0] * w
        bb1 = boundingbox[:, 1] + reg[:, 1] * h
        bb2 = boundingbox[:, 2] + reg[:, 2] * w
        bb3 = boundingbox[:, 3] + reg[:, 3] * h

        boundingbox[:, 0:4] = np.array([bb0, bb1, bb2, bb3]).T
        return boundingbox

    def pad(self, boxesA, w, h):
        boxes = boxesA.copy()
        tmph = boxes[:, 3] - boxes[:, 1] + 1
        tmpw = boxes[:, 2] - boxes[:, 0] + 1
        numbox = boxes.shape[0]

        dx = np.ones(numbox)
        dy = np.ones(numbox)
        edx = tmpw
        edy = tmph
        x = boxes[:, 0:1][:, 0]
        y = boxes[:, 1:2][:, 0]
        ex = boxes[:, 2:3][:, 0]
        ey = boxes[:, 3:4][:, 0]

        tmp = np.where(ex > w)[0]
        if tmp.shape[0] != 0:
            edx[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
            ex[tmp] = w - 1

        tmp = np.where(ey > h)[0]
        if tmp.shape[0] != 0:
            edy[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
            ey[tmp] = h - 1

        tmp = np.where(x < 1)[0]
        if tmp.shape[0] != 0:
            dx[tmp] = 2 - x[tmp]
            x[tmp] = np.ones_like(x[tmp])

        tmp = np.where(y < 1)[0]
        if tmp.shape[0] != 0:
            dy[tmp] = 2 - y[tmp]
            y[tmp] = np.ones_like(y[tmp])

        dy = np.maximum(0, dy - 1)
        dx = np.maximum(0, dx - 1)
        y = np.maximum(0, y - 1)
        x = np.maximum(0, x - 1)
        edy = np.maximum(0, edy - 1)
        edx = np.maximum(0, edx - 1)
        ey = np.maximum(0, ey - 1)
        ex = np.maximum(0, ex - 1)
        return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]

    def rerec(self, bboxA):
        w = bboxA[:, 2] - bboxA[:, 0]
        h = bboxA[:, 3] - bboxA[:, 1]
        l = np.maximum(w, h).T
        bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
        bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
        bboxA[:, 2:4] = bboxA[:, 0:2] + np.repeat([l], 2, axis=0).T
        return bboxA

    def nms(self, dets, thresh, type):  
        x1 = dets[:, 0]  
        y1 = dets[:, 1]  
        x2 = dets[:, 2]  
        y2 = dets[:, 3]  
        scores = dets[:, 4]
    
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]  
        keep = []  
        while order.size > 0:  
            i = order[0]  
            keep.append(i)  
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])  
            xx2 = np.minimum(x2[i], x2[order[1:]])  
            yy2 = np.minimum(y2[i], y2[order[1:]])  
    
            w = np.maximum(0.0, xx2 - xx1 + 1)  
            h = np.maximum(0.0, yy2 - yy1 + 1)  
            inter = w * h  
            if type == 'Min':
                ovr = inter / np.minimum(areas[i], areas[order[1:]])
            else:
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        end = time.time()
        return keep

    def forward(self, d):
        total_boxes = d['boxes']
        numbox = total_boxes.shape[0]
        h = d['frame'].shape[0]
        w = d['frame'].shape[1]

        [dy, edy, dx, edx, y, ey, x, ex, tmpw,tmph] = self.pad(total_boxes, w, h)
        input_img = np.zeros((numbox, self.input_size, self.input_size, 3))
        for k in range(numbox):
            gemfield_tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            gemfield_tmp[int(dy[k]):int(edy[k]) + 1, int(dx[k]):int(edx[k]) + 1] = d['frame'][int(y[k]):int(ey[k]) + 1, int(x[k]):int(ex[k]) + 1]
            input_img[k, :, :, :] = cv2.resize(gemfield_tmp, (self.input_size, self.input_size))

        input_img = (input_img - 127.5) / 128
        input_img = np.swapaxes(input_img, 1, 3)
        self.net.blobs['data'].reshape(numbox, 3, self.input_size, self.input_size)
        self.net.blobs['data'].data[...] = input_img
        out = self.net.forward()
        return out

class PNet(Net):
    def __init__(self):
        super(PNet, self).__init__(1)
        self.detect_thresh_value = 0.8

    def generateBoundingBox(self, map, reg, scale):
        stride = 2
        cellsize = 12
        # y*x -> x*y
        map = map.T
        
        dx1 = reg[0, :, :].T
        dy1 = reg[1, :, :].T
        dx2 = reg[2, :, :].T
        dy2 = reg[3, :, :].T

        (x, y) = np.where(map >= self.detect_thresh_value)
        score = map[x, y]
        # reg shape is  (4, numbox)
        reg = np.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])
        # boundingbox shape is (933, 2)
        boundingbox = np.array([y, x]).T
        # mapping to original image, shape is (2, numbox)
        bb1 = np.fix((stride * (boundingbox) + 1) / scale).T
        # 12 pixel toward right and bottom, quare is generated, shape is (2, numbox), by gemfield
        bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) /scale).T
        # shape is (1, numbox), notice the [score], it reform (numbox,) to (1,numbox), by gemfield
        score = np.array([score])
        # (9, numbox), 2 + 2 + 1 + 4 = 9
        boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)
        return boundingbox_out.T
        
    def processOne(self, frame, scale):
        h = frame.shape[0]
        w = frame.shape[1]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))

        im_data = cv2.resize(frame, (ws, hs))
        im_data = (im_data - 127.5)/128

        # hwc -> cwh
        im_data = np.swapaxes(im_data, 0, 2)

        # PNet.blobs['data'] is caffe._caffe.Blob
        self.net.blobs['data'].reshape(1, 3, ws, hs)
        # c, w, h broadcast to 1, c, w, h
        self.net.blobs['data'].data[...] = np.array([im_data], dtype=np.float)
        # out is dict
        out = self.net.forward()
        
        # 0 means non face, 1 means face
        boxes = self.generateBoundingBox(out['prob1'][0, 1, :, :], out['conv4-2'][0], scale)
        if boxes.shape[0] != 0:
            pick = self.nms(boxes, 0.5, 'Union')
            if len(pick) > 0:
                boxes = boxes[pick, :]
        return boxes

    def process(self, d):
        frame = d['frame']
        for scale in d['scales']:
            boxes = self.processOne(frame, scale)
            if boxes.shape[0] != 0:
                d['boxes'] = np.concatenate((d['boxes'], boxes), axis=0)

        numbox = d['boxes'].shape[0]

        if numbox <= 0:
            return False

        total_boxes = d['boxes']
        pick = self.nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]

        if total_boxes.shape[0] <= 0:
            return False

        total_boxes = self.bbreg(total_boxes[:,0:4], total_boxes[:,5:9].T )
        total_boxes = self.rerec(total_boxes)
        total_boxes = np.fix(total_boxes)
        d['boxes'] = total_boxes
        return True

class RNet(Net):
    def __init__(self):
        super(RNet, self).__init__(2)
        self.input_size = 24
        self.detect_thresh_value = 0.9
    
    def process(self, d):
        out = self.forward(d)
        # score shape is (numbox,)
        score = out['prob1'][:, 1]
        pass_t = np.where(score > self.detect_thresh_value)[0]
        # score shape now is (numbox, 1)
        score = np.array([score[pass_t]]).T
        #total_boxes shape: (numbox, 5)
        total_boxes = d['boxes']
        total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
        #out['conv5-2'] shape is (numbox, 4), and mv shape is (4, numbox)
        mv = out['conv5-2'][pass_t, :].T
        if total_boxes.shape[0] <= 0:
            return False

        pick = self.nms(total_boxes, 0.7, 'Union')
        if len(pick) > 0:
            total_boxes = total_boxes[pick, :]
            total_boxes = self.bbreg(total_boxes, mv[:, pick])
            total_boxes = self.rerec(total_boxes)
        if total_boxes.shape[0] <= 0:
            return False

        total_boxes = np.fix(total_boxes)
        d['boxes'] = total_boxes
        return True

class ONet(Net):
    def __init__(self):
        super(ONet, self).__init__(3)
        self.input_size = 48
        self.detect_thresh_value = 0.95

    def process(self, d):
        out = self.forward(d)
        score = out['prob1'][:, 1]
        points = out['conv6-3']
        pass_t = np.where(score > self.detect_thresh_value)[0]
        points = points[pass_t, :]
        score = np.array([score[pass_t]]).T
        total_boxes = d['boxes']
        total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
        mv = out['conv6-2'][pass_t, :].T

        w = total_boxes[:, 3] - total_boxes[:, 1] + 1
        h = total_boxes[:, 2] - total_boxes[:, 0] + 1

        points[:, 0:5] = np.tile(w, (5, 1)).T * points[:, 0:5] + np.tile(total_boxes[:, 0], (5, 1)).T - 1
        points[:, 5:10] = np.tile(h, (5, 1)).T * points[:, 5:10] + np.tile(total_boxes[:, 1], (5, 1)).T - 1
        if total_boxes.shape[0] <= 0:
            return False

        total_boxes = self.bbreg(total_boxes, mv[:, :])
        pick = self.nms(total_boxes, 0.7, 'Min')
        if len(pick) > 0:
            total_boxes = total_boxes[pick, :]
            points = points[pick, :]
        d['boxes'] = total_boxes
        d['points'] = points
        return True

class Mtcnn(object):
    def __init__(self):
        self.nets = [PNet(), RNet(), ONet()]
        self.d = {}
        self.d['scales'] = [0.3, 0.15, 0.075]

    def preprocess(self, d):
        frame = cv2.cvtColor(d['frame'], cv2.COLOR_BGR2RGB)
        d['frame'] = frame.astype(float)

    def postprocess(self, d):
        self.d['boxes'] = np.maximum(d['boxes'], 0)

    def process(self, frame):
        self.d['frame'] = frame
        self.d['boxes'] = np.zeros((0, 9), np.float)
        self.d['points'] = np.zeros((1, 1))

        self.preprocess(self.d)
        
        for net in self.nets:
            rc = net.process(self.d)
            if not rc:
                return False
        self.postprocess(self.d)
        return True
    
    def drawBoxes(self, frame, imgname):
        boxes = self.d['boxes']
        points = self.d['points']
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        for i in range(x1.shape[0]):
            rightEyeCenter = (points[i][0], points[i][5])
            leftEyeCenter = (points[i][1], points[i][6])
            p3 = (points[i][2], points[i][7])
            p4 = (points[i][3], points[i][8])
            p5 = (points[i][4], points[i][9])
            cv2.rectangle(frame, (int(x1[i]), int(y1[i])),(int(x2[i]), int(y2[i])), (0, 255, 0), 1)
            cv2.circle(frame, rightEyeCenter, 2, (0, 0, 255), 3)
            cv2.circle(frame, leftEyeCenter, 2, (0, 255, 0), 3)
            cv2.circle(frame, p3, 2, (255, 0, 0), 3)
            cv2.circle(frame, p4, 2, (255, 255, 0), 3)
            cv2.circle(frame, p5, 2, (0, 255, 255), 3)

            write_path = './results/{}/bbox/'.format( time.strftime("%Y%m%d", time.localtime()) )
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            cv2.imwrite('{}{}.jpg'.format(write_path, imgname), frame)

class AlignFace(object):
    def __init__(self):
        # reference facial points, a list of coordinates (x,y)
        self.REFERENCE_FACIAL_POINTS_96x112 = [
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 92.3655014],
            [62.72990036, 92.20410156]
        ]
        # made by gemfield
        self.REFERENCE_FACIAL_POINTS_112x112 = [
            [38.29459953, 51.69630051],
            [73.53179932, 51.50139999],
            [56.02519989, 71.73660278],
            [41.54930115, 92.3655014 ],
            [70.72990036, 92.20410156]
        ]

    def __call__(self, frame, facial_5pts):
        # shape from (10,) to (2, 5)
        facial_5pts = np.reshape(facial_5pts, (2, -1))
        dst_img = self.warpAndCrop(frame, facial_5pts, (112, 112))
        return dst_img

    def warpAndCrop(self, src_img, facial_pts, crop_size):
        reference_pts = self.REFERENCE_FACIAL_POINTS_112x112
        ref_pts = np.float32(reference_pts)
        ref_pts_shp = ref_pts.shape

        if ref_pts_shp[0] == 2:
            ref_pts = ref_pts.T

        src_pts = np.float32(facial_pts)
        src_pts_shp = src_pts.shape
        if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
            raise Exception('facial_pts.shape must be (K,2) or (2,K) and K>2')
        # 2*5 to 5*2
        if src_pts_shp[0] == 2:
            src_pts = src_pts.T

        if src_pts.shape != ref_pts.shape:
            raise Exception('facial_pts and reference_pts must have the same shape: {} vs {}'.format(src_pts.shape, ref_pts.shape) )

        tfm = self.getAffineTransform(src_pts, ref_pts)

        face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))
        return face_img

    def getAffineTransform(self, uv, xy):
        options = {'K': 2}
        # Solve for trans1
        trans1, trans1_inv = self.findNonreflectiveSimilarity(uv, xy, options)
        # manually reflect the xy data across the Y-axis
        xyR = xy
        xyR[:, 0] = -1 * xyR[:, 0]

        trans2r, trans2r_inv = self.findNonreflectiveSimilarity(uv, xyR, options)

        # manually reflect the tform to undo the reflection done on xyR
        TreflectY = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        trans2 = np.dot(trans2r, TreflectY)

        # Figure out if trans1 or trans2 is better
        xy1 = self.tformfwd(trans1, uv)
        norm1 = norm(xy1 - xy)

        xy2 = self.tformfwd(trans2, uv)
        norm2 = norm(xy2 - xy)

        if norm1 <= norm2:
            trans = trans1
        else:
            trans2_inv = inv(trans2)
            trans = trans2

        cv2_trans = trans[:, 0:2].T
        return cv2_trans

    def findNonreflectiveSimilarity(self, uv, xy, options=None):
        options = {'K': 2}

        K = options['K']
        M = xy.shape[0]
        x = xy[:, 0].reshape((-1, 1))
        y = xy[:, 1].reshape((-1, 1))

        tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        X = np.vstack((tmp1, tmp2))

        u = uv[:, 0].reshape((-1, 1))
        v = uv[:, 1].reshape((-1, 1))
        U = np.vstack((u, v))

        if rank(X) >= 2 * K:
            r, _, _, _ = lstsq(X, U, rcond=-1)
            r = np.squeeze(r)
        else:
            raise Exception('cp2tform:twoUniquePointsReq')
        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]

        Tinv = np.array([
            [sc, -ss, 0],
            [ss, sc, 0],
            [tx, ty, 1]
        ])

        T = inv(Tinv)
        T[:, 2] = np.array([0, 0, 1])
        return T, Tinv

    def tformfwd(self, trans, uv):
        uv = np.hstack((
            uv, np.ones((uv.shape[0], 1))
        ))
        xy = np.dot(uv, trans)
        xy = xy[:, 0:-1]
        return xy

    def drawBoxes(self, img, imgname):
        write_path = './results/{}/bbox/'.format( time.strftime("%Y%m%d", time.localtime()) )
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        cv2.imwrite('{}{}_aligned.jpg'.format(write_path, imgname), img)

#### gemfield test phase ####
video_path = 'gemfield.mp4'
if __name__ == '__main__':
    if len(sys.argv) == 2:
        video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print('File not found: {}'.format(video_path))
        sys.exit(1)

    videoCapture = cv2.VideoCapture(video_path)
    status, frame = videoCapture.read()
    mtcnn = Mtcnn()
    alignFace = AlignFace()
    frame_num = 0
    while status:
        frame_num += 1
        print('process {}'.format(frame_num))
        rc = mtcnn.process(frame)
        if not rc:
            status, frame = videoCapture.read()
            continue
        mtcnn.drawBoxes(frame, frame_num)
        points = mtcnn.d['points']
        if points.shape[0] > 0:
            for i in range(points.shape[0]):
                #gemfield.org: WARNING! THE IMG WILL CONTAIN THE MARK drew earlier!
                alignFace.drawBoxes(alignFace(frame, points[i,:]), frame_num)
        status, frame = videoCapture.read()
    else:
        print('End for some reasons...')
