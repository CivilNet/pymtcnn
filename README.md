# pymtcnn
- pymtcnn is a caffe based python implementation for MTCNN (Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks);
- You can visit https://zhuanlan.zhihu.com/p/39499030 for more detail.

# Functions
- detect faces;
- align faces with cv2.warpAffine API.


# Usage

## pre configure
There has some options you need pre configure, of course you can leave them as default value, but I don't think that is what you want;
- multi scale values: `self.d['scales'] = [0.3, 0.15, 0.075]`
- PNet detect threshold: `self.detect_thresh_value = 0.8`
- RNet detect threshold: `self.detect_thresh_value = 0.9`
- ONet detect threshold: `self.detect_thresh_value = 0.95`

## call the API
```python
videoCapture = cv2.VideoCapture(video_path)
status, frame = videoCapture.read()
# Mtcnn instance
mtcnn = Mtcnn()
# AlignFace instance
alignFace = AlignFace()
frame_num = 0
while status:
    frame_num += 1
    print('process {}'.format(frame_num))
    # do the mtcnn detect job with this API
    rc = mtcnn.process(frame)
    if not rc:
        status, frame = videoCapture.read()
        continue
    # draw the boxes just for test purpose
    mtcnn.drawBoxes(frame, frame_num)
    points = mtcnn.d['points']
    if points.shape[0] > 0:
        for i in range(points.shape[0]):
            # align the faces
            alignFace.drawBoxes(alignFace(frame, points[i,:]), frame_num)
    status, frame = videoCapture.read()
else:
    print('End for some reasons...')
```
