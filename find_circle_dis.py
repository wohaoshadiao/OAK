# coding=utf-8
import cv2
import depthai as dai
import numpy as np
import time

numClasses = 80
model = dai.OpenVINO.Blob("custom_model.blob")
dim = next(iter(model.networkInputs.values())).dims
W, H = dim[:2]

# 获取了第一个输出张量的名称和张量本身
output_name, output_tenser = next(iter(model.networkOutputs.items()))
if "yolov6" in output_name:
    numClasses = output_tenser.dims[2] - 5
else:
    numClasses = output_tenser.dims[2] // 3 - 5

# 将数字标签转化为物体类型
labelMap = [
    # "class_1","class_2","..."
    "class_%s" % i
    for i in range(numClasses)
]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
# 创建一个yolo算法目标检测的节点对象，用于接受输入节点的数据以及传输数据至下一个节点
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)


# 自定义更改创建左右相机
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)


# 创建xlinkout节点，用于将设备和host通信
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)


# 指定输出流节点为image,host需指定image节点对象
xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")



# Properties
camRgb.setPreviewSize(W, H)

# 设置摄像头的采集分辨率为1080p
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)


# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)


stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
stereo.setSubpixel(True)

# 将加载好的神经网络模型设置到yolo目标检测节点中去检测
detectionNetwork.setBlob(model)
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.input.setBlocking(False)
detectionNetwork.setBoundingBoxScaleFactor(0.5)
detectionNetwork.setDepthLowerThreshold(100)
detectionNetwork.setDepthUpperThreshold(5000)


# Yolo specific parameters
detectionNetwork.setNumClasses(numClasses)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors(
    [
    10,13, 16,30, 33,23,
    30,61, 62,45, 59,119,
    116,90, 156,198, 373,326
    ]
)

# YOLOV5 MASKS
detectionNetwork.setAnchorMasks(
    {
        "side%s" % (W // 8): [0, 1, 2],
        "side%s" % (W // 16): [3, 4, 5],
        "side%s" % (W // 32): [6, 7, 8],
    }
)
detectionNetwork.setIouThreshold(0.5)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)


# Linking
camRgb.preview.link(detectionNetwork.input)
camRgb.preview.link(xoutRgb.input)
detectionNetwork.out.link(xoutNN.input)

stereo.depth.link(detectionNetwork.inputDepth)
detectionNetwork.passthroughDepth.link(xoutDepth.input)
detectionNetwork.outNetwork.link(nnNetworkOut.input)



# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    imageQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

    frame = None
    detections = []

# here
    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    printOutputLayersOnce = True

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        # 归一化物体检测的坐标
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]

        # 调整坐标比例，将比例坐标转化为绝对坐标
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def drawText(frame, text, org, color=(255, 0, 255), thickness=1):
        cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness + 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA
        )

    def drawRect(frame, topLeft, bottomRight, color=(255, 255, 255), thickness=1):
        # 在图像上绘制矩形框,topLeft,bottomRight 分别为左上角和右下角坐标
        cv2.rectangle(frame, topLeft, bottomRight, (0, 0, 0), thickness + 3)
        cv2.rectangle(frame, topLeft, bottomRight, color, thickness)
        print('find_circle')

    def displayFrame(name, frame):
        color = (128, 128, 128)
        for detection in detections:
            bbox = frameNorm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )
            drawText(
                frame=frame,
                text=labelMap[detection.label],
                org=(bbox[0] + 10, bbox[1] + 20),
            )
            drawText(
                frame=frame,
                text=f"{detection.confidence:.2%}",
                org=(bbox[0] + 10, bbox[1] + 35),
            )
            drawRect(
                frame=frame,
                topLeft=(bbox[0], bbox[1]),
                bottomRight=(bbox[2], bbox[3]),
                color=color,
            )
        # Show the frame
        cv2.imshow(name, frame)

    while True:
        imageQueueData = imageQueue.tryGet()
        detectQueueData = detectQueue.tryGet()

        if imageQueueData is not None:
            frame = imageQueueData.getCvFrame()

        if detectQueueData is not None:
            detections = detectQueueData.detections

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord("q"):
            break