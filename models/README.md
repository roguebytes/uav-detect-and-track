# Detector weights

`scratch_best.pt` is an Ultralytics YOLOv9-C trained from scratch (no COCO initialisation) for 150
epochs on 640 px SAHI slices of the real DJI Mini 4 Pro imagery behind the paper: white 16 cm bowls on
grass at 11, 15 and 40 m. It is a training candidate from the same project, not one of the two
detectors characterised in the paper. The paper's released 40 m detector
(`detector_40m_yolov9c_epoch99.pt` in the dataset repository's release) was trained with the original
YOLOv9 repository and is not loadable by the Ultralytics package this repository uses. One class,
`bowl`. It is not in git. Copy it into this directory and check it:

```
sha256sum -c scratch_best.pt.sha256      # expects scratch_best.pt: OK
```

`scratch_best.pt.sha256` should contain:

```
f49cc2dac9c51eaa51ef57c52d9666b6547cfbef3565c64ecbe3cd9138f2e62e  scratch_best.pt
```

Run it tiled at native resolution (`uav_dt.detector.TiledYoloDetector`). Downscaling a 12 MP frame
to 640 px finds nothing. Without the weights, everything still runs with the ground-truth oracle
detector (`detector:=gt`), which is what the CI smoke run uses.
