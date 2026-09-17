# Detector weights

`scratch_best.pt` is an Ultralytics YOLOv9-C fine-tuned on white 16 cm bowls on grass from the same
real DJI Mini 4 Pro imagery at 11, 15 and 40 m as the paper's detectors, trained on SAHI slices at 640 px.
One class, `bowl`. It is not in git. Copy it into this directory and check it:

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
