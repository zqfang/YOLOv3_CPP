YOLOv3 C++ 
=================

Libtorch implementation of YOLOv3, which works on `Windows` and `Linux`.

## Dependency

* cmake
* libTorch >= 1.1, or nightly
* OpenCV   >= 4.0
* C++17
* Win10: vs2017+cuda90, vs2019+cuda10
* Linux

## Usage

For now, only support inference. Please use `Darknet` weights format as input.  

If you need training your own model, try [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3/) and save your weights.  

To test all `jpg` format images in a folder, use your `yolo.custom.cfg` and `custom.weights`, run the code below:  


```shell
# yolov3 <config_file> <weights> <image_folder> 
yolov3.exe yolov3.cfg yolov3.weights images
```

## Build

```shell

# before running this project, you have to edit two file
# -> edit the CMakeList.txt, set correct path to libtorch and OpenCV
# -> edit the yolov3/main.cpp at line 27. Set correct image format. 
cd path/to/YOLOv3
mkdir build
cd build
cmake ..
make # linux
# if windows, open YOLOV3-app.sln and then build 
```

## Performance

Only tested with CPU on Windows for now. 


## Features
Supports  
- NMS
- Soft NMS
- Weighted NMS

## TODO
- Support training
- ...



## Credits

This repo is created based on the implementations below:  
[weixu000](https://github.com/weixu000/libtorch-yolov3-deepsort)  
[PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)  
[YOLO_v3_tutorial_from_scratch](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)







