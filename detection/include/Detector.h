#ifndef DETECTOR_H
#define DETECTOR_H

#include <memory>
#include <array>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "detection_export.h"

enum class YOLOType {
    YOLOv3,
    YOLOv3_TINY
};

struct DETECTION_EXPORT Detection {
	cv::Rect2f bbox;  // struct: (x, y, width, height)
	float scr; // maxProbabilityOfClass
	int64 cls; // class index
};


class DETECTION_EXPORT Detector {
public:
    explicit Detector(const std::array<int64_t, 2> &_inp_dim, 
		              const char* config_file,
					  const char* weight_file, 
		              YOLOType type = YOLOType::YOLOv3);

    ~Detector();

	std::vector<Detection> predict(cv::Mat image);

private:
	torch::DeviceType device_type;

    class Darknet;
    std::unique_ptr<Darknet> net;
	
    std::array<int64_t, 2> inp_dim;
    static const float NMS_threshold;
    static const float confidence_threshold;
};

#endif //DETECTOR_H
