#include <algorithm>
#include <iostream>
#include <math.h>
#include "Detector.h"
#include "Darknet.h"
#include "letterbox.h"
#include <assert.h>
namespace {
	// x, y, h, w
    void center_to_corner(torch::Tensor bbox) {
		// top left x = centerX - w / 2 ...
        bbox.select(1, 0) -= bbox.select(1, 2) / 2;
        bbox.select(1, 1) -= bbox.select(1, 3) / 2;
		// don't covert w,h. Because we will used them in opencv directly
		//bbox.select(1, 2) += bbox.select(1, 2) / 2;
		//bbox.select(1, 3) += bbox.select(1, 3) / 2;
    }

	// pred dim: (grid_size*grid_size*num_anchors, bbox_attrs)
    auto threshold_confidence(torch::Tensor & pred, float conf_thres=0.5) {
		// filter out confidence scores below threshold  
		auto keep = pred.select(1, 4).squeeze_() > conf_thres;
        auto ind = keep.nonzero().squeeze_();
		pred = pred.index_select(0, ind);

		// handle no object found case
		if (ind.numel() == 0)
			return std::make_tuple(ind, ind, ind);

		// max will return (maxValue, indiceOf_maxValue)
        //auto [max_cls_score, max_cls_indice] = pred.slice(1, 5).max(1);
		// combined confidence score
		auto [max_cls_score, max_cls_indice] = pred.slice(1, 5).mul(pred.select(1, 4).unsqueeze(1)).max(1);
        pred = pred.slice(1, 0, 5); // extract (Tx, Ty, H, W, conf)

		return std::make_tuple(pred, max_cls_score, max_cls_indice);
    }

    float iou(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt) {
        auto in = (bb_test & bb_gt).area();  // rect intersect
		auto un = (bb_test | bb_gt).area(); // rect union

        return in / un;
    }

	// traget: 1d tensor
	// pred: 2d tensor
	torch::Tensor iou(torch::Tensor & pred) {
		torch::Tensor ious = torch::empty(pred.size(0));
		auto pacc = pred.accessor<float, 2>();
		cv::Rect2f bb_test(pacc[0][0], pacc[0][1], pacc[0][2], pacc[0][3]);

		for (int64_t i = 0; i < pred.size(0); i++) {
            cv::Rect2f bb_gt(pacc[i][0], pacc[i][1], pacc[i][2], pacc[i][3]);
			ious[i] = iou(bb_test, bb_gt);
		}
		return ious;
	}

	// Orignial NMS
	void NMS(std::vector<Detection>& dets, float threshold) {

		// Sort by score
		std::sort(dets.begin(), dets.end(),
			[](const Detection& a, const Detection& b) { return a.scr > b.scr; });

		// Perform non-maximum suppression
		for (size_t i = 0; i < dets.size(); ++i) {
			// remove_if: for [beg,end) -> remove element where op(elem) == true
			// remove_*() could not remove element only if used with erase.
			dets.erase(std::remove_if(dets.begin() + i + 1, dets.end(),
				[&](const Detection& d) {
					if (dets[i].cls != d.cls)
						return false;
					return iou(dets[i].bbox, d.bbox) > threshold;
				}),
				dets.end());
		}
	}

	// Soft-NMS
	// method {0: "orgignal NMS", 1: "linear", 2: "gaussian"}
	void softNMS(std::vector<Detection> & dets, float threshold,  
		         const int method =0, float sigma=0.5, float epsilon = 0.001) {

			// Sort by score
			std::sort(dets.begin(), dets.end(),
				[](const Detection& a, const Detection& b) { return a.scr > b.scr; });
			
			// https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx
			// Perform non-maximum suppression
			for (size_t i = 0; i < dets.size(); ++i) {
				dets.erase(std::remove_if(dets.begin() + i + 1, dets.end(),
					[&](const Detection& d)
					{
						if (dets[i].cls != d.cls)
							return false;

						float ov = iou(dets[i].bbox, d.bbox);
						float weight=1.0;

						if (method == 1) // "linear"
						{
							if (ov > threshold) weight -= ov;
						} 
						else if (method == 2) // "gaussian"
						{
							weight = exp((ov * ov) / sigma * -1);
						}
						else // original NMS
						{
							if (ov > threshold) weight = 0;
						}
						float newscore = d.scr * weight;

						// if box newscore falls below epsilon, discard the box 
						return newscore < epsilon;
					}),
					dets.end());
			}
	}
	
	//https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/utils.py
	//weighted average bboxs which IOU > threshold, see line: 251-262
	//pred dim: [Tx, Ty, Tw, Th, conf, max_class_score, max_class_indice]
	void weightedNMS(std::vector<Detection> & dets, torch::Tensor & pred, float threshold) {

		// sort desending by class_score
		auto index = pred.select(1, 5).argsort(-1, true);
		pred = pred.index_select(0, index); // reorder 2d tensor

		torch::Tensor score, label, label_match, overlap, invalid;
		torch::Tensor ind, det, noz_ind, weights;

		while (pred.size(0) > 0)
		{	
			label = pred[0][6]; //target class
			score = pred[0][5]; //target score

			//select target class where iou > thres
			label_match = pred.select(1, 6) == label;
			overlap = iou(pred.slice(1, 0, 4)) > threshold;
			invalid = overlap == label_match;	
			ind = invalid.nonzero().squeeze();

			// weight average -> weights*pred[invalid,:4].sum(0) / weights.sum(0)
			weights = pred.index_select(0, ind).select(1, 4).unsqueeze(1);
			det = pred.index_select(0, ind).slice(1, 0, 5).mul(weights).sum(0).div(weights.sum(0));
			
			// keep the weighted bbox
			auto det_acc = det.accessor<float, 1>();
			dets.emplace_back(Detection{ cv::Rect2f(det_acc[0], det_acc[1], det_acc[2], det_acc[3]),  
				                         score.item<float>(),
										 static_cast<int64>(label.item<float>())});
			// select rest bboxes 
			noz_ind = (invalid == 0).nonzero().squeeze_();
			pred = pred.index_select(0, noz_ind);
		}
	}
}

const float Detector::NMS_threshold = 0.4f;
const float Detector::confidence_threshold = 0.6f;

Detector::Detector(const std::array<int64_t, 2> &_inp_dim, 
	               const char* config_file, const char* weight_file, YOLOType type) {
    switch (type) {
    case YOLOType::YOLOv3:
		net = std::make_unique<Darknet>(config_file);
		net->load_weights(weight_file);
        break;
    case YOLOType::YOLOv3_TINY:
		// TODO: change the path of tiny input
        net = std::make_unique<Darknet>("D:\\Github\\YOLOv3_torch\\models\\yolov3-tiny.cfg");
        net->load_weights("D:\\Github\\YOLOv3_torch\\models\\yolov3-tiny.weights");
        break;
    default:
        break;
    }


	if (torch::cuda::is_available()) 
		device_type = torch::kCUDA;
	else 
		device_type = torch::kCPU;
	
	net->to(device_type);
    net->eval();

    inp_dim = _inp_dim;
}

Detector::~Detector() = default;

std::vector<Detection> Detector::predict(cv::Mat image) {

	//turn off ...
    torch::NoGradGuard no_grad;

    int64_t orig_dim[] = {image.rows, image.cols};
	// adjust rows and cols, and pad 128 to blank region
	// resize to model input dim
    image = letterbox_img(image, inp_dim);  

	// BGR2RGB: if models trained in RGB order. Be carefull
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F);

    //TODO: Batch Input
	// TODO: normalization using IMAGENET stats ?
	auto img_tensor = torch::from_blob(image.data, {1, inp_dim[0], inp_dim[1], 3})
	        .permute({0, 3, 1, 2}).div_(255.0).to(device_type);

	// input tensor: [B, C, H, W]
	// output tensor: (batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    auto prediction = net->forward(img_tensor);

	// collect detected objects
	std::vector<Detection> dets;

	//TODO: Batch inference
	// for (auto &pred: prediction){}
	prediction.squeeze_(0);
    auto [bbox, scr, cls] = threshold_confidence(prediction, confidence_threshold);
    
	// NO objects found!
	if (bbox.numel()== 0) 
		return dets;

	// move data to cpu
	bbox = bbox.cpu();
	cls = cls.cpu();
	scr = scr.cpu();

	// Covert to (left, top, weight, height)
	center_to_corner(bbox);
	//Covert back to orig_dim scale
	inv_letterbox_bbox(bbox, inp_dim, orig_dim);

	// if use weighted NMS
	//torch::Tensor pred = torch::cat({ bbox, scr.unsqueeze(1), cls.type_as(scr).unsqueeze(1) }, 1).cpu();
	//weightedNMS(dets, pred, NMS_threshold);

	// if soft or original NMS
	auto bbox_acc = bbox.accessor<float, 2>();
	auto scr_acc = scr.accessor<float, 1>();
	auto cls_acc = cls.accessor<int64, 1>();

	for (int64_t i = 0; i < bbox_acc.size(0); ++i) {
		// Rect2f <- [bx,by, bw, bh]
		auto d = Detection{ cv::Rect2f(bbox_acc[i][0], bbox_acc[i][1], bbox_acc[i][2], bbox_acc[i][3]),
							scr_acc[i], 
			                cls_acc[i]
		                  };
		dets.emplace_back(d);
	}
	//softNMS(dets, NMS_threshold, 2, 0.5, 0.001);
	NMS(dets, NMS_threshold);

    return dets;

}
