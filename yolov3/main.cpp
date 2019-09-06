#include <iostream>
#include <sstream>
#include <chrono>
#include <string>

#include <opencv2/opencv.hpp>
#include "../detection/inc/Detector.h"
#include "util.h"
using namespace std;

int main(int argc, const char *argv[]) {

    //if (argc < 3 || argc > 4) {
    //    throw runtime_error("usage: yolov3 <config_file> <weights> <image_folder> ");
    //}

	std::cout << "Load weights ..." << std::endl;
	const char* config = argv[1];
	const char* weights = argv[2];

	std::array<int64_t, 2> inp_dim = { 416, 416 }; 	//model input image size (row, column)
	Detector detector(inp_dim, config, weights);

	// using opencv's glob function to get file list in a folder
	std::vector<std::string> filenames;
	std::string path = argv[3];
	path = path+"\\*.jpg";
	cv::glob(path, filenames, false);

	std::cout << "Start to inference ..." << std::endl;
	auto start = std::chrono::steady_clock::now();
	std::vector<cv::Mat> images;
	std::vector<std::vector<Detection>> results;
	// read images and inference
	for (auto& img : filenames) {
		cv::Mat origin_image = cv::imread(img, cv::IMREAD_COLOR);
		images.push_back(origin_image);
		auto dets = detector.predict(origin_image); 
		results.push_back(dets);
	}
	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Average time per image: " << duration / images.size() << "ms" << std::endl;

	//auto classnames = read_txt("D:\\Github\\YOLOv3_torch\\models\\classnames.txt");

	for (size_t i = 0; i < results.size(); i++) {

		auto res = results[i];
		auto imgclone = images[i].clone();
		std::string outname(filenames[i]);
		std::cout << outname << std::endl;
		//print class_index if objects found!
		for (auto& d : res) {
			std::cout << d.cls <<" "<< d.scr << std::endl;	
			// draw box on images		
			draw_bbox(imgclone, d.bbox, std::to_string(d.cls), color_map(d.cls));
			//draw_bbox(imgclone, d.bbox, classnames[d.cls], color_map(d.cls));
		}
		cv::imwrite(outname.replace(outname.find("jpg"), 3, "bbox.jpg"), imgclone);
	}

    std::cout << "Done" << std::endl;

	////For camera 
	//std::string input_path = "Current";
	//cv::VideoCapture cap(input_path);
	//if (!cap.isOpened()) {
	//	throw runtime_error("Cannot open cv::VideoCapture");
	//}

	////array<int64_t, 2> orig_dim{ int64_t(cap.get(cv::CAP_PROP_FRAME_HEIGHT)), int64_t(cap.get(cv::CAP_PROP_FRAME_WIDTH)) };
	////array<int64_t, 2> inp_dim;
	//cv::Mat image;
	//cv::namedWindow("Real time Object Detection", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
	//while (cap.read(image)) {
	//	auto frame_processed = static_cast<uint32_t>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;
	//	auto dets = detector.detect(image);

	//	for (auto& d : dets) {
	//		//if (std::is_empty(d)) {
	//		//	std::cout << "no object found in: " << filenames[i] << std::endl;
	//		//}
	//		std::cout << d.cls << " " << d.scr << std::endl;
	//		// draw box onto images		
	//		draw_bbox(image, d.bbox, std::to_string(d.cls), color_map(d.cls));
	//	}
	//	cv::imshow("Output", image);
	//	switch (cv::waitKey(1) & 0xFF) {
	//	case 'q':
	//		return 0;
	//	case ' ':
	//		cv::imwrite(to_string(frame_processed) + ".jpg", image);
	//		break;
	//	default:
	//		break;
	//	}
	//}

	 return 0;
}

