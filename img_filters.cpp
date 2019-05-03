#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <iostream>


int main(int argc, char* argv[])
{
	cv::Mat img = cv::imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat eimg,img2;

	cv::namedWindow("Color image",CV_WINDOW_AUTOSIZE);
	cv::imshow("Color image", img);
	
	cv::bilateralFilter(img,eimg,30,120,100);
        cv::namedWindow("Bilateral Filtered",CV_WINDOW_AUTOSIZE);
        cv::imshow("Bilateral Filtered", eimg);
        
	cv::medianBlur(img,img2,5);
        cv::namedWindow("Median Filtered",CV_WINDOW_AUTOSIZE);
        cv::imshow("Median Filtered", img2);


        cv::waitKey(0);
	cv::destroyWindow("Color image");
	cv::destroyWindow("Bilateral Filtered");
	cv::destroyWindow("Median Filtered");


	return 0;
}
