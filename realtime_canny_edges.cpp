#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int, char**)
{
    cv::VideoCapture cap; 
    cap.open(0);
    if(!cap.isOpened())  
       {
     cout << "Error opening camera!";
     return -1;
 }

    cv::Mat edges;
    for(;;)
    {
        cv::Mat frame;
        cap >> frame; 
        cv::cvtColor(frame, edges, CV_BGR2GRAY);
        cv::GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        cv::Canny(edges, edges, 0, 30, 3);
        cv::imshow("Live image", frame);
        cv::imshow("Edges", edges);
        if(waitKey(10) >= 0)
	{	cv::destroyAllWindows(); 
		break;}
    }
    
    return 0;
}