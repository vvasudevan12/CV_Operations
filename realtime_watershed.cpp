#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class WatershedSegmenter{
private:
    cv::Mat markers;
public:
    void setMarkers(cv::Mat& markerImage)
    {
        markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(cv::Mat &image)
    {
        cv::watershed(image, markers);
        markers.convertTo(markers,CV_8U);
        return markers;
    }
};


int main(int, char**)
{
    cv::VideoCapture cap; 
    cap.open(0);
    if(!cap.isOpened())  
       {
     cout << "Error opening camera!";
     return -1;
 }


    for(;;)
    {
        cv::Mat frame;
        cap >> frame; 
        cv::imshow("Live image", frame);
       	cv::Mat image = frame;
        cv::Mat blank(image.size(),CV_8U,cv::Scalar(0xFF));
        cv::Mat dest;
       
        cv::Mat markers(image.size(),CV_8U,cv::Scalar(-1));
        //Rect(topleftcornerX, topleftcornerY, width, height);
        //top rectangle
        markers(Rect(0,0,image.cols, 5)) = Scalar::all(1);
        //bottom rectangle
        markers(Rect(0,image.rows-5,image.cols, 5)) = Scalar::all(1);
        //left rectangle
        markers(Rect(0,0,5,image.rows)) = Scalar::all(1);
        //right rectangle
        markers(Rect(image.cols-5,0,5,image.rows)) = Scalar::all(1);
        //centre rectangle
        int centreW = image.cols/4;
        int centreH = image.rows/4;
        markers(Rect((image.cols/2)-(centreW/2),(image.rows/2)-(centreH), centreW, centreH*2.8)) = Scalar::all(2);
        markers.convertTo(markers,CV_BGR2GRAY);
        imshow("markers", markers);

        //Create watershed segmentation object
        WatershedSegmenter segmenter;
        segmenter.setMarkers(markers);
        cv::Mat wshedMask = segmenter.process(image);
        cv::Mat mask;
        convertScaleAbs(wshedMask, mask, 1, 0);
        double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
        bitwise_and(image, image, dest, mask);
        dest.convertTo(dest,CV_8U);

        imshow("final_result", dest);
        

	if(cv::waitKey(30) >= 0)
	{	cv::destroyAllWindows(); 
		break;}
	
    }
    
    return 0;
}