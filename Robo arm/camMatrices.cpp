#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <sstream>
#include <stdio.h>


#define PI 3.14159265

using namespace cv;
using namespace std;

void mouseEvent (int evt, int x, int y, int flags, void*);

int main(int, char**)
{
    VideoCapture cap; 
    cap.open(0);
    
    Mat imgGray,edges;	
    int ddepth = CV_16S; 
    int delta = 0;
    int scale = 1;
    //-----------------------Set Camera Properties------------------------------------------------------------------
    cap.set(CV_CAP_PROP_FOCUS, 0);
    cap.set(CV_CAP_PROP_BRIGHTNESS, -15);
    cap.set(CV_CAP_PROP_SATURATION, 200);
    cap.set(CV_CAP_PROP_CONTRAST, 10);

    if(!cap.isOpened())  
       {
        cout << "Error opening camera!";
        return -1;
       }

    //-------------------Camera Calibration Matrices-----------------------------------------------------------------------------------------------------------
    Mat cameraMatrix = Mat::eye(3,3,CV_64F);
    cameraMatrix.at<double>(0,0) = 6.0486080554129342e+002;   cameraMatrix.at<double>(0,1) = 0;     cameraMatrix.at<double>(0,2) = 3.1950000000000000e+002;
    cameraMatrix.at<double>(1,0) = 0;   cameraMatrix.at<double>(1,1) = 6.0486080554129342e+002;     cameraMatrix.at<double>(1,2) = 2.3950000000000000e+002;
    cameraMatrix.at<double>(2,0) = 0;   cameraMatrix.at<double>(2,1) = 0;     cameraMatrix.at<double>(2,2) = 1;
        
    Mat distCoeffs = Mat::zeros(5,1,CV_64F);
    distCoeffs.at<double>(0,0) = 8.2620139698452666e-002; distCoeffs.at<double>(1,0) = -2.7675886003881384e-001; distCoeffs.at<double>(2,0) = 0; 
    distCoeffs.at<double>(3,0) = 0; distCoeffs.at<double>(4,0) = 5.9528994991108919e-001;
    
  

    for(;;)
    {
        Mat frame,framed,imgfilt, imgc;
        cap >> framed; 
	int rcol = 640; int rlin = 480;
	int x1 = 60; int y1 = 0;
        int Bolts = 0, Nuts = 0;
	Rect roi(x1,y1,rcol - 150,rlin - 200);
	Point2d aux, auxtemp; Point2d pt1 = Point2d(226,0); Point2d pt2 = Point2d(226,280);	
	Point2d pt3 = Point2d(0,3); Point2d pt4 = Point2d(490,3);


	undistort(framed,frame,cameraMatrix,distCoeffs,cameraMatrix);	
	
	//--------Edge Detection-------------------------------------------------
	frame = frame(roi);
	flip(frame,frame,-1);
        cvtColor(frame, imgGray, CV_BGR2GRAY);
        GaussianBlur(imgGray, imgGray, Size(3,3), 0, 0);

	Mat gdx,gdy,abs_gdx,abs_gdy;
	Sobel(imgGray, gdx, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(gdx,abs_gdx);

	Sobel(imgGray, gdy, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(gdy,abs_gdy);

	addWeighted(abs_gdx,0.5,abs_gdy,0.5, 0, edges);
	threshold(edges,edges,50,255,1);
	bitwise_not(edges,edges);

       	namedWindow("Live image", WINDOW_NORMAL);
        
        imshow("Edges", edges);

	cv::line(frame, pt1, pt2, Scalar(250,250,250));
	cv::line(frame, pt3, pt4, Scalar(250,250,250));
	
	imshow("Live image", frame);
	
	setMouseCallback( "Live image", mouseEvent, &frame );

	if(waitKey(4) >= 0)
	{	cv::destroyAllWindows(); 
		break;}
    }

    return 0;
}

void mouseEvent (int evt, int x, int y, int flags, void*)
{
	if(evt == CV_EVENT_LBUTTONDOWN)
     {  	
		cout << "[ " << x << ", " << y << "]\n";
				
	}
}
