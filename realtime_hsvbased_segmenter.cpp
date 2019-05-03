#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace std;
using namespace cv;

Mat edges,frame,bw,HSV_image, bw_filt;


Scalar hsvlow(0,0,0),hsvhigh(180,255,255);


int main ( int argc, char **argv )
{
    VideoCapture cap;
    cap.open(0);
    if(!cap.isOpened())  
       {
    	cout << "Error opening camera!";
    	return -1; }

    namedWindow("Live image");
    namedWindow("segmented");

    Mat kernel2 = Mat::ones(7,7,CV_8U);
    
    int loop;
    float change;	
   
    
    for(;;)
    {	cv::Mat frame;
    	cap >> frame;
	Scalar hsvlow(0,0,0),hsvhigh(180,255,255);
	
    	cv::imshow("Live image", frame);  
	cvtColor(frame, HSV_image, CV_BGR2HSV);
	imshow("HSV image",HSV_image);

	Vec3b p = HSV_image.at<Vec3b>(frame.rows/2,frame.cols/2);
	
	//cout << hsvlow  << hsvhigh << "\n";
        for (loop=0;loop<3;loop++) { 
                change=p[loop]*0.3; 
		if((loop==0))
           	{   change = p[loop]*0.08;
		    hsvhigh[loop] = p[loop] + change;
		    hsvlow[loop]  = p[loop] - change;	}
           if((loop==1) && (hsvlow[loop]<127))
		{	hsvlow[loop] = 127;}
		if((loop==1) && (hsvlow[loop]>=127))
		{	hsvlow[loop] = p[loop] - change;}
		if((loop==2) && (hsvlow[loop]<150))
		{	hsvlow[loop] = 150;}
		if((loop==2) && (hsvlow[loop]>=150))
		{	hsvlow[loop] = p[loop] - change;}
		
        }
        inRange( HSV_image, hsvlow,hsvhigh,bw);
        imshow("segmented",bw);
	morphologyEx(bw,bw_filt,MORPH_CLOSE,kernel2); 
	imshow("segmented and filled",bw_filt);
    	if(cv::waitKey(90) >= 0)
	{	cv::destroyAllWindows(); 
		break;}
	  
}

    return 0;
}

