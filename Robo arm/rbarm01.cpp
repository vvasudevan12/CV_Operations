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

double axisProperties(vector<Point> contour);


int main(int, char**)
{
    VideoCapture cap; 
    cap.open(0);
    
    Mat imgGray,edges;	
    int ddepth = CV_16S; 
    int delta = 0;
    int scale = 1;
    vector< vector<Point> > contours, contours2;
    vector <Point> shapeBolt2(25); vector <Point> shapeBolt1(36);
    vector <Point> shapeNut(28);
    vector <Vec4i> hierarchy; 
	
	//----------------Shape definitions----------------------------------------------------------------------------
    shapeBolt2[0] = Point(274,138);     shapeBolt2[1] = Point(274,140);	
    shapeBolt2[2] = Point(273,141);     shapeBolt2[3] = Point(273,145);	
    shapeBolt2[4] = Point(274,146);     shapeBolt2[5] = Point(274, 169);	
    shapeBolt2[6] = Point(273, 170);     shapeBolt2[7] = Point(273, 179);	
    shapeBolt2[8] = Point(272, 180);     shapeBolt2[9] = Point(272, 183);	
    shapeBolt2[10] = Point(273, 184);     shapeBolt2[11] = Point(273, 185);	
    shapeBolt2[12] = Point(274, 186);     shapeBolt2[13] = Point(279, 186);	
    shapeBolt2[14] = Point(280, 185);    shapeBolt2[15] = Point(280, 180);	
    shapeBolt2[16] = Point(281, 179);    shapeBolt2[17] = Point(281, 165);	
    shapeBolt2[18] = Point(282, 164);    shapeBolt2[19] = Point(282, 149);	
    shapeBolt2[20] = Point(283, 148);    shapeBolt2[21] = Point(283, 147);	
    shapeBolt2[22] = Point(285, 145);    shapeBolt2[23] = Point(285, 139);	
    shapeBolt2[24] = Point(284, 138);    	
    
    shapeBolt1[0] = Point(247, 139);     shapeBolt1[1] = Point(246, 140);	
    shapeBolt1[2] = Point(243, 140);     shapeBolt1[3] = Point(243, 141);	
    shapeBolt1[4] = Point(242, 142);     shapeBolt1[5] = Point(242, 145);	
    shapeBolt1[6] = Point(243, 146);     shapeBolt1[7] = Point(244, 146);	
    shapeBolt1[8] = Point(245, 147);     shapeBolt1[9] = Point(245, 151);	
    shapeBolt1[10] = Point(246, 152);     shapeBolt1[11] = Point(246, 173);	
    shapeBolt1[12] = Point(245, 174);     shapeBolt1[13] = Point(245, 182);	
    shapeBolt1[14] = Point(246, 183);    shapeBolt1[15] = Point(246, 184);	
    shapeBolt1[16] = Point(247, 185);    shapeBolt1[17] = Point(247, 186);	
    shapeBolt1[18] = Point(248, 187);    shapeBolt1[19] = Point(252, 187);	
    shapeBolt1[20] = Point(253, 186);    shapeBolt1[21] = Point(253, 185);	
    shapeBolt1[22] = Point(255, 183);    shapeBolt1[23] = Point(255, 166);	
    shapeBolt1[24] = Point(256, 165);    shapeBolt1[25] = Point(256, 157);	
    shapeBolt1[26] = Point(257, 156);    shapeBolt1[27] = Point(257, 148);	
    shapeBolt1[28] = Point(258, 147);    shapeBolt1[29] = Point(260, 147);	
    shapeBolt1[30] = Point(261, 146);    shapeBolt1[31] = Point(261, 142);	
    shapeBolt1[32] = Point(260, 141);    shapeBolt1[33] = Point(260, 140);	
    shapeBolt1[34] = Point(257, 140);    shapeBolt1[35] = Point(256, 139);	
    

    shapeNut[0] = Point(247,119);
    shapeNut[1] = Point(245,121);
    shapeNut[2] = Point(245,122);
    shapeNut[3] = Point(244,123);
    shapeNut[4] = Point(244,124);
    shapeNut[5] = Point(243,125);
    shapeNut[6] = Point(243,131);
    shapeNut[7] = Point(244,131);
    shapeNut[8] = Point(245,132);
    shapeNut[9] = Point(245,134);
    shapeNut[10] = Point(246,133);
    shapeNut[11] = Point(248,135);
    shapeNut[12] = Point(249,135);
    shapeNut[13] = Point(250,136);
    shapeNut[14] = Point(251,135);
    shapeNut[15] = Point(252,135);
    shapeNut[16] = Point(254,133);
    shapeNut[17] = Point(254,131);
    shapeNut[18] = Point(251,128);
    shapeNut[19] = Point(252,127);
    shapeNut[20] = Point(253,127);
    shapeNut[21] = Point(254,126);
    shapeNut[22] = Point(255,126);
    shapeNut[23] = Point(257,124);
    shapeNut[24] = Point(257,123);
    shapeNut[25] = Point(258,122);
    shapeNut[26] = Point(258,121);
    shapeNut[27] = Point(256,119);
 
    
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
	double theta1, theta2, angle;
	Point2d aux, auxtemp; Point2d pt1 = Point2d(226,0); Point2d pt2 = Point2d(226,280);	
	Point2d pt3 = Point2d(0,3); Point2d pt4 = Point2d(490,3);
	float RWx , RWy;

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

       	
        namedWindow("Edges", WINDOW_NORMAL);
        imshow("Edges", edges);
	
	//-------------Morphological Operations------------------------------------------
	Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(-1,-1));
	morphologyEx(edges,imgfilt,MORPH_OPEN,element);
	morphologyEx(imgfilt,imgfilt,MORPH_CLOSE,element,Point(-1,-1),3);
	dilate(imgfilt,imgfilt, getStructuringElement(MORPH_ELLIPSE, Size(1,1)));
	
	


	//-------------------Contour Operations-------------------------------------------------
	findContours(imgfilt.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++){
		double area  = contourArea(contours[i]);
		
		if (area > 0 && area <= 200)
			drawContours(imgfilt, contours, i, CV_RGB(0,0,0), -1);
	}
	
	//----------------------Classifier---------------------------------------------------------------
	findContours(imgfilt.clone(), contours2, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	
	vector<Point2d> massCenter(contours2.size());
	vector<Moments> moment(contours2.size());
	
	for (int i = 0; i < contours2.size(); i++)
	{
		moment[i] = moments(contours2[i], false);
		massCenter[i] = Point2d(moment[i].m10/moment[i].m00, moment[i].m01/moment[i].m00);
	}


	if (!contours2.empty() && !hierarchy.empty())
	{
		aux = Point2d(0,0); auxtemp = Point2d(0,0);

		int idx = 0;
		for(; idx >= 0 ; idx = hierarchy[idx][0])
		{

			double compareBolt1 = matchShapes(contours2[idx], shapeBolt1, CV_CONTOURS_MATCH_I3, 0.0);
			double compareBolt2 = matchShapes(contours2[idx], shapeBolt2, CV_CONTOURS_MATCH_I3, 0.0);
			if ((compareBolt1 < 0.55) || (compareBolt2 < 0.55) && (axisProperties(contours2[idx]) >= 2.50) && (axisProperties(contours2[idx]) < 10) && (arcLength(contours2[idx], true) >= 100) && (arcLength(contours2[idx], true) <= 190))
				Bolts = Bolts + 1;
			double compareNut = matchShapes(contours2[idx], shapeNut, CV_CONTOURS_MATCH_I3, 0.0);	
			if ((compareNut < 0.55)&&((axisProperties(contours2[idx]) < 1.3) && (arcLength(contours2[idx], true) <290)))
				Nuts = Nuts + 1;
			
			aux.x = 0; aux.y = 0;
			
			aux.x = (-1 * (massCenter[idx].x - 250));
			aux.y = ((massCenter[idx].y - 5 ));
			
			//Real World Coordinate Conversion
			RWx = massCenter[idx].x*0.0977505112474437 + massCenter[idx].y*0 + 0;
			RWy = massCenter[idx].x*0.0001992745510725 + massCenter[idx].y*0.09744525547445255 - 0.09744525547445255;
			
			
			auxtemp.x = (-1 * (massCenter[idx].x - 226));
			auxtemp.y = ((massCenter[idx].y + 40 ));
			
			/*if(auxtemp.x < 0 && auxtemp.y < 180)
			{	aux.x = auxtemp.x*0.81+5; aux.y = auxtemp.y-5; }
			else if (auxtemp.x > 0 && auxtemp.y <180)
			{	aux.x = auxtemp.x*0.8;  aux.y =auxtemp.y*1.05; }
  			else if (auxtemp.x > 0 && auxtemp.y >180)
			{	aux.x = auxtemp.x*0.85;  aux.y =auxtemp.y*1.04; }
  			else if (auxtemp.x < 0 && auxtemp.y >180)
			{	aux.x = auxtemp.x*0.90;  aux.y =auxtemp.y; }*/
  
			aux.x = aux.x*(46.8/490); aux.y = aux.y*(26.7/280);

			string xy = "[" + to_string(cvRound(aux.x)) + ", " + to_string(cvRound(aux.y)) + "]";
			string RWxy = "[" + to_string(RWx) + ", " + to_string(RWy) + "]";
			
			Point2f circleCenter;
			float circleRadius;
			minEnclosingCircle(contours2[idx], circleCenter, circleRadius);
			
			if ((PI*circleRadius*circleRadius > 2500 && axisProperties(contours2[idx]) < 3.5)||(PI*circleRadius*circleRadius > 940 && axisProperties(contours2[idx]) < 2.5)||(PI*circleRadius*circleRadius > 2000 && axisProperties(contours2[idx]) < 3.0))
			{
				putText(frame, "Overlapped" + xy, massCenter[idx], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100,250,100,255), 2);
				putText(frame, RWxy, Point2d(massCenter[idx].x,massCenter[idx].y + 14), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100,100,250,255), 2);
				//cout << PI*circleRadius*circleRadius << "\n";
				//cout << (axisProperties(contours2[idx])) << "\n";
			}
			else if ((axisProperties(contours2[idx]) >= 2.50) && (axisProperties(contours2[idx]) < 10) && (arcLength(contours2[idx], true) >= 100) && (arcLength(contours2[idx], true) <= 190))	
			{
				RotatedRect rec = fitEllipse(contours2[idx]);
				theta1 = atan(auxtemp.y/auxtemp.x) * 180/ PI;
				theta2 = (rec.angle);
				angle = theta1 - (90 - theta2);
				if (angle < 0)
					angle = 180 - abs(angle);
				else if(angle > 180)
					angle = abs(angle) - 180;

				string xy = "[" + to_string(cvRound(aux.x)) + ", " + to_string(cvRound(aux.y)) + ", " + to_string(cvRound(angle)) + "deg]";

				string test = to_string(cvRound(theta1)) + ", " + to_string(cvRound(theta2)) + ", " + to_string(cvRound(angle));
				putText(frame, "Bolt" + xy, massCenter[idx], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100,250,100,255), 2);
				putText(frame, RWxy, Point2d(massCenter[idx].x,massCenter[idx].y + 14), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100,100,250,255), 2);
			}
			else if ((axisProperties(contours2[idx]) < 1.3) && (arcLength(contours2[idx], true) <290))
			{
				putText(frame, "Nut" + xy, massCenter[idx], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100,250,100,255), 2);
				putText(frame, RWxy, Point2d(massCenter[idx].x,massCenter[idx].y + 14), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100,100,250,255), 2);
				angle = 0;
			}
	 		else
			{
				putText(frame, "Unknown Object" + xy, massCenter[idx], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100,250,100,255), 2);
				putText(frame, RWxy, Point2d(massCenter[idx].x,massCenter[idx].y + 14), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100,100,250,255), 2);
				//cout << PI*circleRadius*circleRadius << "\n";
				//cout << (axisProperties(contours2[idx])) << "\n";
			}
		}
	}
	
	cv::line(frame, pt1, pt2, Scalar(250,250,250));
	cv::line(frame, pt3, pt4, Scalar(250,250,250));
	namedWindow("Live image", WINDOW_NORMAL);
	imshow("Live image", frame);

	namedWindow("ALL", WINDOW_NORMAL);
	threshold(imgfilt,imgc, 0, 255, 0);
	imshow("ALL", imgc);
	//cout << "Number of Bolts = " << Bolts << ",";
        //cout << "Number of Nuts = " << Nuts << "\n";
		
        if(waitKey(4) >= 0)
	{	cv::destroyAllWindows(); 
		break;}
    }

    return 0;
}


////--------------------------------------------------------------------------------------------------------------------------------/////

double axisProperties(vector<Point> contour)
{
	float majorAxis, minorAxis;
	Point center;
	if (contour.size()>=5)
	{	RotatedRect minEllipse = fitEllipse(contour);
		majorAxis = max(minEllipse.size.height, minEllipse.size.width);	
		minorAxis = min(minEllipse.size.height, minEllipse.size.width);	
		
		return (majorAxis/minorAxis);    }
}

///----------------------------------MEASURES FOR REAL WORLD COORDINATE-----------------------------------------------------------------------------------------------------//
/*
Origin at left top corner (near Robo Arm) ----> Coordinates with respect to bottom left corner (away from Robo Arm) = [6.5cm , 26.8cm]
Rough sketch>    --------0rigin---------------Robo Arm----------------------B
		 |         |                                                |
	26.8cm ->|	   |                                                | 
		 |         |                                                |
		 |         |                                                | 
		 |--6.5cm--C------------------------------------------------D	

O ---> [0,1] pixels =  [0,0] cm
B ---> [489,0] pixels =  [47.8 , 0] cm
C ---> [0,276] pix  =  [0,26.3] cm
D ---> [489,274] pix =  [47.8, 26.7] cm

[ X ; Y ] = [ a00 a01 ; a10 a11] * [ u ; v ] + [tx ; ty] 

*/








