#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

const char* windowName = "Test";

using namespace cv;

RNG rng(12345);

void cntr(Mat src, Mat &dst)
{
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Mat drawing = Mat::zeros(src.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); ++i)
    {
        Scalar colour = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, colour, 1, LINE_8, hierarchy, 0);
    }
    dst = drawing;
}

int main(int argc, char const *argv[])
{
    Mat img;
    img = imread(argv[1], IMREAD_COLOR);

    std::cout << (int)img.at<uchar>(Point(0,0)) << std::endl;

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat thresh;
    threshold(gray, thresh, 130, 255, THRESH_BINARY_INV);

    Mat canny;
    Canny(thresh, canny, 140, 140*2);
/*
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Mat drawing = Mat::zeros(thresh.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); ++i)
    {
        Scalar colour = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, colour, 1, LINE_8, hierarchy, 0);
    }
*/
    Mat drawing;
    cntr(thresh, drawing);

    namedWindow(windowName, WINDOW_NORMAL);
    resizeWindow(windowName, 600, 400);
    imshow(windowName, drawing);
    waitKey();

    return 0;
}