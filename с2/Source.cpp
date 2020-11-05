#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include"time.h"
#include "math.h"
#include <algorithm>

using namespace cv;
using namespace std;
int clamp(int v, int max, int min)
{
    if (v > max) return max;
    else if (v < min) return min;
    return v;
}
float calculateAVG(Mat photo, int x, int y, int rgb, int radius) {
    float returnPC = 0;
    int size = 2 * radius + 1;
    float* vector = new float[size * size];


    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int idx = (i + radius) * size + j + radius;
            vector[idx] = photo.at<Vec3b>(clamp(x + j, photo.rows - 1, 0), clamp(y + i, photo.cols - 1, 0))[rgb];
        }
    }
    std::sort(&vector[0], vector + size * size, std::greater<float>());
    /* for (int i = 0; i < size*size; i++)
     {
         cout << vector[i] << "||";
     }
     cout << endl;
     cout << ((vector[0] + vector[(size * size - 1)]) / 2) << endl;*/
    return ((vector[0] + vector[(size * size - 1)]) / 2);

}
Mat avg_point(Mat& photo, int radius) {

    for (int x = 0; x < photo.rows; x++) {
        for (int y = 0; y < photo.cols; y++) {
            photo.at<Vec3b>(x, y)[0] = calculateAVG(photo, x, y, 0, radius);//b
            photo.at<Vec3b>(x, y)[1] = calculateAVG(photo, x, y, 1, radius);//g
            photo.at<Vec3b>(x, y)[2] = calculateAVG(photo, x, y, 2, radius);//r
        }
    }

    return photo;
}
float calculatePIC(Mat photo, int x, int y, int rgb, int radius, int sigma) {
    float returnPC = 0;
    int size = 2 * radius + 1;
    float* vector = new float[size * size];
    float norm = 0;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int idx = (i + radius) * size + j + radius;
            photo.at<Vec3b>(clamp(x + j, photo.rows - 1, 0), clamp(y + i, photo.cols - 1, 0))[0];
            vector[idx] = exp(-(i * i + j * j) / (sigma * sigma));
            norm += vector[idx];
        }
    }
    //norm += 1;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            vector[i * size + j] /= norm;
        }
    }
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int idx = (i + radius) * size + j + radius;
            returnPC += photo.at<Vec3b>(clamp(x + j, photo.rows - 1, 0), clamp(y + i, photo.cols - 1, 0))[rgb] * vector[idx];
        }
    }
    return returnPC;

}

Mat Gaussian_blur_filter(Mat& photo, int radius, int sigma) {

    for (int x = 0; x < photo.rows; x++) {
        for (int y = 0; y < photo.cols; y++) {
            photo.at<Vec3b>(x, y)[0] = calculatePIC(photo, x, y, 0, radius, sigma);
            photo.at<Vec3b>(x, y)[1] = calculatePIC(photo, x, y, 1, radius, sigma);
            photo.at<Vec3b>(x, y)[2] = calculatePIC(photo, x, y, 2, radius, sigma);
        }
    }

    return photo;
}

int AddGaussianNoise(const Mat src, Mat& dst, double mean = 34.0, double dev = 50.0)
{
    int result = 0;
    if (!(src.empty()))
    {
        Mat Gaus = Mat(src.size(), CV_16SC3);
        randn(Gaus, Scalar::all(mean), Scalar::all(dev));

        for (int Rows = 0; Rows < src.rows; Rows++)
        {
            for (int Cols = 0; Cols < src.cols; Cols++)
            {
                Vec3b src_pix = src.at<Vec3b>(Rows, Cols);
                Vec3b& ds_pix = dst.at<Vec3b>(Rows, Cols);
                Vec3s ns_pix = Gaus.at<Vec3s>(Rows, Cols);

                for (int i = 0; i < 3; i++)
                {
                    int Dest_Pixel = src_pix.val[i] + ns_pix.val[i];
                    ds_pix.val[i] = clamp(Dest_Pixel, 255, 0);
                }
            }
        }
        result = 1;
    }
    return result;
}

int  main()
{
    time_t t1, t2, t3, t4, t5, t6;

    Mat src = imread("ss.jpg");
    imshow("src", src);//source
    Mat dst = src;
    AddGaussianNoise(src, dst);
    // dst - pic with noise
    imshow("picture with Gaussian noise", dst);
    Mat dst2 = src;
    t1 = clock();
    cv::medianBlur(dst, dst2, 5);
    t2 = clock();
    auto delta1 = t2 - t1;
    cout << "cv::median " << delta1 << endl;
    imshow("cv::medianBlur", dst2);
    Mat pic = dst;
    t3 = clock();
    auto gaus = Gaussian_blur_filter(dst, 3, 50);
    t4 = clock();
    cout << "GAUS: " << (t4 - t3) << endl;
    imshow("Gaussian filtr", gaus);
    imshow("filtr averege point", avg_point(pic, 2));
    waitKey();

}