#include <iostream>
#include <sstream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<ctime>
#include<cmath>
using namespace cv;
using namespace std;


int k = 4,M=4, width, height;


int main()
{
    double* sd_init, * mean, * w,*rank,*u_diff;
    int* rank_index;
    int match,min_index=0,rand_index_temp,z;
    double sd = 3, temp,rand_temp;
    double alpha=0.01,p;
    VideoCapture cap;
    cap.open(R"(E:\jetbrain\Cpp_Project\sample.avi)");
    Mat fram, current;
    double thresh = 0.25,sigma = 6;
    cap >> fram;
    width = fram.cols;
    height = fram.rows;
    Mat back(height, width, CV_8UC1, Scalar::all(0));
    Mat foge(height, width, CV_8UC1, Scalar::all(0));
    namedWindow("back", WINDOW_AUTOSIZE);
    namedWindow("foge", WINDOW_AUTOSIZE);
    sd_init = new double[width * height * k];
    mean = new double[width * height * k];
    w = new double[width * height * k];
    u_diff = new double[k];
    rank = new double[k];
    RNG rng;
    for (int i = 0;i < height;i++)
    {
        for (int j = 0;j < width;j++)
        {
            for (int d = 0;d < k;d++)
            {
                mean[i * width * k + j * k + d] = rng.uniform(0, 256);
                sd_init[i * width * k + j * k + d] = sigma;
                w[i * width * k + j * k + d] = double(1) / k;
            }
        }
    }

    while (true)
    {
        cvtColor(fram, current, COLOR_BGR2GRAY);
        rank_index = new int[k];
        for (int i = 0;i < height;i++)
        {
            for (int j = 0;j < width;j++)
            {
                match = 0;
                temp = 0;
                for (int d = 0;d < k;d++)
                {
                    u_diff[d] = abs(int(current.at<uchar>(i, j)) - mean[i * width * k + j * k + d]);
                    //double differs = abs(int(current.at<uchar>(i, j)) - mean[i * width * k + j * k + d]);
                    if (u_diff[d] < sd * sd_init[i * width * k + j * k + d])
                    {
                        //back.at<uchar>(i, j) = current.at<uchar>(i, j);
                        match = 1;
                        w[i * width * k + j * k + d] = (1 - alpha) * w[i * width * k + j * k + d] + alpha * match;
                        p = alpha / w[i * width * k + j * k + d];
                        mean[i * width * k + j * k + d] = (1 - p) * mean[i * width * k + j * k + d] + p * current.at<uchar>(i, j);
                        sd_init[i * width * k + j * k + d] = sqrt((1 - p) * pow(sd_init[i * width * k + j * k + d],2) + p * pow(u_diff[d], 2));
                    }
                    else
                    {
                        w[i * width * k + j * k + d] = (1 - alpha) * w[i * width * k + j * k + d];

                    }
                    temp += w[i * width * k + j * k + d];
                }
                for (int d = 0;d < k;d++)
                {
                    w[i * width * k + j * k + d] = w[i * width * k + j * k + d] / temp;
                }
                temp = w[i * width * k + j * k];
                back.at<uchar>(i, j) = 0;
                for (int d = 0;d < k;d++)
                {
                    back.at<uchar>(i, j) = int(back.at<uchar>(i, j))+int(mean[i * width * k + j * k + d] * w[i * width * k + j * k + d]);
                    if (w[i * width * k + j * k + d] <= temp)
                    {
                        min_index = d;
                        temp = w[i * width * k + j * k + d];
                    }
                    rank_index[d] = d;
                }

                if (match == 0)
                {
                    mean[i * width * k + j * k + min_index] = int(current.at<uchar>(i, j));
                    sd_init[i * width * k + j * k + min_index] = sigma;
                }
                for (int d = 0;d < k;d++)
                {
                    rank[d] = w[i * width * k + j * k + d] / sd_init[i * width * k + j * k + d];
                }

                for (int d = 1;d < k;d++)
                {
                    for (int m = 0;m < d;m++)
                    {
                        if (rank[d] > rank[m])
                        {
                            rand_temp = rank[d];
                            rank[d] = rank[m];
                            rank[m] = rand_temp;

                            rand_index_temp = rank_index[d];
                            rank_index[d] = rank_index[m];
                            rank_index[m] = rand_index_temp;
                        }
                    }
                }

                //if (match == 0)
                //{
                //	foge.at<uchar>(i, j) = current.at<uchar>(i, j);
                //}
                //else
                //{
                //	foge.at<uchar>(i, j) = 0;
                //}

                match = 0, z = 0;
                while ((match == 0) && (z < M))
                {
                    if (w[i * width * k + j * k + rank_index[z]] >= thresh)
                    {
                        if (u_diff[rank_index[z]] <= sd * sd_init[i * width * k + j * k + rank_index[z]])
                        {
                            foge.at<uchar>(i, j) = 0;
                            match = 1;
                        }
                    }
                    else
                    {
                        foge.at<uchar>(i, j) = current.at<uchar>(i, j);
                    }
                    z = z + 1;
                }
            }
        }
        medianBlur(foge, foge, 3);
        imshow("foge", foge);
        imshow("back", fram);
        cap >> fram;
        Mat foge(height, width, CV_8UC1, Scalar::all(0));
        char s = waitKey(33);
        if (s == 27)
        {
            break;
        }
        delete[] rank_index;
    }
    delete[]sd_init;delete[]mean;delete[]w;delete[]u_diff;delete[]rank;
    destroyAllWindows();
    return 0;
}