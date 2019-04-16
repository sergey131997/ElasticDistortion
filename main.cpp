#include <opencv2/opencv.hpp>

using namespace std;

void ElasticDeformations(const cv::Mat &src,
                         cv::Mat &dst,
                         const cv::Mat &src_grid,
                         cv::Mat &dst_grid,
                         double sigma,
                         double alpha)
{
    cv::Mat dx(src.size(), CV_64FC1);
    cv::Mat dy(src.size(), CV_64FC1);

    cv::randu(dx, cv::Scalar(-1), cv::Scalar(1));
    cv::randu(dy, cv::Scalar(-1), cv::Scalar(1));
    cv::Size kernel_size(sigma*2 + 1, sigma*2 + 1);
    cv::GaussianBlur(dx, dx, kernel_size, sigma);
    cv::GaussianBlur(dy, dy, kernel_size, sigma);
    dx *= alpha;
    dy *= alpha;
    int nCh = src.channels();

    for(int y = 0; y < src.rows; y++)
        for(int x = 0; x < src.cols; x++)
        {
            int org_x = x - dx.at<double>(y, x);
            int org_y = y - dy.at<double>(y, x);

            for(int ch = 0; ch < nCh; ch++)
            {
                dst.data[(y * src.cols + x) * nCh + ch] = src.data[(org_y * src.cols + org_x) * nCh + ch];
                dst_grid.data[(y * src.cols + x) * nCh + ch] = src_grid.data[(org_y * src.cols + org_x) * nCh + ch];
            }
        }
}

void MakeGrid(cv::Mat &grid) {
    int x = grid.size[0] / 10;
    int y = grid.size[1] / 10;
    for (int i = 0; i < grid.size[0]; i+=x)
        cv::line(grid, cv::Point(0,i), cv::Point(grid.size[1], i), cv::Scalar(255, 255, 255));
    for (int i = 0; i < grid.size[1]; i+=y)
        cv::line(grid, cv::Point(i,0), cv::Point(i, grid.size[0]), cv::Scalar(255, 255, 255));
}

int main()
{
    cv::Mat imageBig=cv::imread("/home/sergey/my_first_net.json.png");
    cv::Mat image;
    cv::resize(imageBig, image, cv::Size(680, 420), 0, 0, cv::INTER_AREA);

    int dim(256);
    cv::Mat lut(1, &dim, CV_8U);
    cv::namedWindow("Example");
    int current_alpha = 0;
    int current_sigma = 0;
    cv::createTrackbar("Alpha", "Example", &current_alpha, 60);
    cv::createTrackbar("Sigma", "Example", &current_sigma, 600);
    while (1) {
        cv::Mat image_clone(image.rows, image.cols, image.type());
        cv::Mat grid(image.rows, image.cols, image.type());
        grid.setTo(cv::Scalar(0, 0, 0));
        cv::Mat grid_clone(image.rows, image.cols, image.type());
        cv::Mat two_images(image.rows * 2, image.cols * 2, image.type());
        cv::Mat roi_left_up = two_images(cv::Rect(0, 0, image.cols, image.rows));
        cv::Mat roi_right_up = two_images(cv::Rect(image.cols, 0, image.cols, image.rows));
        cv::Mat roi_left_down = two_images(cv::Rect(0, image.rows, image.cols, image.rows));
        cv::Mat roi_right_down = two_images(cv::Rect(image.cols, image.rows, image.cols, image.rows));
        MakeGrid(grid);
        ElasticDeformations(image, image_clone, grid, grid_clone, current_alpha, current_sigma /20.0);
        image.copyTo(roi_left_up);
        image_clone.copyTo(roi_right_up);
        grid.copyTo(roi_left_down);
        grid_clone.copyTo(roi_right_down);
        cv::imshow("Example", two_images);
        char c = cv::waitKey(33);
        if (c == 27)
            break;
    }
    cv::destroyWindow("Example");
}
