#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <math.h>


void write_text_to_log_file(const std::string& text)
{
    std::ofstream log_file(
        "log_file.txt", std::ios_base::out | std::ios_base::app);
    log_file << text << std::endl;
}



template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
    return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}


static void softmax(float* input, int input_len)
{
    assert(input != NULL);
    assert(input_len != 0);
    int i;
    float m;
    /* Find maximum value from input array */
    m = input[0];
    for (i = 1; i < input_len; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    float sum = 0;
    for (i = 0; i < input_len; i++) {
        sum += expf(input[i] - m);
    }

    for (i = 0; i < input_len; i++) {
        input[i] = expf(input[i] - m - log(sum));

    }
}



using namespace cv;
using namespace std;



struct Base
{
    string model_path = "net.onnx";
    double scale = 1 / 255.;
    int w = 224;
    int h = 224;
    const char* labels[6] = {"Anger", "Contempt", "Fear", "Happy", "Neutral", "Sad"};
    int num_classes = 6;


};

void show_image(string path) {
    Mat image = imread(path);
    imshow("Image", image);
    waitKey(0);
}




int predicted_clases(cv::Mat outputs, int num_classes) {
     float* out = (float*)outputs.data;
     softmax(out, num_classes);
     std::vector<float> out_softmax = {out, out + num_classes};
     int indexes = arg_max(out_softmax);

     for (int i = 0; i < out_softmax.size(); i++) {
         cout << "prob : " << out_softmax[i]*100 << "\n";


     }

     return indexes;}


int main(int argc, char* argv[]) {


    Base conf;  
    auto net = cv::dnn::readNet(conf.model_path);
    cv::Mat frame, image_blob;
    frame = imread(argv[1]);

    cv::dnn::blobFromImage(frame, image_blob, conf.scale, cv::Size(conf.w, conf.h), cv::Scalar(), true, false, CV_32F);
    net.setInput(image_blob);
    Mat outputs = net.forward();
    int indexes = predicted_clases(outputs, conf.num_classes);
    cout << conf.labels[indexes] << endl;

    write_text_to_log_file(conf.labels[indexes]);

    return 0;
}