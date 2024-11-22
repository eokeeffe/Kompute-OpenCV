
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <kompute/Kompute.hpp>
#include <shader/my_shader.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void MatType( cv::Mat inputMat )
{
    int inttype = inputMat.type();

    std::string r, a;
    uchar depth = inttype & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (inttype >> CV_CN_SHIFT);
    switch ( depth ) {
        case CV_8U:{r = "8U";   a = "Mat.at<uchar>(y,x)"; break;}  
        case CV_8S:{  r = "8S";   a = "Mat.at<schar>(y,x)"; break;}  
        case CV_16U:{ r = "16U";  a = "Mat.at<ushort>(y,x)"; break;} 
        case CV_16S:{ r = "16S";  a = "Mat.at<short>(y,x)"; break; }
        case CV_32S:{ r = "32S";  a = "Mat.at<int>(y,x)"; break; }
        case CV_32F:{ r = "32F";  a = "Mat.at<float>(y,x)"; break;} 
        case CV_64F:{ r = "64F";  a = "Mat.at<double>(y,x)"; break;} 
        default:{     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break; }
    }   
    r += "C";
    r += (chans+'0');
    std::cout << "Mat is of type " << r << " and should be accessed with " << a << std::endl;
    
    return;
}

template<typename T>
std::vector<T> getVector(cv::Mat mat){
  std::vector<T> array(mat.cols*mat.rows*mat.channels());

  uint32_t index = 0;
  for(int j=0;j<mat.rows;j++){
    for(int i=0;i<mat.cols;i++){
      array[index++] = mat.at<T>(j,i);
    }
  }
  
  return array;
}

int main(int argc, char*argv[])
{
    
    kp::Manager mgr(0);

    cv::Mat in1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    MatType(in1);

    cv::resize(in1, in1, cv::Size(640, 480), 0, 0, cv::INTER_AREA);

    int width = in1.cols;
    int height = in1.rows;
    int channels = in1.channels();

    std::cout << width << "," << height << "," << channels << std::endl;

    cv::Mat in2 = cv::Mat::ones(cv::Size(width, height),CV_8U);
    // simple colour ramp, will go from 0-255 for uint8
    uint32_t ramp = 0;
    for(int j=0;j<height;j++){
      for(int i=0;i<width;i++){
        in2.at<uchar>(j,i) = ramp++;
      }
    }
    cv::Mat in3 = cv::Mat::zeros(cv::Size(width, height),CV_8U);

    std::vector<uint8_t> vec = getVector<uint8_t>(in1);
    std::vector<uint8_t> vec2 = getVector<uint8_t>(in2);
    std::vector<uint8_t> vec3 = getVector<uint8_t>(in3);

    std::shared_ptr<kp::ImageT<uint8_t>> image  = mgr.imageT<uint8_t>(vec,  width, height, channels);
    std::shared_ptr<kp::ImageT<uint8_t>> image2 = mgr.imageT<uint8_t>(vec2, width, height, channels);
    std::shared_ptr<kp::ImageT<uint8_t>> image3 = mgr.imageT<uint8_t>(vec3, width, height, channels);

    const std::vector<std::shared_ptr<kp::Memory>> params = { image,
                                                              image2,
                                                              image3 };

    kp::Workgroup workgroup = { width, height, channels };

    const std::vector<uint32_t> shader = std::vector<uint32_t>(
      shader::MY_SHADER_COMP_SPV.begin(), shader::MY_SHADER_COMP_SPV.end());
    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, shader, workgroup);

    
    mgr.sequence()
      ->record<kp::OpSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algo)
      ->record<kp::OpSyncLocal>(params)
      ->eval();

    std::vector<uchar> r1 = image->vector();
    std::vector<uchar> r2 = image2->vector();
    std::vector<uchar> r3 = image3->vector();

    cv::Mat out1 = cv::Mat(cv::Size(width, height),in1.type(),r1.data());
    cv::Mat out2 = cv::Mat(cv::Size(width, height),in2.type(),r2.data());
    cv::Mat out3 = cv::Mat(cv::Size(width, height),in3.type(),r3.data());

    cv::imshow("input",out1);
    cv::imshow("pattern",out2);
    cv::imshow("output3",out3);

    cv::waitKey(0);

    return 0;
}
