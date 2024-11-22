
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>

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

cv::Mat sharpenKernel(){
  cv::Mat K = cv::Mat(cv::Size(3,3),CV_32F);
  K.at<float>(0,0) = 0;
  K.at<float>(0,1) = -1;
  K.at<float>(0,2) = 0;

  K.at<float>(1,0) = -1;
  K.at<float>(1,1) = 5;
  K.at<float>(1,2) = -1;

  K.at<float>(2,0) = 0;
  K.at<float>(2,1) = -1;
  K.at<float>(2,2) = 0;
  
  return K;
}

int main(int argc, char*argv[])
{
    int gpu_index = 0;
    int kernel_size = 0;
    float sigma = 10.0f;

    if(argc>4){
      gpu_index = std::stoi(argv[4]);
    }
    if(argc>3){
      sigma = std::stof(argv[3]);
    }
    if(argc>2){
      kernel_size = std::stoi(argv[2]);
    }

    kp::Manager mgr(gpu_index);

    cv::Mat in1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    MatType(in1);

    cv::resize(in1, in1, cv::Size(640, 480), 0, 0, cv::INTER_AREA);

    uint width = in1.cols;
    uint height = in1.rows;
    uint channels = in1.channels();

    std::cout << width << "," << height << "," << channels << std::endl;

    int kernel_length = kernel_size;

    // create 1D gaussian, will apply in 2 directions
    //cv::Mat kernel = cv::getGaussianKernel( kernel_length, sigma);
    //cv::Mat kernel = cv::Mat::ones( cv::Size(kernel_size,kernel_size), CV_32F);
    //kernel /= 7.0;
    cv::Mat kernel = sharpenKernel();

    std::cout << "kernel:" << kernel << std::endl;
    std::cout << kernel.cols << " " << kernel.rows << std::endl;

    // allocate memory for the output image, taking care to
	  // "pad" the borders of the input image so the spatial
	  // size (i.e., width and height) are not reduced
	  int pad = (kernel_size - 1) / 2;
	  cv::Mat padded_image;
    cv::copyMakeBorder(in1, padded_image, pad, pad, pad, pad, cv::BORDER_REPLICATE);

    cv::Mat output = cv::Mat::zeros(cv::Size(width, height),CV_32F);

    std::vector<uchar> vec = getVector<uchar>(padded_image);
    std::vector<float> vec2 = getVector<float>(kernel);
    std::vector<float> vec3 = getVector<float>(output);

    std::shared_ptr<kp::ImageT<uchar>> image  = mgr.imageT<uchar>(vec,  padded_image.cols, padded_image.rows, padded_image.channels());
    std::shared_ptr<kp::ImageT<float>> image2 = mgr.imageT<float>(vec2, kernel.cols, kernel.rows, kernel.channels());
    std::shared_ptr<kp::ImageT<float>> image3 = mgr.imageT<float>(vec3, in1.cols, in1.rows, in1.channels());

    const std::vector<std::shared_ptr<kp::Memory>> params = { image,
                                                              image2,
                                                              image3 };

    uint pw = padded_image.cols;
    uint ph = padded_image.rows;
    uint pc = padded_image.channels();

    std::cout << pw << " " << ph << " " << pc << std::endl;

    kp::Workgroup workgroup = { pw, ph, pc};
    std::vector<float> specConsts({ 2 });
    std::vector<float> push_const_x({ (float)kernel_size,(float)pad, -1.0 });
    std::vector<float> push_const_y({ (float)kernel_size,(float)pad, 1.0 });

    const std::vector<uint32_t> shader = std::vector<uint32_t>(
      shader::MY_SHADER_COMP_SPV.begin(), shader::MY_SHADER_COMP_SPV.end());
    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, shader, workgroup, specConsts,push_const_x);

    mgr.sequence()
      ->record<kp::OpSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algo)
      ->record<kp::OpSyncLocal>(params)
      ->eval();
    
    std::vector<float> r3 = image3->vector();
    output = cv::Mat(cv::Size(in1.cols, in1.rows), CV_32FC1, r3.data());

    cv::normalize(output, output, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    output *= 255;
    output.convertTo(output, CV_8UC1);

    cv::Mat toutput;
    cv::filter2D( in1, toutput, -1, kernel);

    std::cout << "#non zero values:" << countNonZero(output) << std::endl;

    MatType(output);
    MatType(toutput);

    cv::imshow("input",in1);
    cv::imshow("kernel",kernel);
    cv::imshow("output",output);
    cv::imshow("toutput",toutput);

    cv::waitKey(0);

    cv::imwrite("test.tiff",output);

    return 0;
}
