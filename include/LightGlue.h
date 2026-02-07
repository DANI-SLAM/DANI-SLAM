#ifndef LIGHTGLUE_H
#define LIGHTGLUE_H

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <opencv2/core.hpp> // 包含 cv::DMatch
#include <opencv2/features2d.hpp>  //cv::KeyPoint
namespace ORB_SLAM3 {

class LightGlue {
public:
    // 构造函数：加载 TorchScript 模型
    LightGlue();
    ~LightGlue();
    // 接口函数：匹配描述符
    std::vector<cv::DMatch> matchDescriptors(
        const std::vector<cv::KeyPoint>& keypoints1, 
        const torch::Tensor& descriptors1, 
        const std::vector<cv::KeyPoint>& keypoints2, 
        const torch::Tensor& descriptors2,  
        const torch::Tensor& size1, 
        const torch::Tensor& size2,float threshold = 0.05f);

private:
    // TorchScript 模型
    torch::jit::script::Module model;
    static torch::Tensor keypointsToTensor(const std::vector<cv::KeyPoint>& keypoints);
    torch::Tensor normalizeKeypoints(const torch::Tensor&, const torch::Tensor& size);

    // 过滤匹配结果的静态函数
    static std::vector<cv::DMatch> filterMatches(
        const torch::Tensor& matches, 
        const torch::Tensor& scores, 
        float threshold);
};

} // namespace ORB_SLAM3

#endif // LIGHTGLUE_H
