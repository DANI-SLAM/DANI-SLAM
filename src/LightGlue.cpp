#include "LightGlue.h"
#include <iostream>
#include <limits>
namespace ORB_SLAM3 {

// 构造函数：加载模型并初始化
LightGlue::LightGlue() {
    // 加载 TorchScript 模型，使用 CPU
    model = torch::jit::load("/sly_slam/lightglue_scripted_v2.pt");
    model.to(torch::kCPU); 
    model.eval(); // 设置为推理模式
}

LightGlue::~LightGlue(){

}

torch::Tensor LightGlue::keypointsToTensor(const std::vector<cv::KeyPoint>& keypoints) {
    torch::Tensor tensor = torch::zeros({static_cast<int64_t>(keypoints.size()), 2}, torch::kFloat32);
    for (size_t i = 0; i < keypoints.size(); ++i) {
        tensor[i][0] = keypoints[i].pt.x; // x 坐标
        tensor[i][1] = keypoints[i].pt.y; // y 坐标
    }
    return tensor;
}


torch::Tensor LightGlue::normalizeKeypoints(
    const torch::Tensor& kpts, const torch::Tensor& size) 
{
    torch::Tensor normalized_kpts;
    torch::Tensor calculated_size;

    if (size.numel() == 0) {
        calculated_size = 1 + (std::get<0>(kpts.max(0)) - std::get<0>(kpts.min(0)));
    } else {
        calculated_size = size;
    }

    calculated_size = calculated_size.to(kpts.device(), kpts.dtype());


    // 计算平移和缩放因子
    torch::Tensor shift = calculated_size / 2;
    torch::Tensor scale = calculated_size.max().unsqueeze(-1) / 2;

    // 归一化关键点
    normalized_kpts = (kpts - shift.unsqueeze(0)) / scale;

    return normalized_kpts;
}



// 匹配描述符的主函数
std::vector<cv::DMatch> LightGlue::matchDescriptors(
    const std::vector<cv::KeyPoint>& keypoints1, 
    const torch::Tensor& descriptors1, 
    const std::vector<cv::KeyPoint>& keypoints2, 
    const torch::Tensor& descriptors2,  
    const torch::Tensor& size1, 
    const torch::Tensor& size2, 
    float threshold) 
{
    // 将 KeyPoint 转换为 Tensor
    torch::Tensor kpts1_tensor = keypointsToTensor(keypoints1);
    torch::Tensor kpts2_tensor = keypointsToTensor(keypoints2);

    // 对特征点进行归一化
    kpts1_tensor = normalizeKeypoints(kpts1_tensor, size1);
    kpts2_tensor = normalizeKeypoints(kpts2_tensor, size2);

    // 添加 batch 维度
    torch::Tensor desc1_batched = descriptors1.unsqueeze(0);
    torch::Tensor desc2_batched = descriptors2.unsqueeze(0);

    // 构造模型输入
    std::vector<c10::IValue> inputs;
    inputs.push_back(kpts1_tensor.unsqueeze(0));
    inputs.push_back(kpts2_tensor.unsqueeze(0));
    inputs.push_back(desc1_batched);
    inputs.push_back(desc2_batched);

    auto output = model.forward(inputs).toTuple();

    // 获取匹配对和匹配分数
    torch::Tensor matches = output->elements()[0].toTensor();
    torch::Tensor scores = output->elements()[1].toTensor();

    std::vector<cv::DMatch> filtered_matches = filterMatches(matches, scores, threshold);

   
    return filtered_matches;
}


// 静态函数：过滤低置信度的匹配
std::vector<cv::DMatch> LightGlue::filterMatches(
    const torch::Tensor& matches, 
    const torch::Tensor& scores, 
    float threshold) 
{
    std::vector<cv::DMatch> filtered_matches;

    if (matches.numel() == 0 || scores.numel() == 0) {
        return filtered_matches; // 如果没有匹配结果，直接返回空的匹配结果
    }
    auto matches_data = matches.data_ptr<int64_t>();
    auto scores_data = scores.data_ptr<float>();

    // 遍历所有匹配对，筛选出符合置信度要求的匹配
    for (int i = 0; i < matches.size(0); ++i) {
        if (scores_data[i] > threshold) {
            cv::DMatch match;
            match.queryIdx = static_cast<int>(matches_data[i * 2]); // 关键点1索引
            match.trainIdx = static_cast<int>(matches_data[i * 2 + 1]); // 关键点2索引
            match.distance = scores_data[i]; // 匹配的置信度
            filtered_matches.push_back(match);
        }
    }
    return filtered_matches;
}

} // namespace ORB_SLAM3
