#ifndef SPEXTRACTOR_H
#define SPEXTRACTOR_H

#include <vector>
#include <opencv2/core/core.hpp>
#include "SuperPoint.h"  // SuperPoint相关头文件

namespace ORB_SLAM3
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class SPextractor
{
public:
    // 构造函数：初始化SuperPoint模型参数
    SPextractor(int nfeatures, float scaleFactor, int nlevels, 
                float iniThFAST, float minThFAST);

    ~SPextractor() {}

    // 该函数将使用SuperPoint模型进行特征点和描述符的提取
    int operator()(cv::InputArray _image, cv::InputArray _mask,
                   std::vector<cv::KeyPoint>& _keypoints, 
                   cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    // 获取图像层数
    int inline GetLevels() {
        return nlevels;
    }

    // 获取缩放因子
    float inline GetScaleFactor() {
        return scaleFactor;
    }

    // 获取不同层的缩放因子
    std::vector<float> inline GetScaleFactors() {
        return mvScaleFactor;
    }

    // 获取每层的反向缩放因子
    std::vector<float> inline GetInverseScaleFactors() {
        return mvInvScaleFactor;
    }

    // 获取每层的sigma平方
    std::vector<float> inline GetScaleSigmaSquares() {
        return mvLevelSigma2;
    }

    // 获取每层的反向sigma平方
    std::vector<float> inline GetInverseScaleSigmaSquares() {
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;// 图像金字塔
    std::vector<cv::Rect2i> mvDynamicArea;//动态区域

protected:

    // 图像金字塔的生成，若需要多层次特征
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints,cv::Mat &_desc);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);
    

    int nfeatures;  // 期望的特征点数量
    double scaleFactor;  // 缩放因子
    int nlevels;  // 金字塔层数
    float iniThFAST;  
    float minThFAST;  

    std::vector<int> mnFeaturesPerLevel;  // 每层的特征点数目

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;  // 每层的缩放因子
    std::vector<float> mvInvScaleFactor;  // 每层的反向缩放因子
    std::vector<float> mvLevelSigma2;  // 每层的sigma平方
    std::vector<float> mvInvLevelSigma2;  // 每层的反向sigma平方

    std::shared_ptr<SuperPoint> model;
};

#ifndef ORB_SLAM3_ORBEXTRACTOR_DEFINED
#define ORB_SLAM3_ORBEXTRACTOR_DEFINED
typedef SPextractor ORBextractor;
#endif


} // namespace ORB_SLAM3

#endif // SPEXTRACTOR_H
