#ifndef LGMATCHER_H
#define LGMATCHER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include "sophus/sim3.hpp"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"
#include "LightGlue.h"  // 确保引入LightGlue头文件

namespace ORB_SLAM3
{

class LGmatcher
{
public:
    // 构造函数，初始化LightGlue匹配器
    LGmatcher(float nnratio, bool checkOri);

    // 使用SuperPoint和LightGlue进行特征匹配
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th = 3, const bool bFarPoints = false, const float thFarPoints = 50.0f);

    // 将上一帧中的地图点投影到当前帧，并搜索匹配点。用于帧间追踪。
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

    // 将关键帧中的地图点投影到当前帧，并搜索匹配点。用于重定位。
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist);

    // 使用相似变换投影地图点，并进行匹配。用于回环检测或地图融合。
    int SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th, float ratioHamming=1.0);

    // 替换特征匹配函数
    int SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                           std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming = 1.0);

    // 使用DBoW3进行全局词袋模型匹配。保留回环检测功能。
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

    // 匹配初始化过程中的两帧，用于单目初始化。
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize = 10);

    // 基于极线约束进行三角化匹配，用于生成新的地图点。
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<std::pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse = false);

    // 使用Sim3变换搜索匹配点，用于回环检测和地图融合。
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th);

    // 将地图点投影到关键帧中并搜索重复的地图点。用于地图融合。
    int Fuse(KeyFrame* pKF, const std::vector<MapPoint*> &vpMapPoints, const float th = 3.0 , const bool bRight = false);

    // 使用Sim3变换将地图点投影到关键帧中并搜索重复的地图点。
    int Fuse(KeyFrame* pKF, Sophus::Sim3f &Scw, const std::vector<MapPoint*> &vpPoints, float th, std::vector<MapPoint*> &vpReplacePoint);

    
    void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    float RadiusByViewingCos(const float &viewCos);

public:
    static const float TH_LOW;
    static const float TH_HIGH;
    static const int HISTO_LENGTH;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    float mfNNratio;  // 最邻近匹配比率
    bool mbCheckOrientation;  // 是否检查方向一致性
    LightGlue lg;  // LightGlue实例
};

typedef LGmatcher ORBmatcher;

} // namespace ORB_SLAM3

#endif // LGMATCHER_H
