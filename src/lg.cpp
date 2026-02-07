/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "LGmatcher.h"

#include "LightGlue.h"

#include <torch/torch.h>

#include<limits.h>

#include<opencv2/core/core.hpp>

#include "Thirdparty/DBow3/src/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM3
{
    //通过投影进行特征点匹配，用于在连续帧之间建立地图点和图像特征点之间的关系，从而实现相机的位姿估计和地图的构建

    const float LGmatcher::TH_HIGH = 0.7f;
    const float LGmatcher::TH_LOW = 0.3f; 
    const int LGmatcher::HISTO_LENGTH = 30; // 匹配时用于直方图计算的长度

    // 构造函数,参数默认值为0.6,true      nnratio最近邻比率，用于判定最佳和次佳匹配之间的距离比率
    LGmatcher::LGmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri), lg()//checkOri是否检查特征点方向
    {
    }
    int LGmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
    {
        int nmatches=0, left = 0, right = 0;

        const bool bFactor = th!=1.0;

        for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
        {
            MapPoint* pMP = vpMapPoints[iMP];
            if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
                continue;

            if(bFarPoints && pMP->mTrackDepth>thFarPoints)
                continue;

            if(pMP->isBad())
                continue;

            if(pMP->mbTrackInView)
            {
                const int &nPredictedLevel = pMP->mnTrackScaleLevel;

                // The size of the window will depend on the viewing direction
                float r = RadiusByViewingCos(pMP->mTrackViewCos);

                if(bFactor)
                    r*=th;

                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

                if(!vIndices.empty()){
                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    int bestDist=256;
                    int bestLevel= -1;
                    int bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        if(F.mvpMapPoints[idx])
                            if(F.mvpMapPoints[idx]->Observations()>0)
                                continue;

                        if(F.Nleft == -1 && F.mvuRight[idx]>0)
                        {
                            const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                            if(er>r*F.mvScaleFactors[nPredictedLevel])
                                continue;
                        }

                        const cv::Mat &d = F.mDescriptors.row(idx);

                        const int dist = DescriptorDistance(MPdescriptor,d);

                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                        : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                          : F.mvKeysRight[idx - F.Nleft].octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                         : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                           : F.mvKeysRight[idx - F.Nleft].octave;
                            bestDist2=dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if(bestDist<=TH_HIGH)
                    {
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;

                        if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2){
                            F.mvpMapPoints[bestIdx]=pMP;

                            if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                                F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                                nmatches++;
                                right++;
                            }

                            nmatches++;
                            left++;
                        }
                    }
                }
            }

            if(F.Nleft != -1 && pMP->mbTrackInViewR){
                const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
                if(nPredictedLevel != -1){
                    float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                    const vector<size_t> vIndices =
                            F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                    if(vIndices.empty())
                        continue;

                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    int bestDist=256;
                    int bestLevel= -1;
                    int bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        if(F.mvpMapPoints[idx + F.Nleft])
                            if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                                continue;


                        const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                        const int dist = DescriptorDistance(MPdescriptor,d);

                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = F.mvKeysRight[idx].octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = F.mvKeysRight[idx].octave;
                            bestDist2=dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if(bestDist<=TH_HIGH)
                    {
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;

                        if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                            nmatches++;
                            left++;
                        }


                        F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                        nmatches++;
                        right++;
                    }
                }
            }
        }
        std::cout << "111111" << "匹配对数为：" << nmatches << std::endl;
        return nmatches;
    }
    // 用于Tracking::SearchLocalPoints中匹配更多地图点         在当前帧中通过投影方式寻找与地图点匹配的特征点
    // int LGmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
    // {
    //     int nmatches = 0;  // 记录匹配数量

    //     // 用于批量存储地图点的描述符、关键点和匹配输入
    //     std::vector<cv::KeyPoint> keypoints1;
    //     cv::Mat descriptors1;

    //     // 1. 遍历所有地图点并收集需要匹配的数据
    //     for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
    //         MapPoint* pMP = vpMapPoints[iMP];

    //         // 跳过无效或距离太远的地图点
    //         if (pMP->isBad()) continue;
    //         if (bFarPoints && pMP->mTrackDepth > thFarPoints) continue;

    //         // 获取地图点的投影坐标 (u, v)
    //         float u = pMP->mTrackProjX;
    //         float v = pMP->mTrackProjY;

    //         // 在投影坐标附近筛选当前帧的特征点
    //         int windowSize = 15;
    //         vector<size_t> vIndices = F.GetFeaturesInArea(u, v, windowSize);
    //         if (vIndices.empty()) continue;  // 无特征点时跳过

    //         // 获取地图点的描述符
    //         cv::Mat desc1 = pMP->GetDescriptor();

    //         // 累积地图点的描述符和投影点作为 LightGlue 的输入
    //         float scale = std::max(1.0f, pMP->mTrackDepth);
    //         cv::KeyPoint kp1(u, v, scale);  // 使用投影位置创建一个虚拟关键点
    //         keypoints1.push_back(kp1);
    //         descriptors1.push_back(desc1);
    //     }

    //     // 若没有有效的地图点，提前返回
    //     if (keypoints1.empty())
    //     {
    //         std::cout <<"无有效的地图点"<< std::endl;
    //         return nmatches;
    //     }
            

    //     // 2. 准备当前帧的描述符和关键点
    //     std::vector<cv::KeyPoint> keypoints2;
    //     cv::Mat descriptors2;

    //     for (size_t idx = 0; idx < F.N; idx++) {
    //         keypoints2.push_back(F.mvKeysUn[idx]);
    //         descriptors2.push_back(F.mDescriptors.row(idx));
    //     }
    //     // std::cout << "1前5个关键帧关键点坐标（如果存在）:" << std::endl;
    //     // for (int i = 0; i < std::min(5, (int)keypoints2.size()); i++) {
    //     //     std::cout << "KeyFrame " << i << ": (" << keypoints2[i].pt.x << ", " << keypoints2[i].pt.y << ")" << std::endl;
    //     // }

    //     // std::cout << "1前5个地图点坐标（如果存在）:" << std::endl;
    //     // for (int i = 0; i < std::min(5, (int)keypoints1.size()); i++) {
    //     //     std::cout << "MapPoint " << i << ": (" << keypoints1[i].pt.x << ", " << keypoints1[i].pt.y << ")" << std::endl;
    //     // }
    //     // 将描述符转换为 torch::Tensor
    //     torch::Tensor desc1 = torch::from_blob(const_cast<float*>(descriptors1.ptr<float>()), {static_cast<long>(descriptors1.rows), static_cast<long>(descriptors1.cols)}, torch::kFloat);
    //     torch::Tensor desc2 = torch::from_blob(const_cast<float*>(descriptors2.ptr<float>()), {static_cast<long>(descriptors2.rows), static_cast<long>(descriptors2.cols)}, torch::kFloat);

    //     std::cout << "Number of keypoints in mappoints: " << keypoints1.size() << std::endl;
    //     std::cout << "Number of descriptors in mappoints: " << desc1.size(0) << std::endl;

    //     std::cout << "Number of keypoints in currentframe: " << keypoints2.size() << std::endl;
    //     std::cout << "Number of descriptors in currentframe: " << desc2.size(0) << std::endl;
    //     // 获取当前帧的图像尺寸（以左图为例）
    //     int imgWidth = F.mnMaxX - F.mnMinX;  // 图像的宽度
    //     int imgHeight = F.mnMaxY - F.mnMinY; // 图像的高度

    //     // LightGlue 需要尺寸作为输入
    //     torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
    //     torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);

    //     std::cout << "111111" << std::endl;
    //     // 使用 LightGlue 进行批量匹配
    //     std::vector<cv::DMatch> matches = lg.matchDescriptors(keypoints1, desc1, keypoints2, desc2, size1, size2);

    //     // 4. 处理匹配结果
    //     for (const auto &match : matches) {
    //         // 根据距离阈值筛选匹配
    //         if (match.distance >= TH_LOW) {
    //             MapPoint* pMP = vpMapPoints[match.queryIdx];  // 找到对应的地图点
    //             F.mvpMapPoints[match.trainIdx] = pMP;  // 将匹配结果关联到当前帧的特征点
    //             nmatches++;
    //         }
    //     }

    //     std::cout << "LightGlue Matches  1: " << nmatches << std::endl;
    //     return nmatches;
    // }

    // 根据观察的视角来计算匹配的时的搜索窗口大小   用于适应视角变化对特征点匹配的影响
    float LGmatcher::RadiusByViewingCos(const float &viewCos)
    {
        // 当视角相差小于3.6°，对应cos(3.6°)=0.998，搜索范围是2.5，否则是4
        if(viewCos>0.998)
            return 2.5;
        else
            return 4.0;
    }

    /**
     * @brief 通过词袋，对关键帧的特征点进行跟踪
     * 
     * @param[in] pKF               关键帧
     * @param[in] F                 当前普通帧
     * @param[in] vpMapPointMatches F中地图点对应的匹配，NULL表示未匹配
     * @return int                  成功匹配的数量
     */
    int LGmatcher::SearchByBoW(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches)
    {
        // 从关键帧中提取地图点
        const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

        // 初始化匹配结果，将其大小设为帧中关键点的数量，并初始化为空指针
        vpMapPointMatches = vector<MapPoint*>(F.N, static_cast<MapPoint*>(NULL));

        // 获取词袋模型特征向量
        const DBoW3::FeatureVector &vFeatVecKF = pKF->mFeatVec;
        const DBoW3::FeatureVector &vFeatVecF = F.mFeatVec;

        int nmatches = 0; // 匹配数量

        // 旋转直方图，用于处理匹配点的方向一致性检查
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f / HISTO_LENGTH;

        // 存储整个帧和关键帧的所有特征点和描述符
        vector<cv::KeyPoint> keypointsKF, keypointsF;
        cv::Mat descriptorsKF, descriptorsF;
        vector<unsigned int> vIndicesKFAll, vIndicesFAll;

        // 遍历所有节点，收集所有特征点
        DBoW3::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW3::FeatureVector::const_iterator Fit = vFeatVecF.begin();
        DBoW3::FeatureVector::const_iterator KFend = vFeatVecKF.end();
        DBoW3::FeatureVector::const_iterator Fend = vFeatVecF.end();

        while (KFit != KFend && Fit != Fend)
        {
            if (KFit->first == Fit->first)
            {
                // 获取在该节点下的所有特征点索引
                const vector<unsigned int> vIndicesKFNode = KFit->second;
                const vector<unsigned int> vIndicesFNode = Fit->second;

                // 添加关键帧和当前帧的特征点
                for (size_t iKF = 0; iKF < vIndicesKFNode.size(); iKF++)
                {
                    const unsigned int realIdxKF = vIndicesKFNode[iKF];
                    MapPoint* pMP = vpMapPointsKF[realIdxKF];

                    if (!pMP || pMP->isBad())
                        continue;

                    keypointsKF.push_back(pKF->mvKeysUn[realIdxKF]);
                    descriptorsKF.push_back(pKF->mDescriptors.row(realIdxKF));
                    vIndicesKFAll.push_back(realIdxKF);
                }

                for (size_t iF = 0; iF < vIndicesFNode.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesFNode[iF];

                    if (vpMapPointMatches[realIdxF])
                        continue;  // 已匹配则跳过

                    keypointsF.push_back(F.mvKeys[realIdxF]);
                    descriptorsF.push_back(F.mDescriptors.row(realIdxF));
                    vIndicesFAll.push_back(realIdxF);
                }
            }

            KFit++;
            Fit++;
        }

        // 将描述符转换为张量
        torch::Tensor descKF = torch::from_blob(const_cast<float*>(descriptorsKF.ptr<float>()), {static_cast<long>(descriptorsKF.rows), static_cast<long>(descriptorsKF.cols)}, torch::kFloat);
        torch::Tensor descF = torch::from_blob(const_cast<float*>(descriptorsF.ptr<float>()), {static_cast<long>(descriptorsF.rows), static_cast<long>(descriptorsF.cols)}, torch::kFloat);

        // std::cout << "Number of keypoints in keyframe: " << keypointsKF.size() << std::endl;
        // std::cout << "Number of descriptors in keyframe: " << descKF.size(0) << std::endl;

        // std::cout << "Number of keypoints in frame: " << keypointsF.size() << std::endl;
        // std::cout << "Number of descriptors in frame: " << descF.size(0) << std::endl;

        // std::cout << "2前5个关键帧关键点坐标（如果存在）:" << std::endl;
        // for (int i = 0; i < std::min(5, (int)keypointsKF.size()); i++) {
        //     std::cout << "KeyFrame " << i << ": (" << keypointsKF[i].pt.x << ", " << keypointsKF[i].pt.y << ")" << std::endl;
        // }

        // std::cout << "2前5个地图点坐标（如果存在）:" << std::endl;
        // for (int i = 0; i < std::min(5, (int)keypointsF.size()); i++) {
        //     std::cout << "MapPoint " << i << ": (" << keypointsF[i].pt.x << ", " << keypointsF[i].pt.y << ")" << std::endl;
        // }
        int imgWidth = F.mnMaxX - F.mnMinX;  // 图像的宽度
        int imgHeight = F.mnMaxY - F.mnMinY; // 图像的高度

        int imgWidth1 = pKF->mnMaxX - pKF->mnMinX;  // 图像的宽度
        int imgHeight1 = pKF->mnMaxY - pKF->mnMinY; // 图像的高度
        // LightGlue 需要尺寸作为输入
        torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth1), static_cast<float>(imgHeight1)}, torch::kFloat32);
        torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);

        std::cout << "222222" << std::endl;
        // 使用 LightGlue 进行精确匹配（一次性匹配所有点）
        vector<cv::DMatch> matches = lg.matchDescriptors(keypointsKF, descKF, keypointsF, descF, size1, size2);
        std::cout << "Number of matches: " << matches.size() << std::endl;

        // 处理 LightGlue 匹配结果
        for (const auto& match : matches)
        {
            const int idxKF = match.queryIdx;
            const int idxF = match.trainIdx;

            const float distance = match.distance;
            if (distance >= 0.1)
            {
                vpMapPointMatches[vIndicesFAll[idxF]] = vpMapPointsKF[vIndicesKFAll[idxKF]];
                nmatches++;

                // 如果需要检查方向一致性，填充直方图
                if (mbCheckOrientation)
                {
                    const cv::KeyPoint &kpKF = pKF->mvKeysUn[vIndicesKFAll[idxKF]];
                    const cv::KeyPoint &kpF = F.mvKeys[vIndicesFAll[idxF]];

                    float rot = kpKF.angle - kpF.angle;
                    if (rot < 0.0)
                        rot += 360.0f;
                    int bin = round(rot * factor);
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    assert(bin >= 0 && bin < HISTO_LENGTH);
                    rotHist[bin].push_back(vIndicesFAll[idxF]);
                }
            }
        }

        // 如果需要检查方向一致性
        if (mbCheckOrientation)
        {
            int ind1 = -1, ind2 = -1, ind3 = -1;
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            // 删除不符合旋转一致性的匹配点
            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0; j < rotHist[i].size(); j++)
                {
                    vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }

        std::cout << "LightGlue Matches 2: " << nmatches << std::endl;
        return nmatches;  // 返回匹配点数量
    }
    

    //在关键帧中通过投影找到与地图点匹配的特征点
    int LGmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3f &Scw, const vector<MapPoint*> &vpPoints,
                                       vector<MapPoint*> &vpMatched, int th, float ratioHamming)
    {
        // 获取相机内参
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // 通过Sim3转换获取位姿变换矩阵
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();  // 获取相机的世界坐标位置

        // 已经在当前KeyFrame中找到的MapPoints的集合，避免重复匹配
        std::set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(nullptr));  // 删除空指针

        int nmatches = 0;  // 匹配计数器

        // 存储筛选后的有效地图点和相应的描述符
        std::vector<cv::KeyPoint> mapKeypoints;
        cv::Mat mapDescriptors;

        // 遍历所有候选MapPoints，筛选有效的地图点
        for (size_t iMP = 0; iMP < vpPoints.size(); iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // 跳过坏的MapPoints和已经匹配到的MapPoints
            if (pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // 获取3D世界坐标
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // 将世界坐标转换到相机坐标系
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // 深度必须为正值
            if (p3Dc(2) < 0.0)
                continue;

            // 投影到图像平面
            Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

            // 确保点在图像范围内
            if (!pKF->IsInImage(uv(0), uv(1)))
                continue;

            // 检查点的深度是否在可接受的尺度范围内
            float maxDistance = pMP->GetMaxDistanceInvariance();
            float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw - Ow;  // 点到相机中心的向量
            float dist = PO.norm();

            if (dist < minDistance || dist > maxDistance)
                continue;

            // 确保观察角度小于60度
            Eigen::Vector3f Pn = pMP->GetNormal();
            if (PO.dot(Pn) < 0.5 * dist)
                continue;

            // 将该地图点的关键点和描述符存储用于后续批量匹配
            mapKeypoints.emplace_back(cv::KeyPoint(uv(0), uv(1), 1.0f));  // 投影位置作为关键点
            mapDescriptors.push_back(pMP->GetDescriptor());  // 存储描述符
        }

        // 如果没有筛选出有效地图点，直接返回
        if (mapKeypoints.empty())
            return nmatches;

        // 获取当前帧的关键点和描述符
        const std::vector<cv::KeyPoint> &keypointsFrame = pKF->mvKeysUn;
        const cv::Mat &descriptorsFrame = pKF->mDescriptors;

        torch::Tensor descmp = torch::from_blob(const_cast<float*>(mapDescriptors.ptr<float>()), {static_cast<long>(mapDescriptors.rows), static_cast<long>(mapDescriptors.cols)}, torch::kFloat);
        torch::Tensor descKF = torch::from_blob(const_cast<float*>(descriptorsFrame.ptr<float>()), {static_cast<long>(descriptorsFrame.rows), static_cast<long>(descriptorsFrame.cols)}, torch::kFloat);

        int imgWidth = pKF->mnMaxX - pKF->mnMinX;  // 图像的宽度
        int imgHeight = pKF->mnMaxY - pKF->mnMinY; // 图像的高度

        // LightGlue 需要尺寸作为输入
        torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
        torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);

        std::cout << "333333" << std::endl;
        // 使用LightGlue进行全局匹配
        std::vector<cv::DMatch> matches = lg.matchDescriptors(mapKeypoints, descmp, keypointsFrame, descKF, size1, size2);
        

        // 处理匹配结果，更新匹配到的MapPoints
        for (const auto& match : matches)
        {
            size_t idxFrame = match.trainIdx;  // 当前帧中特征点的索引
            if (vpMatched[idxFrame] == nullptr)  // 确保未重复匹配
            {
                vpMatched[idxFrame] = vpPoints[match.queryIdx];  // 将地图点与特征点匹配
                nmatches++;  // 递增匹配计数器
            }
        }

        std::cout << "LightGlue Matches 3: " << nmatches << std::endl;

        return nmatches;  // 返回匹配的数量
    }

    //通过投影地图点并在图像中搜索匹配点，然后进行特征匹配，并将匹配结果与地图点关联，
    //重定位 局部地图优化 回环检测或相机重定位后
    int LGmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                                  std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
    {
        // 初始化匹配计数
        int nmatches = 0;

        // 获取相机内参
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // 获取相机的位姿转换矩阵（世界到相机的Sim3变换）
        const Eigen::Matrix4f Tcw = Scw.matrix();  // 4x4转换矩阵
        const Eigen::Vector3f Ow = Scw.inverse().translation();  // 获取相机光心的世界坐标

        // LightGlue 需要的关键点和描述符容器，用于地图点和当前帧所有特征点进行匹配
        std::vector<cv::KeyPoint> keypoints1;   // 地图点的关键点投影位置
        cv::Mat descriptors1;                   // 地图点的描述符
        std::vector<cv::KeyPoint> keypoints2 = pKF->mvKeysUn; // 当前帧所有特征点
        cv::Mat descriptors2 = pKF->mDescriptors;             // 当前帧所有特征点的描述符

        // 遍历所有候选地图点，进行投影和匹配
        for (size_t iMP = 0; iMP < vpPoints.size(); ++iMP) {
            MapPoint* pMP = vpPoints[iMP];        // 当前地图点
            KeyFrame* pKFi = vpPointsKFs[iMP];    // 当前地图点对应的关键帧

            // 跳过无效的地图点或已经匹配的点
            if (pMP->isBad() || vpMatched[iMP]) continue;

            // 获取地图点的3D坐标
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // 转换到相机坐标系
            Eigen::Vector3f p3Dc = Tcw.topLeftCorner(3, 3) * p3Dw + Tcw.topRightCorner(3, 1); 

            // 深度必须为正（即投影点必须在相机前面）
            if (p3Dc(2) < 0.0) continue;

            // 将3D点投影到图像平面上
            const float invz = 1.0f / p3Dc(2); // 深度倒数
            const float x = p3Dc(0) * invz;
            const float y = p3Dc(1) * invz;

            const float u = fx * x + cx;  // 像素坐标u
            const float v = fy * y + cy;  // 像素坐标v

            // 检查投影点是否在图像范围内
            if (!pKF->IsInImage(u, v)) continue;

            // 计算地图点的尺度不变区域（根据地图点距离）
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw - Ow;
            const float dist = PO.norm();

            // 地图点的距离必须在合理范围内
            if (dist < minDistance || dist > maxDistance) continue;

            // 检查地图点的视角角度（确保视角小于60度）
            Eigen::Vector3f Pn = pMP->GetNormal();
            if (PO.dot(Pn) < 0.5 * dist) continue;

            // 预测地图点在图像中的尺度等级
            const int nPredictedLevel = pMP->PredictScale(dist, pKF);

            // 将地图点作为投影点，添加到 keypoints1 和 descriptors1 中
            cv::KeyPoint kp(u, v, pKF->mvScaleFactors[nPredictedLevel]); // 创建关键点
            keypoints1.push_back(kp);
            descriptors1.push_back(pMP->GetDescriptor());  // 获取地图点的描述符
        }

        // 如果没有找到可用的地图点，直接返回
        if (keypoints1.empty()) return nmatches;

        torch::Tensor desc1 = torch::from_blob(const_cast<float*>(descriptors1.ptr<float>()), {static_cast<long>(descriptors1.rows), static_cast<long>(descriptors1.cols)}, torch::kFloat);
        torch::Tensor desc2 = torch::from_blob(const_cast<float*>(descriptors2.ptr<float>()), {static_cast<long>(descriptors2.rows), static_cast<long>(descriptors2.cols)}, torch::kFloat);

        int imgWidth = pKF->mnMaxX - pKF->mnMinX;  // 图像的宽度
        int imgHeight = pKF->mnMaxY - pKF->mnMinY; // 图像的高度

        // LightGlue 需要尺寸作为输入
        torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
        torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);

        std::cout << "444444" << std::endl;
        // 使用 LightGlue 进行地图点与当前帧特征点的匹配
        std::vector<cv::DMatch> matches = lg.matchDescriptors(keypoints1, desc1, keypoints2, desc2, size1, size2);
        

        // 处理匹配结果
        for (const auto& match : matches) {
            // 获取地图点的索引和特征点的索引
            const int mapPointIdx = match.queryIdx;   // 地图点索引
            const int keyPointIdx = match.trainIdx;   // 特征点索引

            // 记录成功匹配的地图点和其对应的特征点
            vpMatched[mapPointIdx] = vpPoints[mapPointIdx];
            vpMatchedKF[mapPointIdx] = pKF;

            // 更新匹配计数
            nmatches++;
        }
        std::cout << "LightGlue Matches  4: " << nmatches << std::endl;
        // 返回匹配的地图点数量
        return nmatches;
    }
    

    int LGmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
    {
        int nmatches = 0; // 记录成功匹配的数量
        vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);// 初始化匹配结果，-1表示未匹配

        // 角度直方图初始化，用于后续处理角度不一致的匹配
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500); // 每个直方图bin初始化容量为500个匹配点
        const float factor = 1.0f / HISTO_LENGTH; // 用于将角度差转换为直方图bin的因子

        // 用于存储匹配结果的距离信息
        vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX); // 初始化为最大整数，表示匹配距离
        vector<int> vnMatches21(F2.mvKeysUn.size(), -1); // 初始化匹配信息，记录第二帧与第一帧的匹配关系

        torch::Tensor desc1 = torch::from_blob(const_cast<float*>(F1.mDescriptors.ptr<float>()), {static_cast<long>(F1.mDescriptors.rows), static_cast<long>(F1.mDescriptors.cols)}, torch::kFloat);
        torch::Tensor desc2 = torch::from_blob(const_cast<float*>(F2.mDescriptors.ptr<float>()), {static_cast<long>(F2.mDescriptors.rows), static_cast<long>(F2.mDescriptors.cols)}, torch::kFloat);

        int imgWidth = F1.mnMaxX - F1.mnMinX;  // 图像的宽度
        int imgHeight = F1.mnMaxY - F1.mnMinY;  // 图像的高度

        // LightGlue 需要尺寸作为输入
        torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
        torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);

        std::cout << "555555" << std::endl;
        // 使用 LightGlue 进行描述符匹配
        std::vector<cv::DMatch> matches = lg.matchDescriptors(F1.mvKeysUn, desc1, F2.mvKeysUn, desc2, size1, size2);
        

        // 遍历匹配结果
        for (const auto &match : matches)
        {
            int i2 = match.trainIdx; // 第二帧中的关键点索引
            int i1 = match.queryIdx; // 第一帧中的关键点索引

            int dist = match.distance; // 使用 LightGlue 计算的匹配距离

            // 如果当前匹配点的距离比之前的匹配距离大，则跳过该匹配点
            if (vMatchedDistance[i2] <= dist)
                continue;

            if (dist >= 0.1) // 自定义阈值，如果匹配距离小于阈值 TH_LOW，认为是有效匹配
            {
                if (vnMatches21[i2] >= 0)  // 如果第二帧中的关键点已经与其他第一帧的关键点匹配，则取消之前的匹配
                {
                    vnMatches12[vnMatches21[i2]] = -1;
                    nmatches--;  // 匹配数减1
                }
                // 记录新的匹配关系
                vnMatches12[i1] = i2;
                vnMatches21[i2] = i1;
                vMatchedDistance[i2] = dist;// 更新匹配距离
                nmatches++;// 匹配数加1

                // 如果启用了角度一致性检查，则计算两帧间特征点的角度差
                if (mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[i2].angle;  // 计算角度差
                    if (rot < 0.0) // 角度差可能为负，调整为正值
                        rot += 360.0f;
                    int bin = round(rot * factor); // 将角度差转换为直方图中的bin
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    assert(bin >= 0 && bin < HISTO_LENGTH);
                    rotHist[bin].push_back(i1);// 将匹配点加入对应角度差的直方图bin中
                }
            }
        }

        // 如果启用了角度一致性检查，则进行直方图修正，保留角度最一致的匹配
        if (mbCheckOrientation)
        {
            int ind1 = -1; // 最大的三个角度bin索引
            int ind2 = -1;
            int ind3 = -1;
            // 计算直方图中具有最多匹配点的三个bin索引
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            // 移除角度差不在前三的匹配点
            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    int idx1 = rotHist[i][j];
                    if (vnMatches12[idx1] >= 0)
                    {
                        vnMatches12[idx1] = -1; // 移除匹配关系
                        nmatches--;// 匹配数减1
                    }
                }
            }
        }

        // 更新上一帧的匹配信息，用于下一次的匹配初始化
        for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
            if (vnMatches12[i1] >= 0)
                vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;
        
        std::cout << "LightGlue Matches  5: " << nmatches << std::endl;

        return nmatches; // 返回最终的有效匹配数
    }

    /*
    * @brief 通过词袋，对关键帧的特征点进行跟踪，该函数用于闭环检测时两个关键帧间的特征点匹配
    * @details 通过bow对pKF和F中的特征点进行快速匹配（不属于同一node的特征点直接跳过匹配） 
    * 对属于同一node的特征点通过描述子距离进行匹配 
    * 通过距离阈值、比例阈值和角度投票进行剔除误匹配
    * @param  pKF1               KeyFrame1
    * @param  pKF2               KeyFrame2
    * @param  vpMatches12        pKF2中与pKF1匹配的MapPoint，vpMatches12[i]表示匹配的地图点，null表示没有匹配，i表示匹配的pKF1 特征点索引
    * @return                    成功匹配的数量
    */
    int LGmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
    {
       // Step 1: 使用DBoW3快速全局匹配来确定可能匹配的候选帧
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    
        // 获取SuperPoint的描述符
        const cv::Mat &Descriptors1 = pKF1->mDescriptors;
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        vpMatches12 = vector<MapPoint*>(pKF1->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
        vector<bool> vbMatched2(pKF2->GetMapPointMatches().size(), false);

        int nmatches = 0;
    
        // Step 2: 使用DBoW3进行词袋模型匹配，找到潜在的候选匹配对
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        // 遍历两个帧的词袋特征
        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while (f1it != f1end && f2it != f2end)
        {
            if (f1it->first == f2it->first)  // 同一个词袋下的特征点匹配
            {
                // 构建待匹配的关键点和描述符列表
                std::vector<cv::KeyPoint> keypoints1;
                std::vector<cv::KeyPoint> keypoints2;
                cv::Mat descriptors1;
                cv::Mat descriptors2;

                for (size_t i1 = 0; i1 < f1it->second.size(); i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    MapPoint* pMP1 = pKF1->GetMapPointMatches()[idx1];

                    if (!pMP1 || pMP1->isBad())
                        continue;

                    keypoints1.push_back(vKeysUn1[idx1]);
                    descriptors1.push_back(Descriptors1.row(idx1));
                }

                for (size_t i2 = 0; i2 < f2it->second.size(); i2++)
                {
                    const size_t idx2 = f2it->second[i2];
                    MapPoint* pMP2 = pKF2->GetMapPointMatches()[idx2];

                    if (!pMP2 || pMP2->isBad() || vbMatched2[idx2])
                        continue;

                    keypoints2.push_back(vKeysUn2[idx2]);
                    descriptors2.push_back(Descriptors2.row(idx2));
                }

                torch::Tensor desc1 = torch::from_blob(const_cast<float*>(descriptors1.ptr<float>()), {static_cast<long>(descriptors1.rows), static_cast<long>(descriptors1.cols)}, torch::kFloat);
                torch::Tensor desc2 = torch::from_blob(const_cast<float*>(descriptors2.ptr<float>()), {static_cast<long>(descriptors2.rows), static_cast<long>(descriptors2.cols)}, torch::kFloat);

                int imgWidth = pKF1->mnMaxX - pKF1->mnMinX;  // 图像的宽度
                int imgHeight = pKF1->mnMaxY - pKF1->mnMinY; // 图像的高度

                // LightGlue 需要尺寸作为输入
                torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
                torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
                std::cout << "666666" << std::endl;
                // Step 3: 使用LightGlue进行精确匹配
                std::vector<cv::DMatch> matches = lg.matchDescriptors(keypoints1, desc1, keypoints2, desc2, size1, size2);
                

                // Step 4: 处理匹配结果
                for (const auto &match : matches)
                {
                    const size_t idx1 = match.queryIdx;
                    const size_t idx2 = match.trainIdx;

                    if (match.distance >= 0.1)  // 根据距离阈值判断是否匹配
                    {
                        vpMatches12[f1it->second[idx1]] = pKF2->GetMapPointMatches()[f2it->second[idx2]];
                        vbMatched2[f2it->second[idx2]] = true;
                        nmatches++;
                    }
                }

                f1it++;
                f2it++;
            }
            else if (f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        std::cout << "LightGlue Matches  6: " << nmatches << std::endl;
        return nmatches;
    }
    int LGmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
    {
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const cv::Mat &Descriptors1 = pKF1->mDescriptors;

        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(),false);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        int nmatches = 0;

        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    int bestDist1=256;
                    int bestIdx2 =-1 ;
                    int bestDist2=256;

                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                            continue;
                        }

                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        int dist = DescriptorDistance(d1,d2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    if(bestDist1<TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;

                            if(mbCheckOrientation)
                            {
                                float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
        std::cout << "666666" << std::endl;
        std::cout<< "匹配对数为：" << nmatches << std::endl;
        return nmatches;
    }

    //用于两个关键帧（KeyFrame）之间寻找特征点匹配对的函数，目的是为后续的三角化计算提供匹配的特征点对
    // int LGmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
    //                                        vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
    // {
    //     // 获取两个关键帧的词袋模型特征向量
    //     const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    //     const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //     // Step 1 计算KF1的相机中心在KF2图像平面的二维像素坐标
    //     Sophus::SE3f T1w = pKF1->GetPose();
    //     Sophus::SE3f T2w = pKF2->GetPose();
    //     Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
    //     Eigen::Vector3f Cw = pKF1->GetCameraCenter();
    //     Eigen::Vector3f C2 = T2w * Cw;

    //     Eigen::Vector2f ep = pKF2->mpCamera->project(C2);
    //     Sophus::SE3f T12;
    //     Sophus::SE3f Tll, Tlr, Trl, Trr;
    //     Eigen::Matrix3f R12; // for fastest computation
    //     Eigen::Vector3f t12; // for fastest computation

    //     GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

    //     if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
    //         T12 = T1w * Tw2;
    //         R12 = T12.rotationMatrix();
    //         t12 = T12.translation();
    //     }
    //     else{
    //         Sophus::SE3f Tr1w = pKF1->GetRightPose();
    //         Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
    //         Tll = T1w * Tw2;
    //         Tlr = T1w * Twr2;
    //         Trl = Tr1w * Tw2;
    //         Trr = Tr1w * Twr2;
    //     }
    //     Eigen::Matrix3f Rll = Tll.rotationMatrix(), Rlr  = Tlr.rotationMatrix(), Rrl  = Trl.rotationMatrix(), Rrr  = Trr.rotationMatrix();
    //     Eigen::Vector3f tll = Tll.translation(), tlr = Tlr.translation(), trl = Trl.translation(), trr = Trr.translation();

    //     // 初始化匹配计数器和结果容器
    //     int nmatches = 0;
    //     vector<bool> vbMatched2(pKF2->N, false);       // 记录KF2中被匹配的特征点
    //     vector<int> vMatches12(pKF1->N, -1);           // 记录KF1与KF2的匹配关系

    //     // 旋转一致性检查的直方图
    //     vector<int> rotHist[HISTO_LENGTH];             // 旋转一致性直方图
    //     for (int i = 0; i < HISTO_LENGTH; i++) {
    //         rotHist[i].reserve(500);
    //     }
    //     const float factor = HISTO_LENGTH/360.0f;      // 用于计算旋转角度的归一化因子

    //     // 词袋特征向量迭代器
    //     DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    //     DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    //     DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
    //     DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

    //     // 遍历词袋模型节点
    //     while (f1it != f1end && f2it != f2end) {
    //         if (f1it->first == f2it->first) { // 仅在词袋节点相同时匹配
    //             // 提取该词袋节点下的特征点
    //             vector<cv::KeyPoint> kpsKF1, kpsKF2;
    //             cv::Mat descKF1, descKF2;
    //             vector<size_t> idx1Vec, idx2Vec;  // 存储特征点索引

    //             // 遍历 KF1 的特征点
    //             for (size_t i1 = 0; i1 < f1it->second.size(); i1++) {
    //                 const size_t idx1 = f1it->second[i1];
    //                 MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

    //                 // 如果已有MapPoint，则跳过该特征点
    //                 if (pMP1)
    //                     continue;

    //                 const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1] >= 0);
    //                 if (bOnlyStereo && !bStereo1)
    //                 continue;

    //                 const cv::KeyPoint& kp1 = (pKF1->NLeft == -1) ? pKF1->mvKeysUn[idx1]
    //                                                               : (idx1 < pKF1->NLeft) ? pKF1->mvKeys[idx1]
    //                                                                                      : pKF1->mvKeysRight[idx1 - pKF1->NLeft];
            
    //                 // 过滤掉不符合条件的特征点
    //                 if (!bStereo1) {
    //                     kpsKF1.push_back(kp1);  // 添加左图特征点
    //                     descKF1.push_back(pKF1->mDescriptors.row(idx1));  // 添加描述符
    //                      idx1Vec.push_back(idx1);  // 保存索引
    //                 }
    //             }

    //             // 遍历 KF2 的特征点
    //             for (size_t i2 = 0; i2 < f2it->second.size(); i2++) {
    //                 const size_t idx2 = f2it->second[i2];
    //                 MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

    //                 // 如果已有MapPoint，则跳过该特征点
    //                 if (pMP2)
    //                     continue;

    //                 const bool bStereo2 = (!pKF2->mpCamera2 && pKF2->mvuRight[idx2] >= 0);
    //                 if (bOnlyStereo && !bStereo2)
    //                 continue;

    //                 const cv::KeyPoint& kp2 = (pKF2->NLeft == -1) ? pKF2->mvKeysUn[idx2]
    //                                                               : (idx2 < pKF2->NLeft) ? pKF2->mvKeys[idx2]
    //                                                                                      : pKF2->mvKeysRight[idx2 - pKF2->NLeft];

    //                 // 过滤掉不符合条件的特征点
    //                 if (!bStereo2) {
    //                     kpsKF2.push_back(kp2);  // 添加右图特征点
    //                     descKF2.push_back(pKF2->mDescriptors.row(idx2));  // 添加描述符
    //                     idx2Vec.push_back(idx2);
    //                 }
    //             }
    //             if (!kpsKF1.empty() &&!descKF1.empty() &&!kpsKF2.empty() &&!descKF2.empty())
    //             {
    //                 torch::Tensor desc1 = torch::from_blob(const_cast<float*>(descKF1.ptr<float>()), {static_cast<long>(descKF1.rows), static_cast<long>(descKF1.cols)}, torch::kFloat);
    //                 torch::Tensor desc2 = torch::from_blob(const_cast<float*>(descKF2.ptr<float>()), {static_cast<long>(descKF2.rows), static_cast<long>(descKF2.cols)}, torch::kFloat);

    //                 int imgWidth = pKF1->mnMaxX - pKF1->mnMinX;  // 图像的宽度
    //                 int imgHeight = pKF1->mnMaxY - pKF1->mnMinY; // 图像的高度

    //                 // LightGlue 需要尺寸作为输入
    //                 torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
    //                 torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);

    //                 std::cout << "777777" << std::endl;
    //                 // 使用 LightGlue 批量匹配该词袋节点下的所有特征点
    //                 std::vector<cv::DMatch> matches = lg.matchDescriptors(kpsKF1, desc1, kpsKF2, desc2, size1, size2);
                    
    //                 // 遍历匹配结果
    //                 for (const cv::DMatch &m : matches) {
    //                     size_t idx1 = f1it->second[m.queryIdx]; // KF1中特征点索引
    //                     size_t idx2 = f2it->second[m.trainIdx]; // KF2中特征点索引

    //                     const cv::KeyPoint &kp1 = kpsKF1[m.queryIdx]; // KF1中的关键点
    //                     const cv::KeyPoint &kp2 = kpsKF2[m.trainIdx]; // KF2中的关键点

    //                     const bool bRight1 = (pKF1->NLeft == -1 || idx1 < pKF1->NLeft) ? false : true;
    //                     const bool bRight2 = (pKF2->NLeft == -1 || idx2 < pKF2->NLeft) ? false : true;
    //                     const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1] >= 0);
    //                     const bool bStereo2 = (!pKF2->mpCamera2 && pKF2->mvuRight[idx2] >= 0);


    //                     // 如果两个关键帧都不是双目帧，且 KF1 和 KF2 没有第二个相机，进行极线约束
    //                     if (!bStereo1 && !bStereo2 && !pKF2->mpCamera2) {
    //                         const float distex = ep(0) - kp2.pt.x;
    //                         const float distey = ep(1) - kp2.pt.y;
    //                         if (distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave]) continue;
    //                     }

    //                     // 检查双目相机的匹配条件
    //                     if (pKF1->mpCamera2 && pKF2->mpCamera2) {    
    //                         if (bRight1 && bRight2) {
    //                             R12 = Rrr;
    //                             t12 = trr;
    //                             T12 = Trr;

    //                             pCamera1= pKF1->mpCamera2;
    //                             pCamera2= pKF2->mpCamera2;
    //                         } 
    //                         else if (bRight1 && !bRight2) {
    //                             R12 = Rrl;
    //                             t12 = trl;
    //                             T12 = Trl;

    //                             pCamera1= pKF1->mpCamera2;
    //                             pCamera2= pKF2->mpCamera2;
    //                         } 
    //                         else if (!bRight1 && bRight2) {
    //                             R12 = Rlr;
    //                             t12 = tlr;
    //                             T12 = Tlr;

    //                             pCamera1= pKF1->mpCamera2;
    //                             pCamera2= pKF2->mpCamera2;
    //                         } 
    //                         else {
    //                             R12 = Rll;
    //                             t12 = tll;
    //                             T12 = Tll;

    //                             pCamera1= pKF1->mpCamera2;
    //                             pCamera2= pKF2->mpCamera2;
    //                         }
                            

    //                         // 如果通过了双目相机几何约束，继续处理匹配
    //                         if (bCoarse || pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])) {
    //                             vMatches12[idx1] = idx2;    // 记录匹配对
    //                             vbMatched2[idx2] = true;    // 标记KF2中已匹配的特征点
    //                             nmatches++;                 // 匹配计数+1

    //                             // 旋转一致性检查
    //                             if (mbCheckOrientation) {
    //                                 float rot = kp1.angle - kp2.angle;
    //                                 if (rot < 0.0f) rot += 360.0f;
    //                                 int bin = round(rot * factor); // 计算旋转角度的直方图索引
    //                                 if (bin == HISTO_LENGTH) bin = 0;
    //                                 rotHist[bin].push_back(idx1);
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //             // 移动到下一个词袋节点
    //             f1it++;
    //             f2it++;
    //         }
    //         else if (f1it->first < f2it->first) {
    //             f1it = vFeatVec1.lower_bound(f2it->first);
    //         }
    //         else {
    //             f2it = vFeatVec2.lower_bound(f1it->first);
    //         }
    //     }

    //     // 处理旋转一致性检查结果
    //     if (mbCheckOrientation) {
    //         int ind1 = -1, ind2 = -1, ind3 = -1;
    //         ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
    //         for (int i = 0; i < HISTO_LENGTH; i++) {
    //             if (i == ind1 || i == ind2 || i == ind3) continue;
    //             for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
    //                 vMatches12[rotHist[i][j]] = -1;
    //                 nmatches--;
    //             }
    //         }
    //     }

    //     // 保存最终的匹配对
    //     vMatchedPairs.clear();
    //     vMatchedPairs.reserve(nmatches);
    //     for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    //         if (vMatches12[i] < 0) continue;
    //         vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
    //     }

    //     std::cout << "LightGlue Matches  7: " << nmatches << std::endl;
    //     return nmatches; // 返回匹配数量
    // }
    int LGmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
    {
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();
        Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
        Eigen::Vector3f Cw = pKF1->GetCameraCenter();
        Eigen::Vector3f C2 = T2w * Cw;

        Eigen::Vector2f ep = pKF2->mpCamera->project(C2);
        Sophus::SE3f T12;
        Sophus::SE3f Tll, Tlr, Trl, Trr;
        Eigen::Matrix3f R12; // for fastest computation
        Eigen::Vector3f t12; // for fastest computation

        GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

        if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
            T12 = T1w * Tw2;
            R12 = T12.rotationMatrix();
            t12 = T12.translation();
        }
        else{
            Sophus::SE3f Tr1w = pKF1->GetRightPose();
            Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
            Tll = T1w * Tw2;
            Tlr = T1w * Twr2;
            Trl = Tr1w * Tw2;
            Trr = Tr1w * Twr2;
        }

        Eigen::Matrix3f Rll = Tll.rotationMatrix(), Rlr  = Tlr.rotationMatrix(), Rrl  = Trl.rotationMatrix(), Rrr  = Trr.rotationMatrix();
        Eigen::Vector3f tll = Tll.translation(), tlr = Tlr.translation(), trl = Trl.translation(), trr = Trr.translation();

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node
        int nmatches=0;
        vector<bool> vbMatched2(pKF2->N,false);
        vector<int> vMatches12(pKF1->N,-1);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it!=f1end && f2it!=f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];

                    MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                    // If there is already a MapPoint skip
                    if(pMP1)
                    {
                        continue;
                    }

                    const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                    if(bOnlyStereo)
                        if(!bStereo1)
                            continue;

                    const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                    : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                             : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                    const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                       : true;

                    const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;

                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2];

                        MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                        // If we have already matched or there is a MapPoint skip
                        if(vbMatched2[idx2] || pMP2)
                            continue;

                        const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                        if(bOnlyStereo)
                            if(!bStereo2)
                                continue;

                        const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                        const int dist = DescriptorDistance(d1,d2);

                        if(dist>TH_LOW || dist>bestDist)
                            continue;

                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                        : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                 : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                        const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                           : true;

                        if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                        {
                            const float distex = ep(0)-kp2.pt.x;
                            const float distey = ep(1)-kp2.pt.y;
                            if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            {
                                continue;
                            }
                        }

                        if(pKF1->mpCamera2 && pKF2->mpCamera2){
                            if(bRight1 && bRight2){
                                R12 = Rrr;
                                t12 = trr;
                                T12 = Trr;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else if(bRight1 && !bRight2){
                                R12 = Rrl;
                                t12 = trl;
                                T12 = Trl;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera;
                            }
                            else if(!bRight1 && bRight2){
                                R12 = Rlr;
                                t12 = tlr;
                                T12 = Tlr;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else{
                                R12 = Rll;
                                t12 = tll;
                                T12 = Tll;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera;
                            }

                        }

                        if(bCoarse || pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])) // MODIFICATION_2
                        {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    if(bestIdx2>=0)
                    {
                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                        : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                     : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                        vMatches12[idx1]=bestIdx2;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = kp1.angle-kp2.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vMatches12[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }

        }

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
        {
            if(vMatches12[i]<0)
                continue;
            vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
        }
        std::cout << "777777" << std::endl;
        std::cout<< "匹配对数为：" << nmatches << std::endl;
        return nmatches;
    }
    // int LGmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
    // {
    //     // 获取相机姿态和相机内参
    //     GeometricCamera* pCamera;
    //     Sophus::SE3f Tcw;
    //     Eigen::Vector3f Ow;

    //     if(bRight) {
    //         Tcw = pKF->GetRightPose();
    //         Ow = pKF->GetRightCameraCenter();
    //         pCamera = pKF->mpCamera2;
    //     }
    //     else {
    //         Tcw = pKF->GetPose();
    //         Ow = pKF->GetCameraCenter();
    //         pCamera = pKF->mpCamera;
    //     }

    //     // 获取相机内参
    //     const float &fx = pKF->fx;
    //     const float &fy = pKF->fy;
    //     const float &cx = pKF->cx;
    //     const float &cy = pKF->cy;
    //     const float &bf = pKF->mbf;

    //     int nFused = 0; // 记录成功融合的点数量

    //     const int nMPs = vpMapPoints.size(); // 获取输入地图点的数量

    //     // 准备批量处理的局部变量
    //     std::vector<cv::KeyPoint> kpMapPoints; // 存储所有地图点的投影
    //     std::vector<cv::KeyPoint> kpKeyFrame;  // 存储所有在局部区域内的关键帧特征点
    //     cv::Mat descMapPoints, descKeyFrame;   // 对应的描述子矩阵

    //     std::vector<int> mapPointIndices;      // 记录地图点的索引，用于后续匹配处理

    //     // 使用集合来避免重复插入
    //     std::set<std::pair<float, float>> uniqueKeypoints;

    //     // 遍历地图点，进行投影并在局部区域内收集特征点
    //     for(int i = 0; i < nMPs; i++) {
    //         MapPoint* pMP = vpMapPoints[i];

    //         // 跳过无效或已经存在于关键帧中的地图点
    //         if(!pMP || pMP->isBad() || pMP->IsInKeyFrame(pKF))
    //             continue;

    //         // 获取地图点的世界坐标并转换为相机坐标
    //         Eigen::Vector3f p3Dw = pMP->GetWorldPos();
    //         Eigen::Vector3f p3Dc = Tcw * p3Dw;

    //         // 确保深度为正
    //         if(p3Dc(2) < 0.0f)
    //             continue;

    //         const float invz = 1 / p3Dc(2); // 计算深度的倒数

    //         // 将地图点投影到图像平面
    //         const Eigen::Vector2f uv = pCamera->project(p3Dc);

    //         // 确保投影点在图像范围内
    //         if(!pKF->IsInImage(uv(0), uv(1)))
    //             continue;

    //         // 计算右图像投影
    //         const float ur = uv(0) - bf * invz;

    //         // 确保距离在合理范围内
    //         const float maxDistance = pMP->GetMaxDistanceInvariance();
    //         const float minDistance = pMP->GetMinDistanceInvariance();
    //         Eigen::Vector3f PO = p3Dw - Ow;
    //         const float dist3D = PO.norm();

    //         if(dist3D < minDistance || dist3D > maxDistance)
    //             continue;

    //         // 确保视角小于60度
    //         Eigen::Vector3f Pn = pMP->GetNormal();
    //         if(PO.dot(Pn) < 0.5 * dist3D)
    //             continue;

    //         // 预测地图点的尺度等级
    //         int nPredictedLevel = pMP->PredictScale(dist3D, pKF);
    //         float scale = pKF->mvScaleFactors[nPredictedLevel];

    //         // 在该区域内搜索关键帧的特征点
    //         const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
    //         const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0), uv(1), radius, bRight);

    //         if(vIndices.empty())
    //             continue;
    //         // 创建地图点的 `cv::KeyPoint`，并设置尺度和响应
    //         cv::KeyPoint kpMapPoint(uv(0), uv(1), scale);
    //         kpMapPoints.push_back(kpMapPoint);
    //         descMapPoints.push_back(pMP->GetDescriptor());
    //         mapPointIndices.push_back(i); // 记录地图点索引

    //         // 收集在局部区域内的关键帧特征点及其描述符
    //         for(const auto& idx : vIndices) {
    //             const cv::KeyPoint &kp = (pKF->NLeft == -1)? pKF->mvKeysUn[idx]
    //                                                         : (!bRight)? pKF->mvKeys[idx]
    //                                                                     : pKF->mvKeysRight[idx];
    //             if (uniqueKeypoints.find(std::make_pair(kp.pt.x, kp.pt.y)) == uniqueKeypoints.end())
    //             {
    //                 kpKeyFrame.push_back(kp);
    //                 descKeyFrame.push_back(pKF->mDescriptors.row(idx));
    //                 uniqueKeypoints.insert(std::make_pair(kp.pt.x, kp.pt.y));
    //             }

    //         }
    //     }

    //     std::cout << "8关键帧中地图点的数量: " << kpKeyFrame.size() << std::endl;
    //     std::cout << "8关键帧中描述符的数量: " << descKeyFrame.rows << std::endl;
    //     // 输出在匹配前当前帧投影区域内的特征点数量和描述符数量
    //     std::cout << "8地图点关键点: " << kpMapPoints.size() << std::endl;
    //     std::cout << "8地图点描述符: " << descMapPoints.rows << std::endl;
    //     std::cout << "前5个关键帧关键点坐标（如果存在）:" << std::endl;
    //     for(int i = 0; i < std::min(5, (int)kpKeyFrame.size()); i++) {
    //         std::cout << "KeyFrame " << i << ": (" << kpKeyFrame[i].pt.x << ", " << kpKeyFrame[i].pt.y << ")" << std::endl;
    //     }

    //     std::cout << "前5个地图点坐标（如果存在）:" << std::endl;
    //     for(int i = 0; i < std::min(5, (int)kpMapPoints.size()); i++) {
    //         std::cout << "MapPoint " << i << ": (" << kpMapPoints[i].pt.x << ", " << kpMapPoints[i].pt.y << ")" << std::endl;
    //     }

    //     // 只有在关键点和描述符都不为空的情况下才进行后续匹配操作
    //     if (!kpMapPoints.empty() &&!kpKeyFrame.empty() &&!descMapPoints.empty() &&!descKeyFrame.empty())
    //     {
    //         // 转换描述符格式并匹配
    //         torch::Tensor descTensorKF = torch::from_blob(const_cast<float*>(descKeyFrame.ptr<float>()), {static_cast<long>(descKeyFrame.rows), static_cast<long>(descKeyFrame.cols)}, torch::kFloat);
    //         torch::Tensor descTensorMP = torch::from_blob(const_cast<float*>(descMapPoints.ptr<float>()), {static_cast<long>(descMapPoints.rows), static_cast<long>(descMapPoints.cols)}, torch::kFloat);

    //         int imgWidth = pKF->mnMaxX - pKF->mnMinX;  // 图像的宽度
    //         int imgHeight = pKF->mnMaxY - pKF->mnMinY; // 图像的高度

    //         // LightGlue 需要尺寸作为输入
    //         torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
    //         torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
    //         std::cout << "888888" << std::endl;
    //         std::vector<cv::DMatch> matches = lg.matchDescriptors(kpKeyFrame, descTensorKF, kpMapPoints, descTensorMP, size1, size2);

    //         // 处理匹配结果
    //         for(const auto& match : matches) {
    //             int idxMapPoint = mapPointIndices[match.queryIdx]; // 地图点索引
    //             int idxKeyFrame = match.trainIdx;                  // 关键帧特征点索引

    //             MapPoint* pMP = vpMapPoints[idxMapPoint];

    //             // 检查距离阈值是否满足
    //             if(match.distance < TH_LOW)
    //                 continue;

    //             // 确定关键帧的正确索引
    //             if(bRight) idxKeyFrame += pKF->NLeft;

    //             MapPoint* pMPinKF = pKF->GetMapPoint(idxKeyFrame);

    //             // 如果关键帧中已经存在地图点，则进行替换或融合
    //             if(pMPinKF) {
    //                 if(!pMPinKF->isBad()) {
    //                     if(pMPinKF->Observations() > pMP->Observations())
    //                         pMP->Replace(pMPinKF);
    //                     else
    //                         pMPinKF->Replace(pMP);
    //                 }
    //             }
    //             else {
    //                 // 否则将地图点添加到关键帧中
    //                 pMP->AddObservation(pKF, idxKeyFrame);
    //                 pKF->AddMapPoint(pMP, idxKeyFrame);
    //             }

    //             nFused++; // 更新成功融合的点数量
    //         }
    //     }

    //     std::cout << "LightGlue Matches  8: " << nFused << std::endl;
    //     return nFused; // 返回成功融合的地图点数量
    // }
    int LGmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
    {
        GeometricCamera* pCamera;
        Sophus::SE3f Tcw;
        Eigen::Vector3f Ow;

        if(bRight){
            Tcw = pKF->GetRightPose();
            Ow = pKF->GetRightCameraCenter();
            pCamera = pKF->mpCamera2;
        }
        else{
            Tcw = pKF->GetPose();
            Ow = pKF->GetCameraCenter();
            pCamera = pKF->mpCamera;
        }

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        int nFused=0;

        const int nMPs = vpMapPoints.size();

        // For debbuging
        int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
        for(int i=0; i<nMPs; i++)
        {
            MapPoint* pMP = vpMapPoints[i];

            if(!pMP)
            {
                count_notMP++;
                continue;
            }

            if(pMP->isBad())
            {
                count_bad++;
                continue;
            }
            else if(pMP->IsInKeyFrame(pKF))
            {
                count_isinKF++;
                continue;
            }

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0f)
            {
                count_negdepth++;
                continue;
            }

            const float invz = 1/p3Dc(2);

            const Eigen::Vector2f uv = pCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
            {
                count_notinim++;
                continue;
            }

            const float ur = uv(0)-bf*invz;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist3D = PO.norm();

            // Depth must be inside the scale pyramid of the image
            if(dist3D<minDistance || dist3D>maxDistance) {
                count_dist++;
                continue;
            }

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist3D)
            {
                count_normal++;
                continue;
            }

            int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius,bRight);

            if(vIndices.empty())
            {
                count_notidx++;
                continue;
            }

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                size_t idx = *vit;
                const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                              : (!bRight) ? pKF -> mvKeys[idx]
                                                                          : pKF -> mvKeysRight[idx];

                const int &kpLevel= kp.octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                if(pKF->mvuRight[idx]>=0)
                {
                    // Check reprojection error in stereo
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float &kpr = pKF->mvuRight[idx];
                    const float ex = uv(0)-kpx;
                    const float ey = uv(1)-kpy;
                    const float er = ur-kpr;
                    const float e2 = ex*ex+ey*ey+er*er;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                        continue;
                }
                else
                {
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float ex = uv(0)-kpx;
                    const float ey = uv(1)-kpy;
                    const float e2 = ex*ex+ey*ey;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                        continue;
                }

                if(bRight) idx += pKF->NLeft;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if(bestDist<=TH_LOW)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if(pMPinKF)
                {
                    if(!pMPinKF->isBad())
                    {
                        if(pMPinKF->Observations()>pMP->Observations())
                            pMP->Replace(pMPinKF);
                        else
                            pMPinKF->Replace(pMP);
                    }
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP,bestIdx);
                }
                nFused++;
            }
            else
                count_thcheck++;

        }
        std::cout << "888888" << "匹配对数为：" << nFused << std::endl;
        return nFused;
    }
    float LGmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        float dist = (float)cv::norm(a, b, cv::NORM_L2);

        return dist;
    }

    /**
     * @brief 闭环矫正中使用。将当前关键帧闭环匹配上的关键帧及其共视关键帧组成的地图点投影到当前关键帧，融合地图点
     * 
     * @param[in] pKF                   当前关键帧
     * @param[in] Scw                   当前关键帧经过闭环Sim3 后的世界到相机坐标系的Sim变换
     * @param[in] vpPoints              与当前关键帧闭环匹配上的关键帧及其共视关键帧组成的地图点
     * @param[in] th                    搜索范围系数
     * @param[out] vpReplacePoint       替换的地图点
     * @return int                      融合（替换和新增）的地图点数目
     */
    int LGmatcher::Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
    {
        // 获取相机内参
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // 分解 Scw
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // 关键帧中已存在的地图点集合
        const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();
        int nFused = 0;

        // 准备LightGlue的输入
        std::vector<cv::KeyPoint> kpsKF = pKF->mvKeysUn;  // 获取关键帧的所有特征点
        cv::Mat descKF;
        for (size_t i = 0; i < kpsKF.size(); ++i) {
            descKF.push_back(pKF->mDescriptors.row(i));
        }
    
        std::vector<cv::KeyPoint> kpsMP;
        cv::Mat descMP;

        // 遍历每个候选 MapPoint 进行投影和局部匹配
        for (MapPoint* pMP : vpPoints) {
            if (pMP->isBad() || spAlreadyFound.count(pMP)) 
                continue;

            // 地图点到图像的投影检查
            Eigen::Vector3f p3Dc = Tcw * pMP->GetWorldPos();
            if (p3Dc(2) < 0.0f)
                continue;
        
            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);
            if (!pKF->IsInImage(uv(0), uv(1)))
                continue;

            // 将地图点投影和描述符加入 LightGlue 匹配输入
            kpsMP.push_back(cv::KeyPoint(uv(0), uv(1), 1.0f));
            descMP.push_back(pMP->GetDescriptor());
        }

        // 转换描述符格式并匹配
        torch::Tensor descTensorKF = torch::from_blob(const_cast<float*>(descKF.ptr<float>()), {static_cast<long>(descKF.rows), static_cast<long>(descKF.cols)}, torch::kFloat);
        torch::Tensor descTensorMP = torch::from_blob(const_cast<float*>(descMP.ptr<float>()), {static_cast<long>(descMP.rows), static_cast<long>(descMP.cols)}, torch::kFloat);

        int imgWidth = pKF->mnMaxX - pKF->mnMinX;  // 图像的宽度
        int imgHeight = pKF->mnMaxY - pKF->mnMinY; // 图像的高度

        // LightGlue 需要尺寸作为输入
        torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
        torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);

        std::cout << "999999" << std::endl;
        std::vector<cv::DMatch> matches = lg.matchDescriptors(kpsKF, descTensorKF, kpsMP, descTensorMP, size1, size2);

        // 处理匹配结果
        for (const auto& match : matches) {
            int idxKF = match.queryIdx;
            int idxMP = match.trainIdx;

            MapPoint* pMP = vpPoints[idxMP];
            if (!pMP || pMP->isBad())
                continue;

            MapPoint* pMPinKF = pKF->GetMapPoint(idxKF);
            if (pMPinKF) {
                if (!pMPinKF->isBad())
                    vpReplacePoint[idxMP] = pMPinKF;
            } else {
                pMP->AddObservation(pKF, idxKF);
                pKF->AddMapPoint(pMP, idxKF);
            }
            nFused++;
        }

        std::cout << "LightGlue Matches  9: " << nFused << std::endl;
        return nFused;
    }

    /**在两个关键帧之间找到地图点的对应关系，闭环检测，重定位，全局优化
     * @brief 
     * @param[in] pKF1              当前帧
     * @param[in] pKF2              闭环候选帧
     * @param[in] vpMatches12       i表示匹配的pKF1 特征点索引，vpMatches12[i]表示匹配的地图点，null表示没有匹配
     * @param[in] s12               pKF2 到 pKF1 的Sim 变换中的尺度
     * @param[in] R12               pKF2 到 pKF1 的Sim 变换中的旋转矩阵
     * @param[in] t12               pKF2 到 pKF1 的Sim 变换中的平移向量
     * @param[in] th                搜索窗口的倍数
     * @return int                  新增的匹配点对数目
     */
    int LGmatcher::SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
    {
        // 提取 pKF1 的相机参数
        const float &fx = pKF1->fx;  // 相机的焦距 fx
        const float &fy = pKF1->fy;  // 相机的焦距 fy
        const float &cx = pKF1->cx;  // 相机的光心 cx
        const float &cy = pKF1->cy;  // 相机的光心 cy

        // 获取两个关键帧的位姿
        Sophus::SE3f T1w = pKF1->GetPose();  // 第一个关键帧的位姿
        Sophus::SE3f T2w = pKF2->GetPose();  // 第二个关键帧的位姿

        // 计算两个关键帧之间的 Sim3 变换
        Sophus::Sim3f S21 = S12.inverse();  // pKF2 到 pKF1 的 Sim3 变换

        // 获取两个关键帧中的地图点
        const std::vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();  // pKF1 的地图点
        const int N1 = vpMapPoints1.size();  // pKF1 中地图点的数量

        const std::vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();  // pKF2 的地图点
        const int N2 = vpMapPoints2.size();  // pKF2 中地图点的数量

        // 标记两个关键帧中已经匹配的地图点
        std::vector<bool> vbAlreadyMatched1(N1, false);  // 标记 pKF1 中是否已匹配的点
        ::vector<bool> vbAlreadyMatched2(N2, false);  // 标记 pKF2 中是否已匹配的点

        // 遍历已有的匹配，更新标记
        for (int i = 0; i < N1; i++)
        {
            MapPoint* pMP = vpMatches12[i];  // 已匹配的地图点
            if (pMP)
            {
                vbAlreadyMatched1[i] = true;  // 标记 pKF1 中已匹配
                int idx2 = std::get<0>(pMP->GetIndexInKeyFrame(pKF2));  // 获取 pKF2 中的索引
                if (idx2 >= 0 && idx2 < N2)
                    vbAlreadyMatched2[idx2] = true;  // 标记 pKF2 中已匹配
            }
        }

        // 准备存储 LightGlue 匹配所需的关键点和描述符

        std::vector<cv::KeyPoint> keypoints1;  // pKF1 的关键点
        std::vector<cv::KeyPoint> keypoints2;  // pKF2 的关键点
        cv::Mat descriptors1;  // pKF1 的关键点和描述符
        cv::Mat descriptors2;  // pKF2 的关键点和描述符

        // 为 pKF1 提取有效的关键点和描述符
        std::vector<int> vnMatch1(N1, -1);  // 记录匹配情况
        for (int i1 = 0; i1 < N1; i1++)
        {
            MapPoint* pMP = vpMapPoints1[i1];  // 获取 pKF1 中的地图点

            // 如果地图点无效或已匹配则跳过
            if (!pMP || vbAlreadyMatched1[i1] || pMP->isBad())
                continue;

            // 计算该地图点在 pKF2 中的投影位置
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();  // 地图点的世界坐标
            Eigen::Vector3f p3Dc1 = T1w * p3Dw;  // 转换到 pKF1 的相机坐标系
            Eigen::Vector3f p3Dc2 = S21 * p3Dc1;  // 转换到 pKF2 的相机坐标系

            // 如果投影在 pKF2 后方则跳过
            if (p3Dc2(2) < 0.0)
                continue;

            // 计算投影到 pKF2 图像中的像素坐标
            const float invz = 1.0 / p3Dc2(2);  // 深度的倒数
            const float u = fx * p3Dc2(0) * invz + cx;  // x 坐标
            const float v = fy * p3Dc2(1) * invz + cy;  // y 坐标

            // 如果投影超出图像边界则跳过
            if (!pKF2->IsInImage(u, v))
                continue;

            // 将该点的关键点位置和描述符存入容器
            keypoints1.push_back(pKF1->mvKeysUn[i1]);  // pKF1 中的关键点
            descriptors1.push_back(pMP->GetDescriptor());  // 地图点的描述符
        }

        // 为 pKF2 提取有效的关键点和描述符，步骤类似 pKF1
        for (int i2 = 0; i2 < N2; i2++)
        {
            MapPoint* pMP = vpMapPoints2[i2];

            if (!pMP || vbAlreadyMatched2[i2] || pMP->isBad())
                continue;

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc2 = T2w * p3Dw;
            Eigen::Vector3f p3Dc1 = S12 * p3Dc2;

            if (p3Dc1(2) < 0.0)
                continue;

            const float invz = 1.0 / p3Dc1(2);
            const float u = fx * p3Dc1(0) * invz + cx;
            const float v = fy * p3Dc1(1) * invz + cy;

            if (!pKF1->IsInImage(u, v))
                continue;

            keypoints2.push_back(pKF2->mvKeysUn[i2]);  // pKF2 中的关键点
            descriptors2.push_back(pMP->GetDescriptor());  // 地图点的描述符
        }

    
        // 将描述符转换为 torch::Tensor
        torch::Tensor desc1 = torch::from_blob(const_cast<float*>(descriptors1.ptr<float>()), {static_cast<long>(descriptors1.rows), static_cast<long>(descriptors1.cols)}, torch::kFloat);
        torch::Tensor desc2 = torch::from_blob(const_cast<float*>(descriptors2.ptr<float>()), {static_cast<long>(descriptors2.rows), static_cast<long>(descriptors2.cols)}, torch::kFloat);

        int imgWidth = pKF1->mnMaxX - pKF1->mnMinX;  // 图像的宽度
        int imgHeight = pKF1->mnMaxY - pKF1->mnMinY; // 图像的高度

        // LightGlue 需要尺寸作为输入
        torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
        torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);

        // 使用 LightGlue 进行批量匹配
        std::cout << "10 10 10" << std::endl;
        std::vector<cv::DMatch> matches = lg.matchDescriptors(keypoints1, desc1, keypoints2, desc2, size1, size2);
        // 处理 LightGlue 返回的匹配结果
        int nFound = 0;
        for (const auto& match : matches)
        {
            int idx1 = match.queryIdx;  // pKF1 中的索引
            int idx2 = match.trainIdx;  // pKF2 中的索引

            // 如果尚未匹配，记录该匹配
            if (vnMatch1[idx1] == -1)
            {
                vpMatches12[idx1] = vpMapPoints2[idx2];  // 保存匹配的地图点
                nFound++;  // 匹配数量增加
            }
        }
        std::cout << "LightGlue Matches  10: " << nFound << std::endl;
        return nFound;  // 返回成功匹配的数量
    }

    /**
     * @brief 将上一帧跟踪的地图点投影到当前帧，并且搜索匹配点。用于跟踪前一帧
     * 步骤
     * Step 1 建立旋转直方图，用于检测旋转一致性
     * Step 2 计算当前帧和前一帧的平移向量
     * Step 3 对于前一帧的每一个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
     * Step 4 根据相机的前后前进方向来判断搜索尺度范围
     * Step 5 遍历候选匹配点，寻找距离最小的最佳匹配点 
     * Step 6 计算匹配点旋转角度差所在的直方图
     * Step 7 进行旋转一致检测，剔除不一致的匹配
     * @param[in] CurrentFrame          当前帧
     * @param[in] LastFrame             上一帧
     * @param[in] th                    搜索范围阈值，默认单目为7，双目15
     * @param[in] bMono                 是否为单目
     * @return int                      成功匹配的数量
     */
    // int LGmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
    // {
    //     int nmatches = 0;

    //     // Rotation Histogram (to check rotation consistency)
    //     vector<int> rotHist[HISTO_LENGTH];
    //     for (int i = 0; i < HISTO_LENGTH; i++)
    //         rotHist[i].reserve(500);
    //     const float factor = 1.0f / HISTO_LENGTH;

    //     // 获取当前帧和上一帧的姿态
    //     const Sophus::SE3f Tcw = CurrentFrame.GetPose();  // 当前帧的姿态
    //     const Sophus::SE3f Tlw = LastFrame.GetPose();     // 上一帧的姿态

    //     // 创建存储地图点和当前帧特征点的容器
    //     std::vector<cv::KeyPoint> mapKeypoints;       // 用于存储地图点投影到当前帧的关键点
    //     cv::Mat mapDescriptors;                       // 用于存储地图点的描述符

    //     std::vector<cv::KeyPoint> currentKeypoints;   // 用于存储当前帧的关键点
    //     cv::Mat currentDescriptors;                   // 用于存储当前帧的描述符
    //     std::vector<int> currentIndices;              // 当前帧特征点的索引

    //     // 使用集合来避免重复插入
    //     std::set<std::pair<float, float>> uniqueKeypoints;


    //     // 遍历上一帧的所有MapPoints，并将它们投影到当前帧
    //     for (int i = 0; i < LastFrame.mvpMapPoints.size(); i++)
    //     {
    //         MapPoint* pMP = LastFrame.mvpMapPoints[i];

    //         // 检查MapPoint是否为空或为离群点
    //         if (!pMP || LastFrame.mvbOutlier[i])
    //             continue;

    //         // 获取MapPoint的世界坐标并将其投影到当前帧中
    //         Eigen::Vector3f x3Dw = pMP->GetWorldPos();  // 获取地图点的世界坐标
    //         Eigen::Vector3f x3Dc = Tcw * x3Dw;          // 将地图点转换到当前帧的相机坐标系下

    //         const float invzc = 1.0 / x3Dc(2);          // 计算深度的倒数
    //         if (invzc < 0)  // 确保深度为正
    //             continue;

    //         // 将3D点投影到当前帧的像素平面
    //         Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);  // 投影到像素坐标

    //         // 判断投影点是否在图像范围内
    //         if (uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
    //             continue;
    //         if (uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
    //             continue;

    //         // 搜索当前帧中在投影点周围的特征点，用于后续匹配
    //         const float radius = th * fabs(invzc);  // 根据深度动态调整搜索半径
    //         vector<size_t> vIndices = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius);  // 获取投影点附近的特征点

    //         if (vIndices.empty())  // 如果在该区域没有找到特征点，跳过该MapPoint
    //             continue;

    //         // 将地图点投影位置和描述符保存起来，用于批量匹配
    //         mapKeypoints.emplace_back(cv::KeyPoint(uv(0), uv(1), 1.0f));  // 投影位置作为关键点
    //         mapDescriptors.push_back(pMP->GetDescriptor());  // 存储描述符

    //         // 将投影区域内的当前帧特征点保存起来
    //         for (size_t idx : vIndices)
    //         {
    //              const cv::KeyPoint& kp = CurrentFrame.mvKeysUn[idx];
    //             // 使用关键点坐标来判断是否已添加过该点
    //             if (uniqueKeypoints.find(std::make_pair(kp.pt.x, kp.pt.y)) == uniqueKeypoints.end())
    //             {
    //                 currentKeypoints.push_back(kp);  // 当前帧中的关键点
    //                 currentDescriptors.push_back(CurrentFrame.mDescriptors.row(idx));  // 当前帧中的描述符
    //                 currentIndices.push_back(idx);  // 保存对应的索引
    //                 // 将关键点坐标加入集合，避免重复
    //                 uniqueKeypoints.insert(std::make_pair(kp.pt.x, kp.pt.y));
    //             }
    //         }
    //     }
    //     // 输出在匹配前地图点投影的特征点数量和描述符数量
    //     std::cout << "Before matching, mapKeypoints num: " << mapKeypoints.size() << std::endl;
    //     std::cout << "Before matching, mapDescriptors num: " << mapDescriptors.rows << std::endl;
    //     // 输出在匹配前当前帧投影区域内的特征点数量和描述符数量
    //     std::cout << "Before matching, currentKeypoints num: " << currentKeypoints.size() << std::endl;
    //     std::cout << "Before matching, currentDescriptors num: " << currentDescriptors.rows << std::endl;

    //     int imgWidth = CurrentFrame.mnMaxX - CurrentFrame.mnMinX;  // 图像的宽度
    //     int imgHeight = CurrentFrame.mnMaxY - CurrentFrame.mnMinY; // 图像的高度

    //     // LightGlue 需要尺寸作为输入
    //     torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
    //     torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
    //     // 使用LightGlue对所有地图点和投影区域内的当前帧特征点进行批量匹配
    //     if (!mapDescriptors.empty() && !currentDescriptors.empty())
    //     {
    //         // 先将描述符转换为torch::Tensor
    //         torch::Tensor mapDescriptorsTensor = torch::from_blob(const_cast<float*>(mapDescriptors.ptr<float>()), {static_cast<long>(mapDescriptors.rows), static_cast<long>(mapDescriptors.cols)}, torch::kFloat);
    //         torch::Tensor currentDescriptorsTensor = torch::from_blob(const_cast<float*>(currentDescriptors.ptr<float>()), {static_cast<long>(currentDescriptors.rows), static_cast<long>(currentDescriptors.cols)}, torch::kFloat);
    //         std::cout << "11 11 11" << std::endl;
    //         // 调用LightGlue进行描述符匹配
    //         std::vector<cv::DMatch> matches = lg.matchDescriptors(mapKeypoints, mapDescriptorsTensor, currentKeypoints, currentDescriptorsTensor, size1, size2);
    //         std::cout << "11匹配成功: " <<std:: endl;
    //         // 处理匹配结果
    //         for (const auto& match : matches)
    //         {
    //             int idxCurrent = currentIndices[match.trainIdx];  // 获取当前帧的匹配点索引
    //             int idxMapPoint = match.queryIdx;  // 获取地图点的索引

    //             // 确保这个特征点还未被分配其他MapPoint
    //             if (CurrentFrame.mvpMapPoints[idxCurrent] == NULL)
    //             {
    //                 MapPoint* pMP = LastFrame.mvpMapPoints[idxMapPoint];

    //                 // 匹配成功，当前帧中的特征点关联上该MapPoint
    //                 CurrentFrame.mvpMapPoints[idxCurrent] = pMP;
    //                 nmatches++;

    //                 // 如果需要检查旋转一致性
    //                 if (mbCheckOrientation)
    //                 {
    //                     const cv::KeyPoint &kpLF = mapKeypoints[idxMapPoint];  // 地图点的投影关键点
    //                     const cv::KeyPoint &kpCF = CurrentFrame.mvKeysUn[idxCurrent];  // 当前帧中的匹配关键点

    //                     float rot = kpLF.angle - kpCF.angle;
    //                     if (rot < 0.0)
    //                         rot += 360.0f;
    //                     int bin = round(rot * factor);
    //                     if (bin == HISTO_LENGTH)
    //                         bin = 0;
    //                     assert(bin >= 0 && bin < HISTO_LENGTH);
    //                     rotHist[bin].push_back(idxCurrent);
    //                 }
    //             }
    //         }
    //     }
    //     // 应用旋转一致性约束
    //     if (mbCheckOrientation)
    //     {
    //         int ind1 = -1, ind2 = -1, ind3 = -1;
    //         ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    //         for (int i = 0; i < HISTO_LENGTH; i++)
    //         {
    //             if (i != ind1 && i != ind2 && i != ind3)
    //             {
    //                 for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
    //                 {
    //                     CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
    //                     nmatches--;
    //                 }
    //             }
    //         }
    //     }

    //     std::cout << "LightGlue Matches  11: " << nmatches << std::endl;
    //     return nmatches;  // 返回成功匹配的数量
    // }
    int LGmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
    {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        const Eigen::Vector3f twc = Tcw.inverse().translation();

        const Sophus::SE3f Tlw = LastFrame.GetPose();
        const Eigen::Vector3f tlc = Tlw * twc;

        const bool bForward = tlc(2)>CurrentFrame.mb && !bMono;
        const bool bBackward = -tlc(2)>CurrentFrame.mb && !bMono;

        for(int i=0; i<LastFrame.N; i++)
        {
            MapPoint* pMP = LastFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!LastFrame.mvbOutlier[i])
                {
                    // Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const float xc = x3Dc(0);
                    const float yc = x3Dc(1);
                    const float invzc = 1.0/x3Dc(2);

                    if(invzc<0)
                        continue;

                    Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
                        continue;
                    if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
                        continue;

                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                     : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                    vector<size_t> vIndices2;

                    if(bForward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave);
                    else if(bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave);
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1);

                    if(vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                    {
                        const size_t i2 = *vit;

                        if(CurrentFrame.mvpMapPoints[i2])
                            if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                                continue;

                        if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2]>0)
                        {
                            const float ur = uv(0) - CurrentFrame.mbf*invzc;
                            const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                            if(er>radius)
                                continue;
                        }

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                        : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                            cv::KeyPoint kpCF = (CurrentFrame.Nleft == -1) ? CurrentFrame.mvKeysUn[bestIdx2]
                                                                           : (bestIdx2 < CurrentFrame.Nleft) ? CurrentFrame.mvKeys[bestIdx2]
                                                                                                             : CurrentFrame.mvKeysRight[bestIdx2 - CurrentFrame.Nleft];
                            float rot = kpLF.angle-kpCF.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                    if(CurrentFrame.Nleft != -1){
                        Eigen::Vector3f x3Dr = CurrentFrame.GetRelativePoseTrl() * x3Dc;
                        Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dr);

                        int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                             : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                        // Search in a window. Size depends on scale
                        float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                        vector<size_t> vIndices2;

                        if(bForward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave, -1,true);
                        else if(bBackward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave, true);
                        else
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1, true);

                        const cv::Mat dMP = pMP->GetDescriptor();

                        int bestDist = 256;
                        int bestIdx2 = -1;

                        for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                        {
                            const size_t i2 = *vit;
                            if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                                if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations()>0)
                                    continue;

                            const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                            const int dist = DescriptorDistance(dMP,d);

                            if(dist<bestDist)
                            {
                                bestDist=dist;
                                bestIdx2=i2;
                            }
                        }

                        if(bestDist<=TH_HIGH)
                        {
                            CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft]=pMP;
                            nmatches++;
                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                            : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                                cv::KeyPoint kpCF = CurrentFrame.mvKeysRight[bestIdx2];

                                float rot = kpLF.angle-kpCF.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdx2  + CurrentFrame.Nleft);
                            }
                        }

                    }
                }
            }
        }

        //Apply rotation consistency
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }
        }
        std::cout << "11 11 11" << "匹配对数为：" << nmatches << std::endl;
        return nmatches;
    }
    /**
     * @brief 通过投影的方式将关键帧中未匹配的地图点投影到当前帧中,进行匹配，并通过旋转直方图进行筛选
     * 
     * @param[in] CurrentFrame          当前帧
     * @param[in] pKF                   参考关键帧
     * @param[in] sAlreadyFound         已经找到的地图点集合，不会用于PNP
     * @param[in] th                    匹配时搜索范围，会乘以金字塔尺度
     * @param[in] ORBdist               匹配的ORB描述子距离应该小于这个阈值    
     * @return int                      成功匹配的数量
     */
    // int LGmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
    // {
    //     int nmatches = 0;

    //     // 获取当前帧的位姿Tcw
    //     const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    //     Eigen::Vector3f Ow = Tcw.inverse().translation();  // 当前相机的中心位置

    //     // 旋转直方图，用于检查旋转一致性
    //     vector<int> rotHist[HISTO_LENGTH];
    //     for (int i = 0; i < HISTO_LENGTH; i++)
    //         rotHist[i].reserve(500);
    //     const float factor = 1.0f / HISTO_LENGTH;

    //     // 获取关键帧中的地图点
    //     const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    //     // 存储投影点的关键点和描述符，用于批量匹配
    //     std::vector<cv::KeyPoint> mapKeypoints;   // 投影的地图点位置作为关键点
    //     cv::Mat mapDescriptors;                   // 地图点的描述符

    //     // 当前帧局部区域内的关键点和描述符
    //     std::vector<cv::KeyPoint> localKeypoints;
    //     cv::Mat localDescriptors;
    //     std::vector<int> mapPointIndices;         // 记录与当前帧投影区域匹配的地图点索引

    //     // 使用集合来避免重复插入
    //     std::set<std::pair<float, float>> uniqueKeypoints;

    //     // 遍历关键帧中的每一个地图点
    //     for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    //     {
    //         MapPoint* pMP = vpMPs[i];

    //         // 跳过已经找到的地图点或无效的地图点
    //         if (pMP && !pMP->isBad() && !sAlreadyFound.count(pMP))
    //         {
    //             // 将地图点投影到当前帧
    //             Eigen::Vector3f x3Dw = pMP->GetWorldPos();  // 地图点的世界坐标
    //             Eigen::Vector3f x3Dc = Tcw * x3Dw;          // 将地图点转换到当前相机坐标系

    //             // 获取投影点的像素坐标
    //             const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

    //             // 判断投影点是否在图像范围内
    //             if (uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX ||
    //                 uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
    //                 continue;

    //             // 计算3D距离并预测尺度等级
    //             Eigen::Vector3f PO = x3Dw - Ow;
    //             float dist3D = PO.norm();
    //             const float maxDistance = pMP->GetMaxDistanceInvariance();
    //             const float minDistance = pMP->GetMinDistanceInvariance();

    //             // 深度必须在关键帧的地图点距离范围内
    //             if (dist3D < minDistance || dist3D > maxDistance)
    //                 continue;

    //             int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

    //             // 在投影点周围的局部区域内进行特征点选择
    //             const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];
    //             const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nPredictedLevel - 1, nPredictedLevel + 1);

    //             if (vIndices2.empty())
    //                 continue;

    //             // 存储地图点的描述符和投影点位置
    //             mapKeypoints.emplace_back(cv::KeyPoint(uv(0), uv(1), 1.0f));  // 投影位置作为关键点
    //             mapDescriptors.push_back(pMP->GetDescriptor());  // 存储描述符
    //             mapPointIndices.push_back(i);  // 记录地图点的索引

    //             // 遍历投影区域内的所有特征点，并存储
    //             for (auto idx : vIndices2)
    //             {
    //                 const cv::KeyPoint& kp = CurrentFrame.mvKeysUn[idx];
    //                 if (CurrentFrame.mvpMapPoints[idx] == nullptr && uniqueKeypoints.find(std::make_pair(kp.pt.x, kp.pt.y)) == uniqueKeypoints.end())  // 确保该特征点还没有匹配地图点
    //                 {
    //                     localKeypoints.push_back(CurrentFrame.mvKeysUn[idx]);
    //                     localDescriptors.push_back(CurrentFrame.mDescriptors.row(idx));  // 将特征点对应的描述符添加到局部描述符集合中
    //                     uniqueKeypoints.insert(std::make_pair(kp.pt.x, kp.pt.y));
    //                 }
    //             }
    //         }
    //     }

    //     // 如果没有可以匹配的特征点或地图点，直接返回
    //     if (localKeypoints.empty() || mapKeypoints.empty())
    //         return nmatches;

    //     // 转换描述符为torch::Tensor
    //     torch::Tensor localDescriptorsTensor = torch::from_blob(const_cast<float*>(localDescriptors.ptr<float>()), {static_cast<long>(localDescriptors.rows), static_cast<long>(localDescriptors.cols)}, torch::kFloat).clone();
    //     torch::Tensor mapDescriptorsTensor = torch::from_blob(const_cast<float*>(mapDescriptors.ptr<float>()), {static_cast<long>(mapDescriptors.rows), static_cast<long>(mapDescriptors.cols)}, torch::kFloat).clone();

    //     std::cout << "Number of keypoints in CF: " << localKeypoints.size() << std::endl;
    //     std::cout << "Number of descriptors in CF: " << localDescriptorsTensor.size(0) << std::endl;

    //     std::cout << "Number of keypoints in pKF: " << mapKeypoints.size() << std::endl;
    //     std::cout << "Number of descriptors in pKF: " << mapDescriptorsTensor.size(0) << std::endl;

    //     int imgWidth = CurrentFrame.mnMaxX - CurrentFrame.mnMinX;  // 图像的宽度
    //     int imgHeight = CurrentFrame.mnMaxY - CurrentFrame.mnMinY; // 图像的高度

    //     int imgWidth1 = pKF->mnMaxX - pKF->mnMinX;  // 图像的宽度
    //     int imgHeight1 = pKF->mnMaxY - pKF->mnMinY; // 图像的高度
    //     // LightGlue 需要尺寸作为输入
    //     torch::Tensor size1 = torch::tensor({static_cast<float>(imgWidth), static_cast<float>(imgHeight)}, torch::kFloat32);
    //     torch::Tensor size2 = torch::tensor({static_cast<float>(imgWidth1), static_cast<float>(imgHeight1)}, torch::kFloat32);
    //     std::cout << "12 12 12" << std::endl;
    //     // 使用LightGlue进行批量匹配
    //     std::vector<cv::DMatch> matches;
    //     matches = lg.matchDescriptors(localKeypoints, localDescriptorsTensor, mapKeypoints, mapDescriptorsTensor, size1, size2);


    //     // 遍历匹配结果，处理每个匹配点
    //     for (const auto& match : matches)
    //     {
    //         const int idxCurrent = match.queryIdx;  // 当前帧中匹配的特征点索引
    //         const int idxMap = mapPointIndices[match.trainIdx];  // 地图点的索引
    //         const float dist = match.distance;

    //         // 如果匹配距离小于阈值，则认为匹配成功
    //         if (dist >= 0.1)
    //         {
    //             MapPoint* pMP = vpMPs[idxMap];
    //             CurrentFrame.mvpMapPoints[idxCurrent] = pMP;  // 当前帧的特征点与地图点关联
    //             nmatches++;

    //             // 检查旋转一致性
    //             if (mbCheckOrientation)
    //             {
    //                 float rot = pKF->mvKeysUn[idxMap].angle - CurrentFrame.mvKeysUn[idxCurrent].angle;
    //                 if (rot < 0.0f)
    //                     rot += 360.0f;
    //                 int bin = round(rot * factor);
    //                 if (bin == HISTO_LENGTH)
    //                     bin = 0;
    //                 rotHist[bin].push_back(idxCurrent);
    //             }
    //         }
    //     }

    //     // 如果启用了旋转一致性检查，进行直方图筛选
    //     if (mbCheckOrientation)
    //     {
    //         int ind1 = -1, ind2 = -1, ind3 = -1;
    //         ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    //         for (int i = 0; i < HISTO_LENGTH; i++)
    //         {
    //             if (i != ind1 && i != ind2 && i != ind3)
    //             {
    //                 for (size_t j = 0; j < rotHist[i].size(); j++)
    //                 {
    //                     CurrentFrame.mvpMapPoints[rotHist[i][j]] = nullptr;
    //                     nmatches--;
    //                 }
    //             }
    //         }
    //     }

    //     std::cout << "LightGlue Matches  12: " << nmatches << std::endl;
    //     return nmatches;
    // }
    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
    {
        int nmatches = 0;

        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMPs[i];

            if(pMP)
            {
                if(!pMP->isBad() && !sAlreadyFound.count(pMP))
                {
                    //Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
                        continue;
                    if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
                        continue;

                    // Compute predicted scale level
                    Eigen::Vector3f PO = x3Dw-Ow;
                    float dist3D = PO.norm();

                    const float maxDistance = pMP->GetMaxDistanceInvariance();
                    const float minDistance = pMP->GetMinDistanceInvariance();

                    // Depth must be inside the scale pyramid of the image
                    if(dist3D<minDistance || dist3D>maxDistance)
                        continue;

                    int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                    // Search in a window
                    const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                    const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nPredictedLevel-1, nPredictedLevel+1);

                    if(vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                    {
                        const size_t i2 = *vit;
                        if(CurrentFrame.mvpMapPoints[i2])
                            continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=ORBdist)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }

                }
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                        nmatches--;
                    }
                }
            }
        }
        std::cout << "12 12 12" << std::endl;
        std::cout<< "匹配对数为：" << nmatches << std::endl;
        return nmatches;
    }
    void LGmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1=0;
        int max2=0;
        int max3=0;

        for(int i=0; i<L; i++)
        {
            const int s = histo[i].size();
            if(s>max1)
            {
                max3=max2;
                max2=max1;
                max1=s;
                ind3=ind2;
                ind2=ind1;
                ind1=i;
            }
            else if(s>max2)
            {
                max3=max2;
                max2=s;
                ind3=ind2;
                ind2=i;
            }
            else if(s>max3)
            {
                max3=s;
                ind3=i;
            }
        }

        if(max2<0.1f*(float)max1)
        {
            ind2=-1;
            ind3=-1;
        }
        else if(max3<0.1f*(float)max1)
        {
            ind3=-1;
        }
    }
    

} //namespace ORB_SLAM
