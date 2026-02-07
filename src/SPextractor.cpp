#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "SPextractor.h"
#include "SuperPoint.h"

using namespace std;

namespace ORB_SLAM3
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

const float factorPI = (float)(CV_PI/180.f);

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;

}


SPextractor::SPextractor(int nfeatures, float scaleFactor, int nlevels, float iniThFAST, float minThFAST)
    : nfeatures(nfeatures), scaleFactor(scaleFactor), nlevels(nlevels), iniThFAST(iniThFAST), minThFAST(minThFAST)
{
    model = make_shared<SuperPoint>();

    try {
        torch::load(model, "/sly_slam/superpoint_new.pt");
        std::cout << "[SPDetector::detect] Model loaded successfully." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }



    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

}

int SPextractor::operator()(cv::InputArray _image, cv::InputArray _mask, 
                            std::vector<cv::KeyPoint>& _keypoints, cv::OutputArray _descriptors, 
                            std::vector<int>& vLappingArea)
{
    //cout << "[SPextractor]: Max Features: " << nfeatures << endl;
    if(_image.empty())
        return -1;

    cv::Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );

    cv::Mat descriptors;

    // Pre-compute the scale pyramid
    ComputePyramid(image);

    std::vector < std::vector<cv::KeyPoint> > allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints, descriptors);
    //ComputeKeyPointsOld(allKeypoints);

    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if( nkeypoints == 0 )
        _descriptors.release();
    else
    {
        // 创建 SuperPoint 的描述符 (256 维，浮点型)
        _descriptors.create(nkeypoints, 256, CV_32F); //superpoint的描述符维度是 256,使用 CV_32F（32位浮点数）
        descriptors.copyTo(_descriptors.getMat());
    }

    //_keypoints.clear();
    //_keypoints.reserve(nkeypoints);
    _keypoints = vector<cv::KeyPoint>(nkeypoints);

    int offset = 0;
    //Modified for speeding up stereo fisheye matching
    int monoIndex = 0, stereoIndex = nkeypoints-1;
    for (int level = 0; level < nlevels; ++level)
    {
        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image
        //Mat workingMat = mvImagePyramid[level].clone();
        //GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors
        //Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        //Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
        //computeDescriptors(workingMat, keypoints, desc, pattern);

        //offset += nkeypointsLevel;


        float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
        
        
        
        int i = 0;
        for (vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(),
                     keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){

            // Scale keypoint coordinates
            if (level != 0){
                keypoint->pt *= scale;
            }

            if(keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]){
                _keypoints.at(stereoIndex) = (*keypoint);
                //desc.row(i).copyTo(descriptors.row(stereoIndex));
                stereoIndex--;
            }
            else{
                _keypoints.at(monoIndex) = (*keypoint);
                //desc.row(i).copyTo(descriptors.row(monoIndex));
                monoIndex++;
            }
            i++;
        }
    }
    // cout << "[SPextractor]: extracted " << _keypoints.size() << " KeyPoints" << endl;
    return monoIndex;
}

void SPextractor::ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints, cv::Mat &_desc)
{
    allKeypoints.resize(nlevels);

    vector<cv::Mat> vDesc;

    const float W = 35;

    for (int level = 0; level < nlevels; ++level)
    {
        SPDetector detector(model);
        detector.detect(mvImagePyramid[level], false);

        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures*10);

        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        const int nCols = width/W;
        const int nRows = height/W;
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);

        for(int i=0; i<nRows; i++)
        {
            const float iniY =minBorderY+i*hCell;
            float maxY = iniY+hCell+6;

            if(iniY>=maxBorderY-3)
                continue;
            if(maxY>maxBorderY)
                maxY = maxBorderY;

            for(int j=0; j<nCols; j++)
            {
                const float iniX =minBorderX+j*wCell;
                float maxX = iniX+wCell+6;
                if(iniX>=maxBorderX-6)
                    continue;
                if(maxX>maxBorderX)
                    maxX = maxBorderX;

                std::vector<cv::KeyPoint> vKeysCell;

                //FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                //         vKeysCell,iniThFAST,true);
                detector.getKeyPoints(iniThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);

                /*if(bRight && j <= 13){
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,10,true);
                }
                else if(!bRight && j >= 16){
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,10,true);
                }
                else{
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,iniThFAST,true);
                }*/


                if(vKeysCell.empty())
                {
                    //FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                    //     vKeysCell,minThFAST,true);
                    detector.getKeyPoints(minThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                    /*if(bRight && j <= 13){
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,5,true);
                    }
                    else if(!bRight && j >= 16){
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,5,true);
                    }
                    else{
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,minThFAST,true);
                    }*/
                }
                if(!vKeysCell.empty())
                {
                    for(std::vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
                // 动态点的剔除，防止动态物体区域的特征点影响后续的跟踪和匹配
                float scale = mvScaleFactor[level]; //获取当前特征点所在图像层级的缩放因子
                for (auto vit = vToDistributeKeys.begin(); vit != vToDistributeKeys.end(); vit++)
                {
                    //minBorderX 和 minBorderY 是图像边界的偏移量，目的是将特征点的坐标调整到考虑边界偏移后的坐标系中
                    vit->pt.x += minBorderX;
                    vit->pt.y += minBorderY;
                    vit->pt *= scale; // 将特征点的坐标根据层次的缩放因子进行缩放
                }

                bool Find = false;  //用于标记是否找到位于动态区域内的特征点
                for (auto vit_kp = vToDistributeKeys.begin(); vit_kp != vToDistributeKeys.end();)
                {
                    for (auto vit_area = mvDynamicArea.begin(); vit_area != mvDynamicArea.end(); vit_area++)//内层循环遍历存储动态区域的容器
                    {
                        Find = false;
                        if (vit_area->contains(vit_kp->pt))
                        {
                            //如果特征点位于动态区域内，将 Find 设为 true，并使用 erase 函数将该特征点从 vToDistributeKeys 中删除
                            Find = true;
                            vit_kp = vToDistributeKeys.erase(vit_kp);
                            break;
                        }
                    }

                    if (!Find) //如果当前特征点不在任何动态区域内，则迭代器 vit_kp 前进到下一个特征点
                    {
                        ++vit_kp;
                    }
                }

                //计算缩放因子的倒数 scale_inverse。这是为了将特征点的坐标从原图重新还原到金字塔层次的图像中
                float scale_inverse = 1 / scale;
                for (auto vit = vToDistributeKeys.begin(); vit != vToDistributeKeys.end(); vit++)
                {
                    vit->pt *= scale_inverse;
                    vit->pt.x -= minBorderX;
                    vit->pt.y -= minBorderY;
                }


            }
        }

        std::vector<cv::KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        //将特征点通过八叉树进行空间上的均匀分布
        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            keypoints[i].octave=level;
            keypoints[i].size = scaledPatchSize;
        }

        cv::Mat desc;
        detector.computeDescriptors(keypoints, desc);
        vDesc.push_back(desc);
    }

    cv::vconcat(vDesc, _desc);//将所有金字塔层次的描述符通过vconcat进行拼接，形成最终的描述符矩阵
    // compute orientations
    //for (int level = 0; level < nlevels; ++level)
        //computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

static bool compareNodes(pair<int,ExtractorNode*>& e1, pair<int,ExtractorNode*>& e2){
    if(e1.first < e2.first){
        return true;
    }
    else if(e1.first > e2.first){
        return false;
    }
    else{
        if(e1.second->UL.x < e2.second->UL.x){
            return true;
        }
        else{
            return false;
        }
    }
}

vector<cv::KeyPoint> SPextractor::DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                                         const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    const float hX = static_cast<float>(maxX-minX)/nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

        //Associate points to childs
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);
                    if(n1.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {

            while(!bFinish)
            {

                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end(),compareNodes);
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    std::vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        std::vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void SPextractor::ComputePyramid(cv::Mat image)
{
    
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        cv::Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        cv::Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);

            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                            cv::BORDER_REFLECT_101);
        }
    }
}


} // namespace ORB_SLAM3
