
#ifndef CMAKE_EXAMPLE_BOW_SEARCH_FN_H
#define CMAKE_EXAMPLE_BOW_SEARCH_FN_H

//
// specifically for ORBSLAM
//
#include "DBoW3/DBoW3.h"
#include <opencv2/opencv.hpp>
#include <pybind11/stl.h>

using namespace std;

class KeyPoint{
public:
    float angle;
    tuple<float, float> pt;
    int class_id;
    int octave;
    float response;
    float size;

    KeyPoint(float angle, tuple<float, float> pt, float response, float size=31, int class_id=-1, int octave=0):
    angle(angle),pt(pt),response(response),size(size),class_id(class_id),octave(octave){

    };
    ~ KeyPoint(){

    };
};

class BowSearchFn {
public:
    const int TH_HIGH = 100;
    const int TH_LOW = 50;
    const int HISTO_LENGTH = 30;

    // 要求最佳匹配和次佳匹配的比例
    float mfNNratio = 0.6;
    bool mbCheckOrientation = true;

    void SetNNRatio(float nnratio){
        mfNNratio = nnratio;
    }

    void SetCheckOrientation(bool checkOri) {
        mbCheckOrientation = checkOri;
    }
    /**
     * 返回F中每个keypoint能对应上KF的哪个KP
     * @param pKF
     * @param F
     * @return nmatches, F.N-dim vector<int>，-1 means not matched.
     */
    tuple<int, vector<int>> SearchByBoW_kf_f(
            const DBoW3::FeatureVector &vFeatVecKF, const DBoW3::FeatureVector &vFeatVecF,
            const vector<KeyPoint> &keysUnKF, const vector<KeyPoint> &keysUnF,
            const cv::Mat &descKF, const cv::Mat &descF) {

        // 保存F中每个keypoint能对应上KF的哪个KP
        vector<int> vpMapPointMatches(keysUnF.size(), -1);

        int nmatches = 0;

        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = HISTO_LENGTH / 360.0f;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        // 将属于同一节点(特定层)的ORB特征进行匹配
        // 四个iterator
        auto KFit = vFeatVecKF.begin();
        auto Fit = vFeatVecF.begin();
        auto KFend = vFeatVecKF.end();
        auto Fend = vFeatVecF.end();

        while (KFit != KFend && Fit != Fend) {
            if (KFit->first == Fit->first) //步骤1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
            {
                const vector<unsigned int> vIndicesKF = KFit->second;
                const vector<unsigned int> vIndicesF = Fit->second;

                // 步骤2：遍历KF中属于该node的特征点
                for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {
                    const unsigned int realIdxKF = vIndicesKF[iKF];

//                    MapPoint *pMP = vpMapPointsKF[realIdxKF]; // 取出KF中该特征对应的MapPoint

                    // TODO:这两个检查项注意下
//                    if(!pMP)
//                        continue;
//                    if(pMP->isBad())
//                        continue;

                    const cv::Mat &dKF = descKF.row(realIdxKF); // 取出KF中该特征对应的描述子

                    int bestDist1 = 256; // 最好的距离（最小距离）
                    int bestIdxF = -1;
                    int bestDist2 = 256; // 倒数第二好距离（倒数第二小距离）

                    // 步骤3：遍历F中属于该node的特征点，找到了最佳匹配点
                    for (size_t iF = 0; iF < vIndicesF.size(); iF++) {
                        const unsigned int realIdxF = vIndicesF[iF];

                        if (vpMapPointMatches[realIdxF] != -1)// 表明这个点已经被匹配过了，不再匹配，加快速度
                            continue;

                        const cv::Mat &dF = descF.row(realIdxF); // 取出F中该特征对应的描述子

                        const int dist = DescriptorDistance(dKF, dF); // 求描述子的距离

                        if (dist < bestDist1)// dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdxF = realIdxF;
                        } else if (dist < bestDist2)// bestDist1 < dist < bestDist2，更新bestDist2
                        {
                            bestDist2 = dist;
                        }
                    }

                    // 步骤4：根据阈值 和 角度投票剔除误匹配
                    if (bestDist1 <= TH_LOW) // 匹配距离（误差）小于阈值
                    {
                        // trick!
                        // 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                        if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
                            // 步骤5：更新特征点的MapPoint在KF的index
                            vpMapPointMatches[bestIdxF] = realIdxKF;

                            const KeyPoint &kp = keysUnKF[realIdxKF];

                            if (mbCheckOrientation) {
                                // trick!
                                // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                                // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                                float rot = kp.angle - keysUnF[bestIdxF].angle;// 该特征点的角度变化值
                                if (rot < 0.0)
                                    rot += 360.0f;
                                int bin = round(rot * factor);// 将rot分配到bin组
                                if (bin == HISTO_LENGTH)
                                    bin = 0;
                                assert(bin >= 0 && bin < HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxF);
                            }
                            nmatches++;
                        }
                    }

                }

                KFit++;
                Fit++;
            } else if (KFit->first < Fit->first) {
                KFit = vFeatVecKF.lower_bound(Fit->first);
            } else {
                Fit = vFeatVecF.lower_bound(KFit->first);
            }
        }

        // 根据方向剔除误匹配的点
        if (mbCheckOrientation) {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            // 计算rotHist中最大的三个的index
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++) {
                // 如果特征点的旋转角度变化量属于这三个组，则保留
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;

                // 将除了ind1 ind2 ind3以外的匹配点去掉
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                    vpMapPointMatches[rotHist[i][j]] = -1;
                    nmatches--;
                }
            }
        }

        return make_tuple(nmatches,vpMapPointMatches);
    }

    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++) {
            unsigned int v = *pa ^*pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    // 取出直方图中值最大的三个index
    void ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3) {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++) {
            const int s = histo[i].size();
            if (s > max1) {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            } else if (s > max2) {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            } else if (s > max3) {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float) max1) {
            ind2 = -1;
            ind3 = -1;
        } else if (max3 < 0.1f * (float) max1) {
            ind3 = -1;
        }
    }
};



#endif //CMAKE_EXAMPLE_BOW_SEARCH_FN_H
