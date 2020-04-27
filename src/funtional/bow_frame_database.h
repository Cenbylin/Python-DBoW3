
#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>
#include "DBoW3/DBoW3.h"
#include <pybind11/iostream.h>

#include "wrapper/wrapper_vocabulary.h"

class BoWFrame {
public:
    DBoW3::BowVector mBowVec;
    int mnId;

    // Variables used by the keyframe database
    // Only change in database
    long unsigned int mnLoopQuery = 0;
    int mnLoopWords = 0;
    float mLoopScore = 0.0;
    long unsigned int mnRelocQuery = 0;
    int mnRelocWords = 0;
    float mRelocScore;

    BoWFrame(DBoW3::BowVector &mBowVec, int mnId) : mBowVec(mBowVec), mnId(mnId) {}
};

class BowFrameDatabase {
public:

    BowFrameDatabase(Vocabulary &voc) : mpVoc(&voc) {
        mvInvertedFile.resize(voc.size()); // number of words
    }


    /**
    * @brief 关键帧被删除后，更新数据库的倒排索引
    * @param pKF 关键帧
    */
    void erase(BoWFrame *pKF) {

        // Erase elements in the Inverse File for the entry
        // 每一个BoWFrame包含多个words，遍历mvInvertedFile中的这些words，然后在word中删除该BoWFrame
        for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end();
             vit != vend; vit++) {
            // List of keyframes that share the word
            list<BoWFrame *> &lKFs = mvInvertedFile[vit->first];

            for (list<BoWFrame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
                if (pKF == *lit) {
                    lKFs.erase(lit);
                    break;
                }
            }
        }
    }

    void clear() {
        mvInvertedFile.clear();// mvInvertedFile[i]表示包含了第i个word id的所有关键帧
        mvInvertedFile.resize(mpVoc->size());// mpVoc：预先训练好的词典
    }


    /**
    * @brief 根据关键帧的词包，更新数据库的倒排索引
    * @param pKF 关键帧，全程只用到vector和mnID
    */
    void add(DBoW3::BowVector &mBowVec, int mnId) {
        BoWFrame *pKF = new BoWFrame(mBowVec, mnId);
        // mvInvertedFile[i]表示包含了第i个word id的所有关键帧

        // 为每一个word添加该BoWFrame。倒排索引！
        for (DBoW3::BowVector::const_iterator vit = mBowVec.begin(), vend = mBowVec.end(); vit != vend; vit++){
            mvInvertedFile[vit->first].push_back(pKF);

            mmID2BoWFrame[mnId] = pKF;
        }
    }

    /**
     * @brief 在重定位中找到与该帧相似的关键帧--步骤1：寻找相似关键帧
     *
     * 1. 找出和当前帧具有公共单词的所有关键帧
     * 2. 只和具有共同单词较多的关键帧进行相似度计算
     * 待做：
     * 3. 将与关键帧相连（权值最高）的前十个关键帧归为一组，计算累计得分
     * 4. 只返回累计得分较高的组中分数最高的关键帧
     * @param F 需要重定位的帧
     * @return  相似度，mnID
     * @see III-E Bags of Words Place Recognition
     */
    list<pair<float, int>> DetectRelocalizationCandidates_similarFrame(DBoW3::BowVector &mBowVec, int mnId) {
        // 记得delete
        BoWFrame *F = new BoWFrame(mBowVec, mnId);
        // 相对于关键帧的闭环检测DetectLoopCandidates，重定位检测中没法获得相连的关键帧
        list<BoWFrame *> lKFsSharingWords;// 用于保存可能与F形成回环的候选帧（只要有相同的word，且不属于局部相连帧）

        // Search all keyframes that share a word with current frame
        // 步骤1：找出和当前帧具有公共单词的所有关键帧
        {
            // words是检测图像是否匹配的枢纽，遍历该pKF的每一个word
            // BowVector map<WordId, WordValue>
            for (DBoW3::BowVector::const_iterator vit = F->mBowVec.begin(), vend = F->mBowVec.end();
                 vit != vend; vit++) {
                // 提取所有包含该word的BoWFrame
                list<BoWFrame *> &lKFs = mvInvertedFile[vit->first];
                for (auto pKFi : lKFs) {
                    if (pKFi->mnRelocQuery != F->mnId)// pKFi还没有标记为pKF的候选帧
                    {
                        pKFi->mnRelocWords = 0;
                        pKFi->mnRelocQuery = F->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                    pKFi->mnRelocWords++;
                }
            }
        }
        if (lKFsSharingWords.empty())
            return list<pair<float, int>>();

        // Only compare against those keyframes that share enough words
        // 步骤2：统计所有闭环候选帧中与当前帧F具有共同单词最多的单词数，并以此决定阈值
        int maxCommonWords = 0;
        for (list<BoWFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
             lit != lend; lit++) {
            if ((*lit)->mnRelocWords > maxCommonWords)
                maxCommonWords = (*lit)->mnRelocWords;
        }

        int minCommonWords = maxCommonWords * 0.8f;

//        list<pair<float, BoWFrame *>> lScoreAndMatch;
        list<pair<float, int>> lScoreAndMatch_mnid;  // 暴露接口只返回mnid

        int nscores = 0;

        // Compute similarity score.
        // 步骤3：遍历所有闭环候选帧，挑选出共有单词数大于阈值minCommonWords且单词匹配度大于minScore存入lScoreAndMatch
        for (list<BoWFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
             lit != lend; lit++) {
            BoWFrame *pKFi = *lit;

            // 当前帧F只和具有共同单词较多的关键帧进行比较，需要大于minCommonWords
            if (pKFi->mnRelocWords > minCommonWords) {
                nscores++;// 这个变量后面没有用到
                float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
                pKFi->mRelocScore = si;
//                lScoreAndMatch.push_back(make_pair(si, pKFi));

                lScoreAndMatch_mnid.push_back(make_pair(si, pKFi->mnId));
            }
        }

        delete F;

        // list<pair<float, BoWFrame *>> 对应python是list[(float, BoWFrame)]

        return lScoreAndMatch_mnid;
    }

    /**
     * @brief 在重定位中找到与该帧相似的关键帧--步骤2：从相似关键帧展开共视关键帧，返回高分
     * 3. 将与关键帧相连（权值最高）的前(best)n个关键帧归为一组，计算累计得分
     * 4. 只返回累计得分较高的组中分数最高的关键帧
     * @param F 需要重定位的帧
     * @param lScoreAndMatch_mnid <相似度，mnID>
     * @param bestCovisibilityKeyFramesmnID <mnID, 前n个共视的list[mnID]>
     * @return
     * @see III-E Bags of Words Place Recognition
     */
    vector<BoWFrame> DetectRelocalizationCandidates_score(DBoW3::BowVector &mBowVec, int mnId,
                                                          list<pair<float, int>> lScoreAndMatch_mnid, list<pair<int, list<int>>> bestCovisibilityKeyFramesmnID) {
        // 记得delete
        BoWFrame *F = new BoWFrame(mBowVec, mnId);

        list<pair<float, BoWFrame *>> lScoreAndMatch;
        // lScoreAndMatch = process(lScoreAndMatch_mnid)
        for (list<pair<float, int>>::iterator it = lScoreAndMatch_mnid.begin(), itend = lScoreAndMatch_mnid.end();
             it != itend;
             it++) {
            lScoreAndMatch.push_back(make_pair(it->first, mmID2BoWFrame[it->second]));
        }

        // 将前端传来的一堆mnid转为bowframe
        std::map<int, list<BoWFrame *>> bestCovisibilityKeyFrames;
        for (list<pair<int, list<int>>>::iterator it = bestCovisibilityKeyFramesmnID.begin(),
                     itend = bestCovisibilityKeyFramesmnID.end(); it != itend; it++) {
            list<BoWFrame *> tmp;
            for (list<int>::iterator it1 = (it->second).begin(), itend1 = (it->second).end();
                 it1 != itend1; it1++) {
                BoWFrame * f = mmID2BoWFrame[*it1];
                tmp.push_back(f);
            }

            bestCovisibilityKeyFrames[it->first] = tmp;
        }

        if (lScoreAndMatch.empty())
            return vector<BoWFrame>();

        list<pair<float, BoWFrame *>> lAccScoreAndMatch;
        float bestAccScore = 0;

        // Lets now accumulate score by covisibility
        // 步骤4：计算候选帧组得分，得到最高组得分bestAccScore，并以此决定阈值minScoreToRetain
        // 单单计算当前帧和某一关键帧的相似性是不够的，这里将与关键帧相连（权值最高，共视程度最高）的前十个关键帧归为一组，计算累计得分
        // 具体而言：lScoreAndMatch中每一个BoWFrame都把与自己共视程度较高的帧归为一组，每一组会计算组得分并记录该组分数最高的BoWFrame，记录于lAccScoreAndMatch
        for (list<pair<float, BoWFrame *> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end();
             it != itend;
             it++) {
            BoWFrame *pKFi = it->second;
//            vector<BoWFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
            list<BoWFrame *> vpNeighs = bestCovisibilityKeyFrames[pKFi->mnId];

            float bestScore = it->first; // 该组最高分数
            float accScore = bestScore;  // 该组累计得分
            BoWFrame *pBestKF = pKFi;    // 该组最高分数对应的关键帧
            for (list<BoWFrame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
                BoWFrame *pKF2 = *vit;
                if (pKF2->mnRelocQuery != F->mnId)
                    continue;

                accScore += pKF2->mRelocScore;// 只有pKF2也在闭环候选帧中，才能贡献分数
                if (pKF2->mRelocScore > bestScore)// 统计得到组里分数最高的BoWFrame
                {
                    pBestKF = pKF2;
                    bestScore = pKF2->mRelocScore;
                }

            }
            lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
            if (accScore > bestAccScore) // 记录所有组中组得分最高的组
                bestAccScore = accScore; // 得到所有组中最高的累计得分
        }

        // Return all those keyframes with a score higher than 0.75*bestScore
        // 步骤5：得到组得分大于阈值的，组内得分最高的关键帧
        float minScoreToRetain = 0.75f * bestAccScore;
        set<BoWFrame *> spAlreadyAddedKF;
        vector<BoWFrame> vpRelocCandidates;
        vpRelocCandidates.reserve(lAccScoreAndMatch.size());
        for (list<pair<float, BoWFrame *> >
             ::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it !=
                                                                                         itend;
             it++) {
            const float &si = it->first;
            // 只返回累计得分大于minScoreToRetain的组中分数最高的关键帧 0.75*bestScore
            if (si > minScoreToRetain) {
                BoWFrame *pKFi = it->second;
                if (!spAlreadyAddedKF.count(pKFi))// 判断该pKFi是否已经在队列中了
                {
                    vpRelocCandidates.push_back(*pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }

        delete F;
        return vpRelocCandidates;
    }
protected:

    // Associated vocabulary
    Vocabulary *mpVoc; ///< 预先训练好的词典

    // Inverted file
    std::vector<list<BoWFrame *> > mvInvertedFile; ///< 倒排索引，mvInvertedFile[i]表示包含了第i个word id的所有关键帧

    // 外部调用的关联
    std::map<int, BoWFrame *> mmID2BoWFrame;
};

#endif
