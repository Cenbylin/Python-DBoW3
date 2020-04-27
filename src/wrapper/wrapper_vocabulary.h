
#ifndef CMAKE_EXAMPLE_WRAPPER_VOCABULARY_H
#define CMAKE_EXAMPLE_WRAPPER_VOCABULARY_H

#include <pybind11/stl.h>
#include <pybind11/iostream.h>

class Vocabulary {
public:
    Vocabulary(int k = 10, int L = 5, DBoW3::WeightingType weighting = DBoW3::TF_IDF,
               DBoW3::ScoringType scoring = DBoW3::L1_NORM, const std::string &path = std::string()) {
        vocabulary = new DBoW3::Vocabulary(k, L, weighting, scoring);
        if (!path.empty())
            load(path);
    }
    Vocabulary(const std::string &filename) {
        vocabulary = new DBoW3::Vocabulary(filename);
//        TF_IDF
//        level:6 k:10
//        std::cout << "weighting" << vocabulary->getWeightingType() << std::endl;
//        std::cout << "levels:" << vocabulary->getDepthLevels() << std::endl;
//        std::cout << "K:" << vocabulary->getBranchingFactor() << std::endl;
    }

    ~Vocabulary() {
        delete vocabulary;
    }

    void create(const std::vector<cv::Mat> &training_features) {
        vocabulary->create(training_features);
    }

    void clear() {
        vocabulary->clear();
    }

    void load(const std::string &path) {
        vocabulary->load(path);
    }

    void save(const std::string &path, bool binary_compressed = true) {
        vocabulary->save(path, binary_compressed);
    }

    DBoW3::BowVector transform(const std::vector<cv::Mat> &features) {
        DBoW3::BowVector word;
        vocabulary->transform(features, word);
        return word;
    }

    std::tuple<DBoW3::BowVector,DBoW3::FeatureVector > transform_bv_fv(const std::vector<cv::Mat>& features, int levelsup) {
        DBoW3::BowVector bv;
        DBoW3::FeatureVector fv;
//        std::cout<< "transform from feature of " << features.size() << std::endl;

        assert(features[0].rows == 1);
        vocabulary->transform(features, bv, fv, levelsup);

//        std::cout<< "transform to bowvector of " << bv.size() << std::endl;
        return std::make_tuple(bv, fv);
    }

    double score(const DBoW3::BowVector &A, const DBoW3::BowVector &B) {
        return vocabulary->score(A, B);
    }

    unsigned int size(){
        return vocabulary->size();
    }

    DBoW3::Vocabulary *vocabulary;
};

#endif //CMAKE_EXAMPLE_WRAPPER_VOCABULARY_H
