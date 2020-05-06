// Wrapper for most external modules
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <exception>
// Opencv includes
#include <opencv2/opencv.hpp>

// DBoW3
#include "DBoW3/DBoW3.h"

// type catser for Numpy <=> cv:Mat
#include "ndarray_converter.h"

#include "wrapper/wrapper_vocabulary.h"
#include "wrapper/wrapper_database.h"
#include "funtional/bow_search_fn.h"
#include "funtional/bow_frame_database.h"

namespace py = pybind11;

PYBIND11_MODULE(DBoW3Py, m) {
    NDArrayConverter::init_numpy();

    m.doc() = R"pbdoc(
        Pybind11 for DBoW3
    )pbdoc";

    py::enum_<DBoW3::WeightingType>(m, "WeightingType")
            .value("TF_IDF", DBoW3::TF_IDF)
            .value("TF", DBoW3::TF)
            .value("IDF", DBoW3::IDF)
            .value("BINARY", DBoW3::BINARY);

    py::enum_<DBoW3::ScoringType>(m, "ScoringType")
            .value("L1_NORM", DBoW3::L1_NORM)
            .value("L2_NORM", DBoW3::L2_NORM)
            .value("CHI_SQUARE", DBoW3::CHI_SQUARE)
            .value("KL", DBoW3::KL)
            .value("BHATTACHARYYA", DBoW3::BHATTACHARYYA)
            .value("DOT_PRODUCT", DBoW3::DOT_PRODUCT);

    // Class
    py::class_<KeyPoint>(m, "KeyPoint")
            .def(py::init<float, tuple<float, float>, float, float, int, int>(),
                    py::arg("angle"),
                    py::arg("pt"),
                    py::arg("response"),
                    py::arg("size")=31.0,
                    py::arg("class_id")=-1,
                    py::arg("octave")=0);

    py::class_<BowSearchFn>(m, "BowSearchFn")
            .def(py::init<>())
            .def("SetNNRatio", &BowSearchFn::SetNNRatio, py::arg("nnratio"))
            .def("SetCheckOrientation", &BowSearchFn::SetCheckOrientation, py::arg("checkOri"))
            .def("search_BoW_kf_f", &BowSearchFn::SearchByBoW_kf_f,
                 py::arg("vFeatVecKF"),py::arg("vFeatVecF"),
                 py::arg("keysUnKF"),py::arg("keysUnF"),
                 py::arg("descKF"),py::arg("descF"))
                 ;

    py::class_<BoWFrame>(m, "BoWFrame")
            .def(py::init<DBoW3::BowVector &, int>(), py::arg("mBowVec"), py::arg("mnId"))
            .def_readonly("mnId", &BoWFrame::mnId)
            .def_readonly("mBowVec", &BoWFrame::mBowVec)
            ;
    py::class_<BowFrameDatabase>(m, "BowFrameDatabase")
            .def(py::init<Vocabulary&>(), py::arg("voc"))
            .def("DetectRelocalizationCandidates_similarFrame",
                    &BowFrameDatabase::DetectRelocalizationCandidates_similarFrame,
                    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
                    py::arg("mBowVec"), py::arg("mnId"),
                    py::return_value_policy::take_ownership)
            .def("DetectRelocalizationCandidates_score",
                 &BowFrameDatabase::DetectRelocalizationCandidates_score,
                 py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
                 py::arg("mBowVec"), py::arg("mnId"),
                 py::arg("lScoreAndMatch_mnid"), py::arg("bestCovisibilityKeyFramesmnID"),
                 py::return_value_policy::take_ownership)
            .def("add", &BowFrameDatabase::add,
                 py::arg("mBowVec"), py::arg("mnId"))
            .def("clear", &BowFrameDatabase::clear)
            ;

    py::class_<DBoW3::BowVector>(m, "BowVector")
            .def(py::init<>());
    py::class_<DBoW3::FeatureVector>(m, "FeatureVector")
            .def(py::init<>());

    py::class_<Vocabulary>(m, "Vocabulary")
            .def(py::init<std::string>(), py::arg("filename"))
            .def(py::init<int, int, DBoW3::WeightingType, DBoW3::ScoringType, std::string>(),
                 py::arg("k") = 10,
                 py::arg("L") = 5,
                 py::arg("weighting") = DBoW3::TF_IDF,
                 py::arg("scoring") = DBoW3::L1_NORM,
                 py::arg("path") = std::string())
            .def("load", &Vocabulary::load,
                 py::arg("path"))
            .def("save", &Vocabulary::save,
                 py::arg("path"), py::arg("binary_compressed") = true)
            .def("create", &Vocabulary::create,
                 py::arg("training_features"))
            .def("transform", &Vocabulary::transform,
                 py::arg("features"),py::return_value_policy::take_ownership)
            .def("transform", &Vocabulary::transform_bv_fv,
                 py::arg("features"),py::arg("levelup"),py::return_value_policy::take_ownership)
            .def("score", &Vocabulary::score,
                 py::arg("A"),py::arg("B"))
            .def("clear", &Vocabulary::clear)
            .def("size", &Vocabulary::size)
            ;


    py::class_<Database>(m, "Database")
            .def(py::init<std::string>(),py::arg("path") = std::string())
            .def("setVocabulary", &Database::setVocabulary,
                 py::arg("vocabulary"),py::arg("use_di"),
                 py::arg("di_levels") = 0)
            .def("save", &Database::save,py::arg("filename"))
            .def("load", &Database::load,py::arg("filename"))
            .def("loadVocabulary", &Database::loadVocabulary,
                 py::arg("filename"),py::arg("use_di"),
                 py::arg("di_levels") = 0)
            .def("add", &Database::add,py::arg("features"))
            .def("query", &Database::query,
                 py::arg("features"),
                 py::arg("max_results")=1,
                 py::arg("max_id")=-1,
                 py::return_value_policy::take_ownership);

    py::class_<DBoW3::Result>(m, "Result")
            .def_readonly("Id", &DBoW3::Result::Id)
            .def_readonly("Score", &DBoW3::Result::Score)
            .def_readonly("nWords", &DBoW3::Result::nWords)
            .def_readonly("bhatScore", &DBoW3::Result::bhatScore)
            .def_readonly("chiScore", &DBoW3::Result::chiScore)
            .def_readonly("sumCommonVi", &DBoW3::Result::sumCommonVi)
            .def_readonly("sumCommonWi", &DBoW3::Result::sumCommonWi)
            .def_readonly("expectedChiScore", &DBoW3::Result::expectedChiScore);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}