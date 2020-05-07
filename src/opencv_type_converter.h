# ifndef __CVTYPE_CONVERTER_H__
# define __CVTYPE_CONVERTER_H__

#include <Python.h>
#include <opencv2/core/core.hpp>
#include <pybind11/pybind11.h>


namespace py = pybind11;
namespace pybind11 { namespace detail {
    
template <> struct type_caster<cv::KeyPoint> {
    public:

    PYBIND11_TYPE_CASTER(cv::KeyPoint, _("cv2.KeyPoint"));

        bool load(handle src, bool) {
            py::tuple pt = reinterpret_borrow<py::tuple>(src.attr("pt"));
            auto x = pt[0].cast<float>();
            auto y = pt[1].cast<float>();
            auto size = src.attr("size").cast<float>();
            auto angle = src.attr("angle").cast<float>();
            auto response = src.attr("response").cast<float>();
            auto octave = src.attr("octave").cast<int>();
            auto class_id = src.attr("class_id").cast<int>();
            // (float x, float y, float _size, float _angle, float _response, int _octave, int _class_id)
            value = cv::KeyPoint(x, y, size, angle, response, octave, class_id);

            return true;
        }

        static handle cast(const cv::KeyPoint &kp, return_value_policy, handle defval) {
            py::object classKP = py::module::import("cv2.KeyPoint");
            py::object cvKP = classKP(kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id);

            // equal to: return handle(cvKP.ptr());
            return {cvKP.ptr()};
        }
};
    
    
}} // namespace pybind11::detail

# endif
