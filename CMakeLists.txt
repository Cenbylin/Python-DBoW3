cmake_minimum_required(VERSION 2.8.12)
project(PyDBoW3)

add_subdirectory(pybind11)
# ----------------------------------------------------------------------------
#   Find Dependencies -- opencv
# ----------------------------------------------------------------------------
find_package(OpenCV 4 REQUIRED)
IF (USE_CONTRIB)
    add_definitions(-DUSE_CONTRIB)
ENDIF ()
if (NOT OpenCV_VERSION VERSION_LESS "3.0")
    ADD_DEFINITIONS(-DOPENCV_VERSION_3)
    SET(OPENCV_VERSION_3 ON)
ELSE ()
    SET(OPENCV_VERSION_3 OFF)
ENDIF ()

include_directories(${OpenCV_INCLUDE_DIRS})

SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${OpenCV_LIBS})
# ----------------------------------------------------------------------------
#   Find Dependencies -- python numpy
# ----------------------------------------------------------------------------
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
SET(REQUIRED_LIrm BRARIES ${PYTHON_LIBRARIES})

include(Cmake/FindNumPy.cmake)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
# ----------------------------------------------------------------------------
#   Find Dependencies -- DBoW3
# ----------------------------------------------------------------------------
#find_package(DBoW3 REQUIRED)
#include_directories(${DBoW3_INCLUDE_DIRS})
#SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${DBoW3_LIBS})
set(CMAKE_MACOSX_RPATH 1)
add_subdirectory(DBoW3)
include_directories(DBoW3/include)
SET(REQUIRED_LIBRARIES DBoW3)

# ----------------------------------------------------------------------------
#   source
# ----------------------------------------------------------------------------
#SET(SOURCE
#        xxx.cpp
#        )

SET(SOURCE_PYBIND
        ${PROJECT_SOURCE_DIR}/src/pydbow3.cpp
        ${PROJECT_SOURCE_DIR}/src/ndarray_converter.cpp
        )

include_directories(
        ${PROJECT_SOURCE_DIR}/src/
)

pybind11_add_module(PyDBoW3 ${SOURCES} ${SOURCE_PYBIND})
target_link_libraries(PyDBoW3 PRIVATE ${REQUIRED_LIBRARIES})


