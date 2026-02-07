cmake_minimum_required(VERSION 3.1)
project(yolo_orb_slam3)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

# Check C++17 support
set(CMAKE_CXX_STANDARD 17)           
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)  

find_package(CUDA REQUIRED)

set(Torch_DIR "${PROJECT_SOURCE_DIR}/../../../Thirdparty/libtorch/share/cmake/Torch")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  sensor_msgs
  image_transport
  cv_bridge
)

find_package(OpenCV 4.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Torch REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp tf sensor_msgs image_transport cv_bridge
)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
${PROJECT_SOURCE_DIR}/../../../include/CameraModels
${PROJECT_SOURCE_DIR}/../../../Thirdparty/Sophus
${Pangolin_INCLUDE_DIRS}
${TORCH_INCLUDE_DIRS}
${catkin_INCLUDE_DIRS}
${CUDA_INCLUDE_DIRS}
)

set(LIBS 
${OpenCV_LIBS}
${TORCH_LIBRARIES} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM3.so
${catkin_LIBRARIES}
${CUDA_LIBRARIES}
-lboost_system
)
add_executable(Mono src/ros_mono.cc)
target_link_libraries(Mono ${LIBS} )

add_executable(Stereo src/ros_stereo.cc)
target_link_libraries(Stereo ${LIBS})

add_executable(RGBD src/ros_rgbd.cc)
target_link_libraries(RGBD ${LIBS})

add_executable(Mono_Inertial src/ros_mono_inertial.cc)
target_link_libraries(Mono_Inertial ${LIBS})

add_executable(Stereo_Inertial src/ros_stereo_inertial.cc)
target_link_libraries(Stereo_Inertial ${LIBS})

