cmake_minimum_required( VERSION 2.8 )
project( MSCKF )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g" )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories(${OpenCV_LIBRARY_DIRS})

find_package(Eigen3)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

add_library( DataLoader src/dataLoader.cpp )
add_library( FeatureTracking src/featureTracking.cpp )
add_library( Visualization src/visualization.cpp )
add_library( MSCKF src/msckf.cpp )

add_executable( main src/main.cpp )

target_link_libraries( main DataLoader FeatureTracking Visualization MSCKF )
target_link_libraries( main ${OpenCV_LIBS} )


