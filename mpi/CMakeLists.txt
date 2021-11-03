#find_package(Boost REQUIRED)

set(_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE _headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${_include_dir}/*.h*)

add_executable(${PROJECT_NAME} ${_headers} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE ${_include_dir})
unset(_include_dir)
target_link_libraries(${PROJECT_NAME})
