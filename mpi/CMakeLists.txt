find_package(MPI REQUIRED)

set(_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE _headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${_include_dir}/*.h*)

SET(project_names "contact")

foreach(project ${project_names})

    add_executable(${project}
        ${_headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/${project}.cpp
    )

    target_compile_features(${project} PRIVATE cxx_std_17)
    target_include_directories(${project} PRIVATE ${_include_dir})
    target_precompile_headers(${project} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/stdafx.h)
    unset(_include_dir)
    target_link_libraries(${project} PRIVATE MPI::MPI_CXX)

endforeach()

add_test(test_contacts "contact" "")