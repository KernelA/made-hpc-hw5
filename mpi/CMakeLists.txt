find_package(MPI REQUIRED)
find_package(ICU REQUIRED COMPONNETS i18n uc io)

set(_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE _headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${_include_dir}/*.h*)

SET(project_names "contact;cellular_automat")

foreach(project ${project_names})

    add_executable(${project}
        ${_headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/${project}.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    )

    target_compile_features(${project} PRIVATE cxx_std_17)
    target_include_directories(${project} PRIVATE ${_include_dir})
    target_precompile_headers(${project} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/stdafx.h)
    target_link_libraries(${project} PRIVATE MPI::MPI_CXX)

endforeach()



add_executable("word_counter"
    ${_headers}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/word_counter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
)

target_compile_features("word_counter" PRIVATE cxx_std_17)
target_include_directories("word_counter" PRIVATE ${_include_dir})
target_precompile_headers("word_counter" PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/stdafx.h)
target_link_libraries("word_counter" PRIVATE MPI::MPI_CXX PRIVATE ICU::i18n PRIVATE ICU::uc PRIVATE ICU::io)

SET(test_projects "test_bin2num;test_num2bit")

foreach(test_project ${test_projects})

    add_executable(${test_project}
        ${_headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/${test_project}.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    )

    target_compile_features(${test_project} PRIVATE cxx_std_17)
    target_include_directories(${test_project} PRIVATE ${_include_dir})
    target_precompile_headers(${test_project} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/stdafx.h)
endforeach()


SET(right_answer "1;2;0;4;8;3")
SET(bin_rep "00000001;00000010;00000000;00000100;00001000;00000011")

list(LENGTH right_answer total_tests)

math(EXPR total_tests "${total_tests} - 1")

foreach(test_num RANGE ${total_tests})
    list(GET right_answer ${test_num} answer)
    list(GET bin_rep ${test_num} bit_mask)
    message(STATUS "Add test ${answer} ${bit_mask}")
    add_test(test_bin2num_${test_num} "test_bin2num" "${answer}" "${bit_mask}")
endforeach()

foreach(test_num RANGE ${total_tests})
    list(GET right_answer ${test_num} answer)
    list(GET bin_rep ${test_num} bit_mask)
    message(STATUS "Add test ${answer} ${bit_mask}")
    add_test(test_num2bit_${test_num} "test_num2bit" "${answer}" "${bit_mask}")
endforeach()

