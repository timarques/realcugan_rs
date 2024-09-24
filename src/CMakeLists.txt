cmake_minimum_required(VERSION 3.9)
project(realcugan-wrapper)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "Clang++")
endif()

###OPENMP
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Threads)
find_package(OpenMP)
find_package(Vulkan REQUIRED)
find_package(ncnn REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/cpp)
include_directories(${CMAKE_SOURCE_DIR}/shaders)

set(SHADERS_FILES
    realcugan_preproc.comp
    realcugan_postproc.comp
    realcugan_4x_postproc.comp
    realcugan_preproc_tta.comp
    realcugan_postproc_tta.comp
    realcugan_4x_postproc_tta.comp
)

foreach(SHADER ${SHADERS_FILES})
    get_filename_component(SHADER_SRC_NAME_WE ${SHADER} NAME_WE)

    set(SHADER_SRC ${CMAKE_SOURCE_DIR}/shaders/${SHADER})
    file(READ ${SHADER_SRC} comp_data)
    string(FIND "${comp_data}" "#version" version_start)
    string(SUBSTRING "${comp_data}" ${version_start} -1 comp_data)
    string(REGEX REPLACE "\n +" "\n" comp_data "${comp_data}")

    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)

    set(SHARE_TEXT2HEX ${CMAKE_SOURCE_DIR}/shaders/${SHADER_SRC_NAME_WE}.text2hex.txt)
    file(WRITE ${SHARE_TEXT2HEX} "${comp_data}")
    file(READ ${SHARE_TEXT2HEX} comp_data_hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," comp_data_hex ${comp_data_hex})
    string(FIND "${comp_data_hex}" "," tail_comma REVERSE)
    string(SUBSTRING "${comp_data_hex}" 0 ${tail_comma} comp_data_hex)

    set(SHADER_HEADER ${CMAKE_SOURCE_DIR}/shaders/${SHADER_SRC_NAME_WE}.hex.h)
    file(WRITE ${SHADER_HEADER} "static const char ${SHADER_SRC_NAME_WE}_comp_data[] = {${comp_data_hex}};\n")
    set_source_files_properties(${SHADER_HEADER} PROPERTIES GENERATED TRUE)
    list(APPEND SHADERS_HEX_FILES ${SHADER_HEADER})
endforeach()

add_custom_target(generate-spirv DEPENDS ${SHADERS_HEX_FILES})

###

add_library(
    realcugan-wrapper STATIC
    ${CMAKE_SOURCE_DIR}/cpp/wrapper.cpp
    ${CMAKE_SOURCE_DIR}/cpp/realcugan.cpp
)
add_dependencies(realcugan-wrapper generate-spirv)
target_link_libraries(realcugan-wrapper PRIVATE ncnn ${Vulkan_LIBRARY})
install(TARGETS realcugan-wrapper ARCHIVE DESTINATION lib)