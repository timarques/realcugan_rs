cmake_minimum_required(VERSION 3.20)
project(realcugan-wrapper)

find_package(Threads)
find_package(OpenMP)
find_package(Vulkan REQUIRED)
find_package(ncnn REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 11)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO is not supported")
endif()

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
    set(SHADER_COMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_SRC_NAME_WE}.comp.hex.h)
    set(SHADER_TEXT2HEX ${CMAKE_BINARY_DIR}/shaders/${SHADER_SRC_NAME_WE}.text2hex.txt)

    file(READ ${CMAKE_SOURCE_DIR}/shaders/${SHADER} comp_data)
    string(FIND "${comp_data}" "#version" version_start)
    string(SUBSTRING "${comp_data}" ${version_start} -1 comp_data)
    string(REGEX REPLACE "\n +" "\n" comp_data "${comp_data}")

    file(WRITE ${SHADER_TEXT2HEX} "${comp_data}")
    file(READ ${SHADER_TEXT2HEX} comp_data_hex HEX)

    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," comp_data_hex ${comp_data_hex})
    string(FIND "${comp_data_hex}" "," tail_comma REVERSE)
    string(SUBSTRING "${comp_data_hex}" 0 ${tail_comma} comp_data_hex)

    file(WRITE ${SHADER_COMP_HEADER} "static const char ${SHADER_SRC_NAME_WE}_comp_data[] = {${comp_data_hex}};\n")
    set_source_files_properties(${SHADER_COMP_HEADER} PROPERTIES GENERATED TRUE)
    list(APPEND SHADERS_HEX_FILES ${SHADER_COMP_HEADER})
endforeach()

add_custom_target(generate-spirv DEPENDS ${SHADERS_HEX_FILES})

include_directories(${CMAKE_SOURCE_DIR}/cpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/shaders)

set(REALCUGAN_LIBS ncnn ${Vulkan_LIBRARY})
if(OPENMP_FOUND)
    list(APPEND REALCUGAN_LIBS ${OpenMP_CXX_LIBRARIES})
endif()

add_library(realcugan-wrapper STATIC ${CMAKE_SOURCE_DIR}/cpp/wrapper.cpp ${CMAKE_SOURCE_DIR}/cpp/realcugan.cpp)
add_dependencies(realcugan-wrapper generate-spirv)
target_link_libraries(realcugan-wrapper ${REALCUGAN_LIBS})
install(TARGETS realcugan-wrapper ARCHIVE DESTINATION lib)
