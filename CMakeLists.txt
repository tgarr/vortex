cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(rag_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(derecho CONFIG REQUIRED)
find_package(cascade CONFIG REQUIRED)


add_library(centroids_search_udl SHARED centroids_search_udl.cpp)
target_include_directories(centroids_search_udl PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

target_link_libraries(centroids_search_udl derecho::cascade)
add_custom_command(TARGET centroids_search_udl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cfg
        ${CMAKE_CURRENT_BINARY_DIR}/cfg
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/setup
        ${CMAKE_CURRENT_BINARY_DIR}/setup
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
        ${CMAKE_CURRENT_BINARY_DIR}/
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/setup
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/setup
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/setup
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/setup
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/setup
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/setup
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/setup.py
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/setup.py
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/setup.py
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/setup.py
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/setup.py
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/setup.py
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/run.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/run.sh 
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/run.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/run.sh 
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/run.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/run.sh 
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/clear_log.sh 
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/clear_log.sh 
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/clear_log.sh 
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cfg/run.sh.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/clear_log.sh.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/dfgs.json.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/udl_dlls.cfg.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
    COMMENT "prepare centroids search configuration"
)
