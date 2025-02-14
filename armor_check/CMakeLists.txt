cmake_minimum_required(VERSION 3.8)
project(armor_check)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

file(GLOB_RECURSE display_srcs "src/display/*.cpp")
file(GLOB_RECURSE traditional_srcs "src/traditional/*.cpp")
file(GLOB_RECURSE yolo_srcs "src/yolo/*.cpp")

foreach(target IN ITEMS display traditional yolo)
    add_executable(${target} src/${target}/${target}.cpp)
    target_include_directories(${target} PRIVATE 
        include/armor_check
        ${OpenCV_INCLUDE_DIRS}
    )
    ament_target_dependencies(${target} 
        rclcpp 
        OpenCV 
        std_msgs
    )
    target_link_libraries(${target} ${OpenCV_LIBS})
endforeach()

install(DIRECTORY resource launch DESTINATION share/${PROJECT_NAME})
install(TARGETS display traditional yolo DESTINATION lib/${PROJECT_NAME})

ament_package()
