cmake_minimum_required(VERSION 3.8)
project(crazyflie_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(crazyflie_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

ament_python_install_package(${PROJECT_NAME}_py)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/cf_interface.py
  scripts/lqr_controller.py
  scripts/deepreach_full_state_controller.py
  scripts/full_state_controller.py
  scripts/pursuit_evasion_controller.py
  scripts/20d_drone_controller.py
  scripts/20d_drone_controller_ghost.py
  scripts/deepreach_controller_test.py
  scripts/6d_test.py
  scripts/calibration_controller.py
  scripts/realmcf_tracker_controller.py
  scripts/input_lemniscate.py
  DESTINATION lib/${PROJECT_NAME})

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CFTraj.msg"
  "msg/CFTrajArray.msg"
  "msg/StateStamped.msg"
  "srv/Command.srv"
  DEPENDENCIES builtin_interfaces
  )

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_add_pytest_test(test_traj_manager test/test_traj_manager.py)
  ament_add_pytest_test(test_tracker_utils test/test_tracker_utils.py)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
