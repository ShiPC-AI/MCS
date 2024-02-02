add_executable(mcs_bf_recall_1 run_mcs_bf_recall_top1.cpp)
target_link_libraries(mcs_bf_recall_1 ${MCS_BASE_LIB} 
    ${MCS_DESCRIPTOR_LIB} ${PCL_LIBRARIES})

add_executable(mcs_recall_1 run_mcs_recall_top1.cpp)
target_link_libraries(mcs_recall_1 ${MCS_BASE_LIB} 
    ${MCS_DESCRIPTOR_LIB} ${PCL_LIBRARIES})
