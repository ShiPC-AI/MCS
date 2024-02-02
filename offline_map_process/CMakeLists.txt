add_subdirectory(cluster)

# make ocscs for all virtual pts               
add_executable(create_map_cluster_centers create_map_cluster_centers.cpp)
target_link_libraries(create_map_cluster_centers ${MCS_GROUND_LIB}
                        ${MCS_BASE_LIB} ${MCS_DESCRIPTOR_LIB} ${MCS_CLUSTER_LIB})

# create virtual pts and map point cloud       
add_executable(create_virtual_pts_and_map create_virtual_pts_and_map.cpp)
target_link_libraries(create_virtual_pts_and_map ${MCS_GROUND_LIB} ${MCS_BASE_LIB})