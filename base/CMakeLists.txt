SET(LIB_NAME mcs_base)
SET(LIB_NAME_UPPER MCS_BASE)

aux_source_directory(./ ${LIB_NAME_UPPER}_SRC)

SET(${LIB_NAME_UPPER}_LIB ${LIB_NAME} CACHE INTERNAL "lib")

ADD_LIBRARY(${${LIB_NAME_UPPER}_LIB} SHARED 
            ${${LIB_NAME_UPPER}_SRC})

TARGET_LINK_LIBRARIES(${${LIB_NAME_UPPER}_LIB} 
                       ${PCL_LIBRARIES})