# Find the STKFMM Library
#
#  STKFMM_FOUND - System has MKL
#  STKFMM_INCLUDE_DIRS - MKL include files directories
#  STKFMM_LIBRARIES - The MKL libraries
#  STKFMM_INTERFACE_LIBRARY - MKL interface library
#  STKFMM_SEQUENTIAL_LAYER_LIBRARY - MKL sequential layer library
#  STKFMM_CORE_LIBRARY - MKL core library
#
#  Example usage:
#
#  find_package(STKFMM)
#  if(STKFMM_FOUND)
#    target_link_libraries(TARGET ${STKFMM_LIBRARIES})
#  endif()

# If already in cache, be silent
if (STKFMM_INCLUDE_DIRS AND STKFMM_LIBRARIES)
  set (STKFMM_FIND_QUIETLY TRUE)
endif()

find_path(STKFMM_INCLUDE_DIR NAMES STKFMM/STKFMM.h HINTS $ENV{STKFMM_ROOT}/include $ENV{STKFMM_BASE}/include)

find_library(STKFMM_LIBRARY
             NAMES STKFMM_STATIC
             PATHS $ENV{STKFMM_ROOT}/lib64
                   $ENV{STKFMM_ROOT}/lib
                   $ENV{STKFMM_BASE}/lib64
                   $ENV{STKFMM_BASE}/lib
             NO_DEFAULT_PATH)

set(STKFMM_INCLUDE_DIRS ${STKFMM_INCLUDE_DIR})
set(STKFMM_LIBRARIES ${STKFMM_LIBRARY})

# Handle the QUIETLY and REQUIRED arguments and set STKFMM_FOUND to TRUE if
# all listed variables are TRUE.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(STKFMM DEFAULT_MSG STKFMM_LIBRARIES STKFMM_INCLUDE_DIRS)
