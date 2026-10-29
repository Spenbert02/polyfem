# hocgv-miso: miso-generated Jacobian validity solvers
# License: see hocgv-miso repo

if(TARGET miso)
    return()
endif()

message(STATUS "Third-party: creating target 'miso' via hocgv-miso")

include(CPM)
CPMAddPackage("gh:fsichetti/hocgv-miso#ddd0b3469efc2c0f26abd38ec4cb072801620d13")

# miso's generated P3 sources are single functions of up to ~1 MB. With debug info
# (RelWithDebInfo), AppleClang 21's RemoveRedundantDebugValues pass ran for over an
# hour per file without finishing; without it, P3TetCGV/CL_m0.cpp compiles in ~90 s.
if(NOT MSVC)
    target_compile_options(miso PRIVATE -g0)
endif()
