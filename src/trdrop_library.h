#pragma once

#include <functional>

//! The command line program as a function, for programs that embed the analysis (the Android app): built
//! with TRDROP_LIBRARY defined, main.cpp has this instead of main().
//!
//! argc / argv: as for the command line program (argv[0] is the program name), the messages go to stderr.
//! progress: called while the videos are analyzed with the frames read and the progress (0 - 1, < 0 if
//! unknown); if it returns false the analysis stops with the error "cancelled". May be empty.
//! Returns the exit code of the program: 0 success, 1 error, 2 invalid arguments.
int trdrop_run(int argc, char ** argv, const std::function<bool(long long, double)> & progress);
