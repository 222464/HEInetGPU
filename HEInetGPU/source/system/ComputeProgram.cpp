/*
HEInetGPU
Copyright (C) 2015 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "ComputeProgram.h"

#include <fstream>
#include <iostream>

using namespace sys;

bool ComputeProgram::loadFromFile(const std::string &name, ComputeSystem &cs) {
	std::ifstream fromFile(name);

	if (!fromFile.is_open()) {
#ifdef SYS_DEBUG
		std::cerr << "Could not open file " << name << "!" << std::endl;
#endif
		return false;
	}

	std::string source = "";

	while (!fromFile.eof() && fromFile.good()) {
		std::string line; 

		std::getline(fromFile, line);

		source += line + "\n";
	}

	_program = cl::Program(cs.getContext(), source);

	if (_program.build(std::vector<cl::Device>(1, cs.getDevice())) != CL_SUCCESS) {
#ifdef SYS_DEBUG
		std::cerr << "Error building: " << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice()) << std::endl;
#endif
		return false;
	}

	return true;
}