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

#pragma once

#include <system/Uncopyable.h>
#include <CL/cl.hpp>

#define SYS_DEBUG

#define SYS_ALLOW_CL_GL_CONTEXT 0

namespace sys {
	class ComputeSystem : private Uncopyable {
	public:
		enum DeviceType {
			_cpu, _gpu, _all, _none
		};

	private:
		cl::Platform _platform;
		cl::Device _device;
		cl::Context _context;
		cl::CommandQueue _queue;

	public:
		bool create(DeviceType type, bool createFromGLContext = false);

		cl::Platform &getPlatform() {
			return _platform;
		}

		cl::Device &getDevice() {
			return _device;
		}

		cl::Context &getContext() {
			return _context;
		}

		cl::CommandQueue &getQueue() {
			return _queue;
		}
	};
}