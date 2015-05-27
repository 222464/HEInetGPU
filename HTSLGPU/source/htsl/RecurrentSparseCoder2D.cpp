/*
HTSLGPU
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

#include "RecurrentSparseCoder2D.h"

using namespace htsl;

void RecurrentSparseCoder2D::Kernels::loadFromProgram(sys::ComputeProgram &program) {
	// Create kernels
	_initializeKernel = cl::Kernel(program.getProgram(), "rscInitialize");
	_excitationKernel = cl::Kernel(program.getProgram(), "rscExcitation");
	_activateKernel = cl::Kernel(program.getProgram(), "rscActivate");
	_learnKernel = cl::Kernel(program.getProgram(), "rscLearn");
}

void RecurrentSparseCoder2D::createRandom(int inputWidth, int inputHeight, int width, int height,
	int receptiveRadius, int recurrentRadius, int inhibitionRadius, float ffWeight, float lWeight, float initBias,
	sys::ComputeSystem &cs, const std::shared_ptr<Kernels> &kernels, std::mt19937 &generator)
{
	_kernels = kernels;

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;
	_width = width;
	_height = height;

	_receptiveRadius = receptiveRadius;
	_recurrentRadius = recurrentRadius;
	_inhibitionRadius = inhibitionRadius;

	// Total size (number of weights) in receptive fields
	int receptiveSize = std::pow(_receptiveRadius * 2 + 1, 2);
	int recurrentSize = std::pow(_recurrentRadius * 2 + 1, 2);
	int inhibitionSize = std::pow(_inhibitionRadius * 2 + 1, 2);

	// Create images
	_excitations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	
	_activations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_activationsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	_spikes = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_spikesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_spikesRecurrentPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	_states = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_statesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	
	_hiddenVisibleWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, receptiveSize);
	_hiddenVisibleWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, receptiveSize);

	_hiddenHiddenPrevWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, recurrentSize);
	_hiddenHiddenPrevWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, recurrentSize);

	_hiddenHiddenWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, inhibitionSize);
	_hiddenHiddenWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height, inhibitionSize);
	
	_biases = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);
	_biasesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _width, _height);

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl_float4 biasColor = { initBias, initBias, initBias, initBias };

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> dimsCoord;
	dimsCoord[0] = _width;
	dimsCoord[1] = _height;
	dimsCoord[2] = 1;

	// Clear to defaults (only prev buffers, since other ones will be written to immediately)
	cs.getQueue().enqueueFillImage(_activationsPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_spikesPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_spikesRecurrentPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_statesPrev, zeroColor, zeroCoord, dimsCoord);

	cs.getQueue().enqueueFillImage(_biasesPrev, biasColor, zeroCoord, dimsCoord);

	int index = 0;

	std::uniform_int_distribution<int> seedDist(0, 10000);

	// Weight RNG seed
	cl_uint2 seed = { seedDist(generator), seedDist(generator) };

	// Initialize weights
	_kernels->_initializeKernel.setArg(index++, _hiddenVisibleWeightsPrev);
	_kernels->_initializeKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
	_kernels->_initializeKernel.setArg(index++, _hiddenHiddenWeightsPrev);
	_kernels->_initializeKernel.setArg(index++, receptiveSize);
	_kernels->_initializeKernel.setArg(index++, recurrentSize);
	_kernels->_initializeKernel.setArg(index++, inhibitionSize);
	_kernels->_initializeKernel.setArg(index++, ffWeight);
	_kernels->_initializeKernel.setArg(index++, lWeight);
	_kernels->_initializeKernel.setArg(index++, seed);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_initializeKernel, cl::NullRange, cl::NDRange(_width, _height));	
}

void RecurrentSparseCoder2D::update(sys::ComputeSystem &cs, const cl::Image2D &inputs, float dt, int iterations) {
	cl_int2 inputDims = { _inputWidth, _inputHeight };
	cl_int2 dims = { _width, _height };
	cl_float2 dimsToInputDims = { static_cast<float>(_inputWidth + 1) / static_cast<float>(_width + 1), static_cast<float>(_inputHeight + 1) / static_cast<float>(_height + 1) };

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> dimsCoord;
	dimsCoord[0] = _width;
	dimsCoord[1] = _height;
	dimsCoord[2] = 1;

	// Clear images
	cs.getQueue().enqueueFillImage(_activationsPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_statesPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_spikesPrev, zeroColor, zeroCoord, dimsCoord);

	// Used to normalize the spikes for recurrent input
	float spikeNorm = 1.0f / iterations;

	// Excite
	{
		int index = 0;

		_kernels->_excitationKernel.setArg(index++, inputs);
		_kernels->_excitationKernel.setArg(index++, _spikesRecurrentPrev);
		_kernels->_excitationKernel.setArg(index++, _hiddenVisibleWeightsPrev);
		_kernels->_excitationKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
		_kernels->_excitationKernel.setArg(index++, _excitations);
		_kernels->_excitationKernel.setArg(index++, inputDims);
		_kernels->_excitationKernel.setArg(index++, dims);
		_kernels->_excitationKernel.setArg(index++, dimsToInputDims);
		_kernels->_excitationKernel.setArg(index++, _receptiveRadius);
		_kernels->_excitationKernel.setArg(index++, _recurrentRadius);
		_kernels->_excitationKernel.setArg(index++, spikeNorm);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_excitationKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	// Used for falloff calculation
	float inhibitionRadiusInv = 1.0f / _inhibitionRadius;

	// Iterative solving
	for (int i = 0; i < iterations; i++) {
		// Activate
		{
			int index = 0;

			_kernels->_activateKernel.setArg(index++, _excitations);
			_kernels->_activateKernel.setArg(index++, _statesPrev);
			_kernels->_activateKernel.setArg(index++, _activationsPrev);
			_kernels->_activateKernel.setArg(index++, _spikesPrev);
			_kernels->_activateKernel.setArg(index++, _hiddenHiddenWeightsPrev);
			_kernels->_activateKernel.setArg(index++, _biasesPrev);
			_kernels->_activateKernel.setArg(index++, _activations);
			_kernels->_activateKernel.setArg(index++, _spikes);
			_kernels->_activateKernel.setArg(index++, _states);
			_kernels->_activateKernel.setArg(index++, dims);
			_kernels->_activateKernel.setArg(index++, _inhibitionRadius);
			_kernels->_activateKernel.setArg(index++, inhibitionRadiusInv);
			_kernels->_activateKernel.setArg(index++, dt);

			cs.getQueue().enqueueNDRangeKernel(_kernels->_activateKernel, cl::NullRange, cl::NDRange(_width, _height));
		}

		// Don't swap on last iteration, since there is no subsequent one
		if (i != iterations - 1) {
			std::swap(_activations, _activationsPrev);
			std::swap(_states, _statesPrev);
			std::swap(_spikes, _spikesPrev);
		}
	}
}

void RecurrentSparseCoder2D::learn(sys::ComputeSystem &cs, const cl::Image2D &inputs, float alpha, float beta, float gamma, float delta, float sparsity, int iterations) {
	cl_int2 inputDims = { _inputWidth, _inputHeight };
	cl_int2 dims = { _width, _height };
	cl_float2 dimsToInputDims = { static_cast<float>(_inputWidth + 1) / static_cast<float>(_width + 1), static_cast<float>(_inputHeight + 1) / static_cast<float>(_height + 1) };
	cl_float4 learningRates = { alpha, beta, gamma, delta };

	float spikeNorm = 1.0f / iterations;

	// Learn
	{
		int index = 0;

		_kernels->_learnKernel.setArg(index++, inputs);
		_kernels->_learnKernel.setArg(index++, _spikes);
		_kernels->_learnKernel.setArg(index++, _spikesRecurrentPrev);
		_kernels->_learnKernel.setArg(index++, _hiddenVisibleWeightsPrev);
		_kernels->_learnKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
		_kernels->_learnKernel.setArg(index++, _hiddenHiddenWeightsPrev);
		_kernels->_learnKernel.setArg(index++, _biasesPrev);
		_kernels->_learnKernel.setArg(index++, _hiddenVisibleWeights);
		_kernels->_learnKernel.setArg(index++, _hiddenHiddenPrevWeights);
		_kernels->_learnKernel.setArg(index++, _hiddenHiddenWeights);
		_kernels->_learnKernel.setArg(index++, _biases);
		_kernels->_learnKernel.setArg(index++, inputDims);
		_kernels->_learnKernel.setArg(index++, dims);
		_kernels->_learnKernel.setArg(index++, dimsToInputDims);
		_kernels->_learnKernel.setArg(index++, _receptiveRadius);
		_kernels->_learnKernel.setArg(index++, _recurrentRadius);
		_kernels->_learnKernel.setArg(index++, _inhibitionRadius);
		_kernels->_learnKernel.setArg(index++, spikeNorm);
		_kernels->_learnKernel.setArg(index++, learningRates);
		_kernels->_learnKernel.setArg(index++, sparsity);
		_kernels->_learnKernel.setArg(index++, sparsity * sparsity);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_learnKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	// Swap buffers
	std::swap(_hiddenVisibleWeights, _hiddenVisibleWeightsPrev);
	std::swap(_hiddenHiddenPrevWeights, _hiddenHiddenPrevWeightsPrev);
	std::swap(_hiddenHiddenWeights, _hiddenHiddenWeightsPrev);
	std::swap(_biases, _biasesPrev);
}

void RecurrentSparseCoder2D::stepEnd() {
	// Swap buffers
	std::swap(_spikes, _spikesRecurrentPrev);
}