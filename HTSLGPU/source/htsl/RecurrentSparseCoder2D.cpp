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
	_eInitializeKernel = cl::Kernel(program.getProgram(), "rsc_eInitialize");
	_iInitializeKernel = cl::Kernel(program.getProgram(), "rsc_iInitialize");

	_eActivationKernel = cl::Kernel(program.getProgram(), "rsc_eActivation");
	_iActivationKernel = cl::Kernel(program.getProgram(), "rsc_iActivation");

	_eLearnKernel = cl::Kernel(program.getProgram(), "rsc_eLearn");
	_iLearnKernel = cl::Kernel(program.getProgram(), "rsc_iLearn");
}

void RecurrentSparseCoder2D::createRandom(const Configuration &config,
	float minInitEWeight, float maxInitEWeight,
	float minInitIWeight, float maxInitIWeight,
	float initEThreshold, float initIThreshold,
	sys::ComputeSystem &cs, const std::shared_ptr<Kernels> &kernels, std::mt19937 &generator)
{
	_kernels = kernels;

	_config = config;

	// Total size (number of weights) in receptive fields
	int eFeedForwardSize = std::pow(_config._eFeedForwardRadius * 2 + 1, 2);
	int eFeedBackSize = std::pow(_config._eFeedBackRadius * 2 + 1, 2);
	int iFeedForwardSize = std::pow(_config._iFeedForwardRadius * 2 + 1, 2);
	int iLateralSize = std::pow(_config._iLateralRadius * 2 + 1, 2);
	int iFeedBackSize = std::pow(_config._iFeedBackRadius * 2 + 1, 2);

	// Create images - neurons
	_eLayer._activations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);
	_eLayer._activationsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);

	_eLayer._states = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);
	_eLayer._statesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);

	_eLayer._thresholds = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);
	_eLayer._thresholdsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);

	_iLayer._activations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);
	_iLayer._activationsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);

	_iLayer._states = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);
	_iLayer._statesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);

	_iLayer._thresholds = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);
	_iLayer._thresholdsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);

	// Create images - weights
	_eFeedForwardWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight, eFeedForwardSize);
	_eFeedForwardWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight, eFeedForwardSize);

	_eFeedBackWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight, eFeedBackSize);
	_eFeedBackWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight, eFeedBackSize);

	_iFeedForwardWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iFeedForwardSize);
	_iFeedForwardWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iFeedForwardSize);

	_iLateralWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iLateralSize);
	_iLateralWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iLateralSize);

	_iFeedBackWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iFeedBackSize);
	_iFeedBackWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iFeedBackSize);

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl_float4 eThresholdColor = { initEThreshold, initEThreshold, initEThreshold, initEThreshold };
	cl_float4 iThresholdColor = { initIThreshold, initIThreshold, initIThreshold, initIThreshold };

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> eDimsCoord;
	eDimsCoord[0] = _config._eWidth;
	eDimsCoord[1] = _config._eHeight;
	eDimsCoord[2] = 1;

	cl::size_t<3> iDimsCoord;
	iDimsCoord[0] = _config._iWidth;
	iDimsCoord[1] = _config._iHeight;
	iDimsCoord[2] = 1;

	// Clear to defaults (only prev buffers, since other ones will be written to immediately)
	cs.getQueue().enqueueFillImage(_eLayer._activationsPrev, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._statesPrev, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._thresholdsPrev, eThresholdColor, zeroCoord, eDimsCoord);

	cs.getQueue().enqueueFillImage(_iLayer._activationsPrev, zeroColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._statesPrev, zeroColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._thresholdsPrev, iThresholdColor, zeroCoord, iDimsCoord);

	int index = 0;

	std::uniform_int_distribution<int> seedDist(0, 10000);

	// Weight RNG seed
	cl_uint2 seedE = { seedDist(generator), seedDist(generator) };
	cl_uint2 seedI = { seedDist(generator), seedDist(generator) };

	// Initialize weights
	_kernels->_eInitializeKernel.setArg(index++, _eFeedForwardWeights._weightsPrev);
	_kernels->_eInitializeKernel.setArg(index++, _eFeedBackWeights._weightsPrev);
	_kernels->_eInitializeKernel.setArg(index++, eFeedForwardSize);
	_kernels->_eInitializeKernel.setArg(index++, eFeedBackSize);
	_kernels->_eInitializeKernel.setArg(index++, minInitEWeight);
	_kernels->_eInitializeKernel.setArg(index++, maxInitEWeight);
	_kernels->_eInitializeKernel.setArg(index++, seedE);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_eInitializeKernel, cl::NullRange, cl::NDRange(_config._eWidth, _config._eHeight));

	// Initialize weights
	_kernels->_iInitializeKernel.setArg(index++, _iFeedForwardWeights._weightsPrev);
	_kernels->_iInitializeKernel.setArg(index++, _iLateralWeights._weightsPrev);
	_kernels->_iInitializeKernel.setArg(index++, _iFeedBackWeights._weightsPrev);
	_kernels->_iInitializeKernel.setArg(index++, iFeedForwardSize);
	_kernels->_iInitializeKernel.setArg(index++, iLateralSize);
	_kernels->_iInitializeKernel.setArg(index++, iFeedBackSize);
	_kernels->_iInitializeKernel.setArg(index++, minInitIWeight);
	_kernels->_iInitializeKernel.setArg(index++, maxInitIWeight);
	_kernels->_iInitializeKernel.setArg(index++, seedI);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_iInitializeKernel, cl::NullRange, cl::NDRange(_config._iWidth, _config._iHeight));
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

	// Used for falloff calculation
	float inhibitionRadiusInv = 1.0f / _inhibitionRadius;

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
		_kernels->_learnKernel.setArg(index++, inhibitionRadiusInv);
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
	std::swap(_eLayer._activations, _eLayer._activationsPrev);
	std::swap(_eLayer._states, _eLayer._statesPrev);
	std::swap(_eLayer._thresholds, _eLayer._thresholdsPrev);

	std::swap(_iLayer._activations, _iLayer._activationsPrev);
	std::swap(_iLayer._states, _iLayer._statesPrev);
	std::swap(_iLayer._thresholds, _iLayer._thresholdsPrev);

	std::swap(_eFeedForwardWeights._weights, _eFeedForwardWeights._weightsPrev);
	std::swap(_eFeedBackWeights._weights, _eFeedBackWeights._weightsPrev);

	std::swap(_iFeedForwardWeights._weights, _iFeedForwardWeights._weightsPrev);
	std::swap(_iLateralWeights._weights, _iLateralWeights._weightsPrev);
	std::swap(_iFeedBackWeights._weights, _iFeedBackWeights._weightsPrev);
}