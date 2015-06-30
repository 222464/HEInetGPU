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

#include "EIlayer.h"

using namespace ei;

void EIlayer::Kernels::loadFromProgram(sys::ComputeProgram &program) {
	// Create kernels
	_eInitializeKernel = cl::Kernel(program.getProgram(), "EIlayer_eInitialize");
	_iInitializeKernel = cl::Kernel(program.getProgram(), "EIlayer_iInitialize");

	_eActivationKernel = cl::Kernel(program.getProgram(), "EIlayer_eActivate");
	_iActivationKernel = cl::Kernel(program.getProgram(), "EIlayer_iActivate");

	_eLearnKernel = cl::Kernel(program.getProgram(), "EIlayer_eLearn");
	_iLearnKernel = cl::Kernel(program.getProgram(), "EIlayer_iLearn");
}

void EIlayer::createRandom(const Configuration &config,
	float minInitEWeight, float maxInitEWeight,
	float minInitIWeight, float maxInitIWeight,
	float initEThreshold, float initIThreshold,
	float sparsityE, float sparsityI,
	sys::ComputeSystem &cs, const std::shared_ptr<Kernels> &eilKernels, std::mt19937 &generator)
{
	_kernels = eilKernels;

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

	_eLayer._shortAverages = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);
	_eLayer._shortAveragesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);

	_eLayer._longAverages = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);
	_eLayer._longAveragesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);

	_eLayer._thresholds = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);
	_eLayer._thresholdsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight);

	_iLayer._activations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);
	_iLayer._activationsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);

	_iLayer._states = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);
	_iLayer._statesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);

	_iLayer._shortAverages = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);
	_iLayer._shortAveragesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);

	_iLayer._longAverages = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);
	_iLayer._longAveragesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);

	_iLayer._thresholds = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);
	_iLayer._thresholdsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight);

	// Create images - weights
	_eFeedForwardWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight, eFeedForwardSize);
	_eFeedForwardWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight, eFeedForwardSize);

	_eFeedBackWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight, eFeedBackSize);
	_eFeedBackWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._eWidth, _config._eHeight, eFeedBackSize);

	_iFeedForwardWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iFeedForwardSize);
	_iFeedForwardWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iFeedForwardSize);

	_iFeedBackWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iFeedBackSize);
	_iFeedBackWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iFeedBackSize);

	_iLateralWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iLateralSize);
	_iLateralWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _config._iWidth, _config._iHeight, iLateralSize);

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl_float4 eLongAverageColor = { sparsityE, sparsityE, sparsityE, sparsityE };
	cl_float4 iLongAverageColor = { sparsityI, sparsityI, sparsityI, sparsityI };
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

	// Clear to defaults
	cs.getQueue().enqueueFillImage(_eLayer._activations, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._activationsPrev, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._states, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._statesPrev, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._shortAverages, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._shortAveragesPrev, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._longAverages, eLongAverageColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._longAveragesPrev, eLongAverageColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._thresholds, eThresholdColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._thresholdsPrev, eThresholdColor, zeroCoord, eDimsCoord);

	cs.getQueue().enqueueFillImage(_iLayer._activations, zeroColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._activationsPrev, zeroColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._states, zeroColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._statesPrev, zeroColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._shortAverages, zeroColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._shortAveragesPrev, zeroColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._longAverages, eLongAverageColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._longAveragesPrev, eLongAverageColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._thresholds, iThresholdColor, zeroCoord, iDimsCoord);
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
	_kernels->_eInitializeKernel.setArg(index++, minInitIWeight);
	_kernels->_eInitializeKernel.setArg(index++, maxInitIWeight);
	_kernels->_eInitializeKernel.setArg(index++, seedE);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_eInitializeKernel, cl::NullRange, cl::NDRange(_config._eWidth, _config._eHeight));

	cl::size_t<3> eFeedForwardWeightsDimsCoord;
	eFeedForwardWeightsDimsCoord[0] = _config._eWidth;
	eFeedForwardWeightsDimsCoord[1] = _config._eHeight;
	eFeedForwardWeightsDimsCoord[2] = eFeedForwardSize;

	cl::size_t<3> eFeedBackWeightsDimsCoord;
	eFeedBackWeightsDimsCoord[0] = _config._eWidth;
	eFeedBackWeightsDimsCoord[1] = _config._eHeight;
	eFeedBackWeightsDimsCoord[2] = eFeedBackSize;

	cs.getQueue().enqueueCopyImage(_eFeedForwardWeights._weightsPrev, _eFeedForwardWeights._weights, zeroCoord, zeroCoord, eFeedForwardWeightsDimsCoord);
	cs.getQueue().enqueueCopyImage(_eFeedBackWeights._weightsPrev, _eFeedBackWeights._weights, zeroCoord, zeroCoord, eFeedBackWeightsDimsCoord);

	index = 0;

	// Initialize weights
	_kernels->_iInitializeKernel.setArg(index++, _iFeedForwardWeights._weightsPrev);
	_kernels->_iInitializeKernel.setArg(index++, _iFeedBackWeights._weightsPrev);
	_kernels->_iInitializeKernel.setArg(index++, _iLateralWeights._weightsPrev);
	_kernels->_iInitializeKernel.setArg(index++, iFeedForwardSize);
	_kernels->_iInitializeKernel.setArg(index++, iLateralSize);
	_kernels->_iInitializeKernel.setArg(index++, iFeedBackSize);
	_kernels->_iInitializeKernel.setArg(index++, minInitEWeight);
	_kernels->_iInitializeKernel.setArg(index++, maxInitEWeight);
	_kernels->_iInitializeKernel.setArg(index++, minInitIWeight);
	_kernels->_iInitializeKernel.setArg(index++, maxInitIWeight);
	_kernels->_iInitializeKernel.setArg(index++, seedI);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_iInitializeKernel, cl::NullRange, cl::NDRange(_config._iWidth, _config._iHeight));

	cl::size_t<3> iFeedForwardWeightsDimsCoord;
	iFeedForwardWeightsDimsCoord[0] = _config._iWidth;
	iFeedForwardWeightsDimsCoord[1] = _config._iHeight;
	iFeedForwardWeightsDimsCoord[2] = iFeedForwardSize;

	cl::size_t<3> iFeedBackWeightsDimsCoord;
	iFeedBackWeightsDimsCoord[0] = _config._iWidth;
	iFeedBackWeightsDimsCoord[1] = _config._iHeight;
	iFeedBackWeightsDimsCoord[2] = iFeedBackSize;

	cl::size_t<3> iLateralWeightsDimsCoord;
	iLateralWeightsDimsCoord[0] = _config._iWidth;
	iLateralWeightsDimsCoord[1] = _config._iHeight;
	iLateralWeightsDimsCoord[2] = iLateralSize;

	cs.getQueue().enqueueCopyImage(_iFeedForwardWeights._weightsPrev, _iFeedForwardWeights._weights, zeroCoord, zeroCoord, iFeedForwardWeightsDimsCoord);
	cs.getQueue().enqueueCopyImage(_iFeedBackWeights._weightsPrev, _iFeedBackWeights._weights, zeroCoord, zeroCoord, iFeedBackWeightsDimsCoord);
	cs.getQueue().enqueueCopyImage(_iLateralWeights._weightsPrev, _iLateralWeights._weights, zeroCoord, zeroCoord, iLateralWeightsDimsCoord);
}

void EIlayer::eActivate(sys::ComputeSystem &cs, const cl::Image2D &feedForwardInput, float eta, float shortAverageSamplesInv, float longAverageDecay) {
	cl_int2 eFeedForwardDims = { _config._eFeedForwardWidth, _config._eFeedForwardHeight };
	cl_int2 eDims = { _config._eWidth, _config._eHeight };
	cl_int2 iDims = { _config._iWidth, _config._iHeight };
	cl_float2 eDimsToEFeedForwardDims = { static_cast<float>(eFeedForwardDims.x + 1) / static_cast<float>(eDims.x + 1), static_cast<float>(eFeedForwardDims.y + 1) / static_cast<float>(eDims.y + 1) };
	cl_float2 eDimsToIDims = { static_cast<float>(iDims.x + 1) / static_cast<float>(eDims.x + 1), static_cast<float>(iDims.y + 1) / static_cast<float>(eDims.y + 1) };

	int index = 0;

	_kernels->_eActivationKernel.setArg(index++, feedForwardInput);
	_kernels->_eActivationKernel.setArg(index++, _iLayer._statesPrev);
	_kernels->_eActivationKernel.setArg(index++, _eFeedForwardWeights._weightsPrev);
	_kernels->_eActivationKernel.setArg(index++, _eFeedBackWeights._weightsPrev);
	_kernels->_eActivationKernel.setArg(index++, _eLayer._thresholdsPrev);
	_kernels->_eActivationKernel.setArg(index++, _eLayer._activationsPrev);
	_kernels->_eActivationKernel.setArg(index++, _eLayer._statesPrev);
	_kernels->_eActivationKernel.setArg(index++, _eLayer._shortAveragesPrev);
	_kernels->_eActivationKernel.setArg(index++, _eLayer._longAveragesPrev);
	_kernels->_eActivationKernel.setArg(index++, _eLayer._activations);
	_kernels->_eActivationKernel.setArg(index++, _eLayer._states);
	_kernels->_eActivationKernel.setArg(index++, _eLayer._shortAverages);
	_kernels->_eActivationKernel.setArg(index++, _eLayer._longAverages);

	_kernels->_eActivationKernel.setArg(index++, eFeedForwardDims);
	_kernels->_eActivationKernel.setArg(index++, eDims);
	_kernels->_eActivationKernel.setArg(index++, iDims);
	_kernels->_eActivationKernel.setArg(index++, eDimsToEFeedForwardDims);
	_kernels->_eActivationKernel.setArg(index++, eDimsToIDims);
	_kernels->_eActivationKernel.setArg(index++, _config._eFeedForwardRadius);
	_kernels->_eActivationKernel.setArg(index++, _config._eFeedBackRadius);
	_kernels->_eActivationKernel.setArg(index++, eta);
	_kernels->_eActivationKernel.setArg(index++, shortAverageSamplesInv);
	_kernels->_eActivationKernel.setArg(index++, longAverageDecay);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_eActivationKernel, cl::NullRange, cl::NDRange(_config._eWidth, _config._eHeight));
}

void EIlayer::iActivate(sys::ComputeSystem &cs, const cl::Image2D &feedBackInput, float eta, float shortAverageSamplesInv, float longAverageDecay) {
	cl_int2 eDims = { _config._eWidth, _config._eHeight };
	cl_int2 iDims = { _config._iWidth, _config._iHeight };
	cl_int2 iFeedBackDims = { _config._iFeedBackWidth, _config._iFeedBackHeight };
	cl_float2 iDimsToEDims = { static_cast<float>(eDims.x + 1) / static_cast<float>(iDims.x + 1), static_cast<float>(eDims.y + 1) / static_cast<float>(iDims.y + 1) };
	cl_float2 iDimsToFeedBackDims = { static_cast<float>(iFeedBackDims.x + 1) / static_cast<float>(iDims.x + 1), static_cast<float>(iFeedBackDims.y + 1) / static_cast<float>(iDims.y + 1) };

	int index = 0;

	_kernels->_iActivationKernel.setArg(index++, _eLayer._statesPrev);
	_kernels->_iActivationKernel.setArg(index++, feedBackInput);
	_kernels->_iActivationKernel.setArg(index++, _iFeedForwardWeights._weightsPrev);
	_kernels->_iActivationKernel.setArg(index++, _iLateralWeights._weightsPrev);
	_kernels->_iActivationKernel.setArg(index++, _iFeedBackWeights._weightsPrev);
	_kernels->_iActivationKernel.setArg(index++, _iLayer._thresholdsPrev);
	_kernels->_iActivationKernel.setArg(index++, _iLayer._activationsPrev);
	_kernels->_iActivationKernel.setArg(index++, _iLayer._statesPrev);
	_kernels->_iActivationKernel.setArg(index++, _iLayer._shortAveragesPrev);
	_kernels->_iActivationKernel.setArg(index++, _iLayer._longAveragesPrev);
	_kernels->_iActivationKernel.setArg(index++, _iLayer._activations);
	_kernels->_iActivationKernel.setArg(index++, _iLayer._states);
	_kernels->_iActivationKernel.setArg(index++, _iLayer._shortAverages);
	_kernels->_iActivationKernel.setArg(index++, _iLayer._longAverages);

	_kernels->_iActivationKernel.setArg(index++, eDims);
	_kernels->_iActivationKernel.setArg(index++, iDims);
	_kernels->_iActivationKernel.setArg(index++, iFeedBackDims);
	_kernels->_iActivationKernel.setArg(index++, iDimsToEDims);
	_kernels->_iActivationKernel.setArg(index++, iDimsToFeedBackDims);
	_kernels->_iActivationKernel.setArg(index++, _config._iFeedForwardRadius);
	_kernels->_iActivationKernel.setArg(index++, _config._iLateralRadius);
	_kernels->_iActivationKernel.setArg(index++, _config._iFeedBackRadius);
	_kernels->_iActivationKernel.setArg(index++, eta);
	_kernels->_iActivationKernel.setArg(index++, shortAverageSamplesInv);
	_kernels->_iActivationKernel.setArg(index++, longAverageDecay);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_iActivationKernel, cl::NullRange, cl::NDRange(_config._iWidth, _config._iHeight));
}

void EIlayer::learn(sys::ComputeSystem &cs, const cl::Image2D &feedForwardShortAverages, const cl::Image2D &feedBackShortAverages,
	const cl::Image2D &feedBackLongAverages,
	float eAlpha, float eBeta, float eDelta,
	float iAlpha, float iBeta, float iGamma, float iDelta,
	float sparsityE, float sparsityI)
{
	// Common
	cl_int2 eDims = { _config._eWidth, _config._eHeight };
	cl_int2 iDims = { _config._iWidth, _config._iHeight };

	// Excitatory
	cl_int2 eFeedForwardDims = { _config._eFeedForwardWidth, _config._eFeedForwardHeight };
	cl_float2 eDimsToEFeedForwardDims = { static_cast<float>(eFeedForwardDims.x + 1) / static_cast<float>(eDims.x + 1), static_cast<float>(eFeedForwardDims.y + 1) / static_cast<float>(eDims.y + 1) };
	cl_float2 eDimsToIDims = { static_cast<float>(iDims.x + 1) / static_cast<float>(eDims.x + 1), static_cast<float>(iDims.y + 1) / static_cast<float>(eDims.y + 1) };

	// Inhibitory
	cl_int2 iFeedBackDims = { _config._iFeedBackWidth, _config._iFeedBackHeight };
	cl_float2 iDimsToEDims = { static_cast<float>(eDims.x + 1) / static_cast<float>(iDims.x + 1), static_cast<float>(eDims.y + 1) / static_cast<float>(iDims.y + 1) };
	cl_float2 iDimsToFeedBackDims = { static_cast<float>(iFeedBackDims.x + 1) / static_cast<float>(iDims.x + 1), static_cast<float>(iFeedBackDims.y + 1) / static_cast<float>(iDims.y + 1) };

	// Excitatory
	{
		int index = 0;

		_kernels->_eLearnKernel.setArg(index++, feedForwardShortAverages);
		_kernels->_eLearnKernel.setArg(index++, _eLayer._shortAveragesPrev);
		_kernels->_eLearnKernel.setArg(index++, _eLayer._longAveragesPrev);
		_kernels->_eLearnKernel.setArg(index++, _iLayer._shortAveragesPrev);
		_kernels->_eLearnKernel.setArg(index++, _iLayer._longAveragesPrev);
		_kernels->_eLearnKernel.setArg(index++, _eFeedForwardWeights._weightsPrev);
		_kernels->_eLearnKernel.setArg(index++, _eFeedBackWeights._weightsPrev);
		_kernels->_eLearnKernel.setArg(index++, _eLayer._thresholdsPrev);
		_kernels->_eLearnKernel.setArg(index++, _eFeedForwardWeights._weights);
		_kernels->_eLearnKernel.setArg(index++, _eFeedBackWeights._weights);
		_kernels->_eLearnKernel.setArg(index++, _eLayer._thresholds);

		_kernels->_eLearnKernel.setArg(index++, eFeedForwardDims);
		_kernels->_eLearnKernel.setArg(index++, eDims);
		_kernels->_eLearnKernel.setArg(index++, iDims);
		_kernels->_eLearnKernel.setArg(index++, eDimsToEFeedForwardDims);
		_kernels->_eLearnKernel.setArg(index++, eDimsToIDims);
		_kernels->_eLearnKernel.setArg(index++, _config._eFeedForwardRadius);
		_kernels->_eLearnKernel.setArg(index++, _config._eFeedBackRadius);

		_kernels->_eLearnKernel.setArg(index++, eAlpha);
		_kernels->_eLearnKernel.setArg(index++, eBeta);
		_kernels->_eLearnKernel.setArg(index++, eDelta);
		_kernels->_eLearnKernel.setArg(index++, sparsityE);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_eLearnKernel, cl::NullRange, cl::NDRange(_config._eWidth, _config._eHeight));
	}

	// Inhibitory
	{
		int index = 0;

		_kernels->_iLearnKernel.setArg(index++, feedBackShortAverages);
		_kernels->_iLearnKernel.setArg(index++, feedBackLongAverages);
		_kernels->_iLearnKernel.setArg(index++, _eLayer._shortAveragesPrev);
		_kernels->_iLearnKernel.setArg(index++, _eLayer._longAveragesPrev);
		_kernels->_iLearnKernel.setArg(index++, _iLayer._shortAveragesPrev);
		_kernels->_iLearnKernel.setArg(index++, _iLayer._longAveragesPrev);
		_kernels->_iLearnKernel.setArg(index++, _iFeedForwardWeights._weightsPrev);
		_kernels->_iLearnKernel.setArg(index++, _iLateralWeights._weightsPrev);
		_kernels->_iLearnKernel.setArg(index++, _iFeedBackWeights._weightsPrev);
		_kernels->_iLearnKernel.setArg(index++, _iLayer._thresholdsPrev);
		_kernels->_iLearnKernel.setArg(index++, _iFeedForwardWeights._weights);
		_kernels->_iLearnKernel.setArg(index++, _iLateralWeights._weights);
		_kernels->_iLearnKernel.setArg(index++, _iFeedBackWeights._weights);
		_kernels->_iLearnKernel.setArg(index++, _iLayer._thresholds);

		_kernels->_iLearnKernel.setArg(index++, eDims);
		_kernels->_iLearnKernel.setArg(index++, iDims);
		_kernels->_iLearnKernel.setArg(index++, iFeedBackDims);
		_kernels->_iLearnKernel.setArg(index++, iDimsToEDims);
		_kernels->_iLearnKernel.setArg(index++, iDimsToFeedBackDims);
		_kernels->_iLearnKernel.setArg(index++, _config._iFeedForwardRadius);
		_kernels->_iLearnKernel.setArg(index++, _config._iLateralRadius);
		_kernels->_iLearnKernel.setArg(index++, _config._iFeedBackRadius);

		_kernels->_iLearnKernel.setArg(index++, iAlpha);
		_kernels->_iLearnKernel.setArg(index++, iBeta);
		_kernels->_iLearnKernel.setArg(index++, iGamma);
		_kernels->_iLearnKernel.setArg(index++, iDelta);
		_kernels->_iLearnKernel.setArg(index++, sparsityI);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_iLearnKernel, cl::NullRange, cl::NDRange(_config._iWidth, _config._iHeight));
	}
}

void EIlayer::exStepBegin(sys::ComputeSystem &cs) {
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

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

	// Clear short run average buffers
	cs.getQueue().enqueueFillImage(_eLayer._shortAverages, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_eLayer._shortAveragesPrev, zeroColor, zeroCoord, eDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._shortAverages, zeroColor, zeroCoord, iDimsCoord);
	cs.getQueue().enqueueFillImage(_iLayer._shortAveragesPrev, zeroColor, zeroCoord, iDimsCoord);
}

void EIlayer::simStepEnd() {
	// Swap buffers
	std::swap(_eLayer._activations, _eLayer._activationsPrev);
	std::swap(_eLayer._states, _eLayer._statesPrev);
	std::swap(_eLayer._shortAverages, _eLayer._shortAveragesPrev);
	std::swap(_eLayer._longAverages, _eLayer._longAveragesPrev);

	std::swap(_iLayer._activations, _iLayer._activationsPrev);
	std::swap(_iLayer._states, _iLayer._statesPrev);
	std::swap(_iLayer._shortAverages, _iLayer._shortAveragesPrev);
	std::swap(_iLayer._longAverages, _iLayer._longAveragesPrev);
}

void EIlayer::exStepEnd() {
	std::swap(_eFeedForwardWeights._weights, _eFeedForwardWeights._weightsPrev);
	std::swap(_eFeedBackWeights._weights, _eFeedBackWeights._weightsPrev);
	std::swap(_eLayer._thresholds, _eLayer._thresholdsPrev);

	std::swap(_iFeedForwardWeights._weights, _iFeedForwardWeights._weightsPrev);
	std::swap(_iLateralWeights._weights, _iLateralWeights._weightsPrev);
	std::swap(_iFeedBackWeights._weights, _iFeedBackWeights._weightsPrev);
	std::swap(_iLayer._thresholds, _iLayer._thresholdsPrev);
}