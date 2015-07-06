#include "HEInet.h"

using namespace ei;

void HEInet::Kernels::loadFromProgram(sys::ComputeProgram &program) {
	// Create kernels
	_predictionInitializeKernel = cl::Kernel(program.getProgram(), "HEInet_predictionInitialize");

	_predictKernel = cl::Kernel(program.getProgram(), "HEInet_predict");

	_predictionLearnKernel = cl::Kernel(program.getProgram(), "HEInet_predictionLearn");

	_updateInputSpikesKernel = cl::Kernel(program.getProgram(), "HEInet_updateInputSpikes");

	_sumSpikesKernel = cl::Kernel(program.getProgram(), "HEInet_sumSpikes");
}

void HEInet::createRandom(const std::vector<EIlayer::Configuration> &eilConfigs,
	int predictionRadiusFromE, int predictionRadiusFromI,
	float minInitEWeight, float maxInitEWeight,
	float minInitIWeight, float maxInitIWeight,
	float initEThreshold, float initIThreshold,
	float sparsityE, float sparsityI,
	sys::ComputeSystem &cs, const std::shared_ptr<EIlayer::Kernels> &eilKernels,
	const std::shared_ptr<Kernels> &heiKernels, std::mt19937 &generator)
{
	_kernels = heiKernels;
	_predictionRadiusFromE = predictionRadiusFromE;
	_predictionRadiusFromI = predictionRadiusFromI;

	_eiLayers.resize(eilConfigs.size());

	// Initialize all layers
	for (int li = 0; li < _eiLayers.size(); li++) {
		_eiLayers[li].createRandom(eilConfigs[li],
			minInitEWeight, maxInitEWeight, minInitIWeight, maxInitIWeight,
			initEThreshold, initIThreshold,
			sparsityE, sparsityI,
			cs, eilKernels, generator);
	}

	int predictionFromESize = std::pow(_predictionRadiusFromE * 2 + 1, 2);
	int predictionFromISize = std::pow(_predictionRadiusFromI * 2 + 1, 2);

	_prediction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);
	_predictionPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);

	_inputSpikes = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);
	_inputSpikesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);

	_inputSpikesHistory = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);
	_inputSpikesHistoryPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);

	_inputSpikeTimers = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);
	_inputSpikeTimersPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);

	_eSpikeSums = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eWidth, eilConfigs.front()._eHeight);
	_eSpikeSumsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eWidth, eilConfigs.front()._eHeight);
	
	_iSpikeSums = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._iWidth, eilConfigs.front()._iHeight);
	_iSpikeSumsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._iWidth, eilConfigs.front()._iHeight);

	_eSpikeSumsIterPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eWidth, eilConfigs.front()._eHeight);
	_iSpikeSumsIterPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._iWidth, eilConfigs.front()._iHeight);

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> eFeedForwardDimsCoord;
	eFeedForwardDimsCoord[0] = eilConfigs.front()._eFeedForwardWidth;
	eFeedForwardDimsCoord[1] = eilConfigs.front()._eFeedForwardHeight;
	eFeedForwardDimsCoord[2] = 1;

	cl::size_t<3> ePredictionWeightsDims;
	ePredictionWeightsDims[0] = eilConfigs.front()._eFeedForwardWidth;
	ePredictionWeightsDims[1] = eilConfigs.front()._eFeedForwardHeight;
	ePredictionWeightsDims[2] = predictionFromESize;

	cl::size_t<3> iPredictionWeightsDims;
	iPredictionWeightsDims[0] = eilConfigs.front()._eFeedForwardWidth;
	iPredictionWeightsDims[1] = eilConfigs.front()._eFeedForwardHeight;
	iPredictionWeightsDims[2] = predictionFromISize;

	cl::size_t<3> eDims;
	eDims[0] = eilConfigs.front()._eWidth;
	eDims[1] = eilConfigs.front()._eHeight;
	eDims[2] = 1;

	cl::size_t<3> iDims;
	iDims[0] = eilConfigs.front()._iWidth;
	iDims[1] = eilConfigs.front()._iHeight;
	iDims[2] = 1;

	cs.getQueue().enqueueFillImage(_prediction, zeroColor, zeroCoord, eFeedForwardDimsCoord);
	cs.getQueue().enqueueFillImage(_predictionPrev, zeroColor, zeroCoord, eFeedForwardDimsCoord);

	cs.getQueue().enqueueFillImage(_inputSpikes, zeroColor, zeroCoord, eFeedForwardDimsCoord);
	cs.getQueue().enqueueFillImage(_inputSpikesPrev, zeroColor, zeroCoord, eFeedForwardDimsCoord);

	cs.getQueue().enqueueFillImage(_inputSpikesHistory, zeroColor, zeroCoord, eFeedForwardDimsCoord);
	cs.getQueue().enqueueFillImage(_inputSpikesHistoryPrev, zeroColor, zeroCoord, eFeedForwardDimsCoord);

	cs.getQueue().enqueueFillImage(_inputSpikeTimers, zeroColor, zeroCoord, eFeedForwardDimsCoord);
	cs.getQueue().enqueueFillImage(_inputSpikeTimersPrev, zeroColor, zeroCoord, eFeedForwardDimsCoord);

	cs.getQueue().enqueueFillImage(_eSpikeSums, zeroColor, zeroCoord, eDims);
	cs.getQueue().enqueueFillImage(_eSpikeSumsPrev, zeroColor, zeroCoord, eDims);
	cs.getQueue().enqueueFillImage(_iSpikeSums, zeroColor, zeroCoord, iDims);
	cs.getQueue().enqueueFillImage(_iSpikeSumsPrev, zeroColor, zeroCoord, iDims);
	cs.getQueue().enqueueFillImage(_eSpikeSumsIterPrev, zeroColor, zeroCoord, eDims);
	cs.getQueue().enqueueFillImage(_iSpikeSumsIterPrev, zeroColor, zeroCoord, iDims);

	_predictionFromEWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight, predictionFromESize);
	_predictionFromEWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight, predictionFromESize);
	
	_predictionFromIWeights._weights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight, predictionFromISize);
	_predictionFromIWeights._weightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight, predictionFromISize);

	std::uniform_int_distribution<int> seedDist(0, 10000);

	cl_uint2 seed = { seedDist(generator), seedDist(generator) };

	int index = 0;

	_kernels->_predictionInitializeKernel.setArg(index++, _predictionFromEWeights._weightsPrev);
	_kernels->_predictionInitializeKernel.setArg(index++, _predictionFromIWeights._weightsPrev);
	_kernels->_predictionInitializeKernel.setArg(index++, predictionFromESize);
	_kernels->_predictionInitializeKernel.setArg(index++, predictionFromISize);
	_kernels->_predictionInitializeKernel.setArg(index++, minInitEWeight);
	_kernels->_predictionInitializeKernel.setArg(index++, maxInitEWeight);
	_kernels->_predictionInitializeKernel.setArg(index++, seed);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_predictionInitializeKernel, cl::NullRange, cl::NDRange(eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight));

	cs.getQueue().enqueueCopyImage(_predictionFromEWeights._weightsPrev, _predictionFromEWeights._weights, zeroCoord, zeroCoord, ePredictionWeightsDims);
	cs.getQueue().enqueueCopyImage(_predictionFromIWeights._weightsPrev, _predictionFromIWeights._weights, zeroCoord, zeroCoord, iPredictionWeightsDims);
}

void HEInet::spikeSumBegin(sys::ComputeSystem &cs) {
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> eDims;
	eDims[0] = _eiLayers.front().getConfig()._eWidth;
	eDims[1] = _eiLayers.front().getConfig()._eHeight;
	eDims[2] = 1;

	cl::size_t<3> iDims;
	iDims[0] = _eiLayers.front().getConfig()._iWidth;
	iDims[1] = _eiLayers.front().getConfig()._iHeight;
	iDims[2] = 1;

	cs.getQueue().enqueueFillImage(_eSpikeSums, zeroColor, zeroCoord, eDims);
	cs.getQueue().enqueueFillImage(_eSpikeSumsPrev, zeroColor, zeroCoord, eDims);
	cs.getQueue().enqueueFillImage(_iSpikeSums, zeroColor, zeroCoord, iDims);
	cs.getQueue().enqueueFillImage(_iSpikeSumsPrev, zeroColor, zeroCoord, iDims);
}

void HEInet::sumSpikes(sys::ComputeSystem &cs, float scalar) {
	int index = 0;

	_kernels->_sumSpikesKernel.setArg(index++, _eiLayers.front()._eLayer._states);
	_kernels->_sumSpikesKernel.setArg(index++, _eSpikeSumsPrev);
	_kernels->_sumSpikesKernel.setArg(index++, _eSpikeSums);
	_kernels->_sumSpikesKernel.setArg(index++, scalar);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_sumSpikesKernel, cl::NullRange, cl::NDRange(_eiLayers.front().getConfig()._eWidth, _eiLayers.front().getConfig()._eHeight));

	index = 0;

	_kernels->_sumSpikesKernel.setArg(index++, _eiLayers.front()._iLayer._states);
	_kernels->_sumSpikesKernel.setArg(index++, _iSpikeSumsPrev);
	_kernels->_sumSpikesKernel.setArg(index++, _iSpikeSums);
	_kernels->_sumSpikesKernel.setArg(index++, scalar);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_sumSpikesKernel, cl::NullRange, cl::NDRange(_eiLayers.front().getConfig()._iWidth, _eiLayers.front().getConfig()._iHeight));
}

void HEInet::setInputPhase(sys::ComputeSystem &cs, const cl::Image2D &inputPhaseImage) {
	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> eFeedForwardDimsCoord;
	eFeedForwardDimsCoord[0] = _eiLayers.front().getConfig()._eFeedForwardWidth;
	eFeedForwardDimsCoord[1] = _eiLayers.front().getConfig()._eFeedForwardHeight;
	eFeedForwardDimsCoord[2] = 1;

	cs.getQueue().enqueueCopyImage(inputPhaseImage, _inputSpikeTimersPrev, zeroCoord, zeroCoord, eFeedForwardDimsCoord);
}

void HEInet::setInputPhase(sys::ComputeSystem &cs, cl_uint4 color) {
	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> eFeedForwardDimsCoord;
	eFeedForwardDimsCoord[0] = _eiLayers.front().getConfig()._eFeedForwardWidth;
	eFeedForwardDimsCoord[1] = _eiLayers.front().getConfig()._eFeedForwardHeight;
	eFeedForwardDimsCoord[2] = 1;

	cs.getQueue().enqueueFillImage(_inputSpikeTimersPrev, color, zeroCoord, eFeedForwardDimsCoord);
}

void HEInet::update(sys::ComputeSystem &cs, const cl::Image2D &inputFrequencyImage, const cl::Image2D &zeroImage, float eta, float shDecay, float saDecay) {
	// Update input spikes
	int index = 0;

	_kernels->_updateInputSpikesKernel.setArg(index++, inputFrequencyImage);
	_kernels->_updateInputSpikesKernel.setArg(index++, _inputSpikeTimersPrev);
	_kernels->_updateInputSpikesKernel.setArg(index++, _inputSpikesHistoryPrev);
	_kernels->_updateInputSpikesKernel.setArg(index++, _inputSpikeTimers);
	_kernels->_updateInputSpikesKernel.setArg(index++, _inputSpikes);
	_kernels->_updateInputSpikesKernel.setArg(index++, _inputSpikesHistory);
	_kernels->_updateInputSpikesKernel.setArg(index++, shDecay);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_updateInputSpikesKernel, cl::NullRange, cl::NDRange(_eiLayers.front().getConfig()._eFeedForwardWidth, _eiLayers.front().getConfig()._eFeedForwardHeight));

	const cl::Image2D* pLayerInput = &_inputSpikesPrev;

	// Feed forward
	for (int li = 0; li < _eiLayers.size(); li++) {
		_eiLayers[li].eActivate(cs, *pLayerInput, eta, shDecay, saDecay);

		pLayerInput = &_eiLayers[li]._eLayer._statesPrev;
	}

	pLayerInput = &zeroImage;

	// Feed back
	for (int li = _eiLayers.size() - 1; li >= 0; li--) {
		_eiLayers[li].iActivate(cs, *pLayerInput, eta, shDecay, saDecay);

		pLayerInput = &_eiLayers[li]._iLayer._statesPrev;
	}
}

void HEInet::predict(sys::ComputeSystem &cs) {
	cl_float2 eFeedForwardDimsToEDims = { static_cast<float>(_eiLayers.front().getConfig()._eWidth + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardWidth + 1), static_cast<float>(_eiLayers.front().getConfig()._eHeight + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardHeight + 1) };
	cl_float2 eFeedForwardDimsToIDims = { static_cast<float>(_eiLayers.front().getConfig()._iWidth + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardWidth + 1), static_cast<float>(_eiLayers.front().getConfig()._iHeight + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardHeight + 1) };

	cl_int2 eDims = { _eiLayers.front().getConfig()._eWidth, _eiLayers.front().getConfig()._eHeight };
	cl_int2 iDims = { _eiLayers.front().getConfig()._iWidth, _eiLayers.front().getConfig()._iHeight };

	int index = 0;

	_kernels->_predictKernel.setArg(index++, _eSpikeSumsPrev);
	_kernels->_predictKernel.setArg(index++, _iSpikeSumsPrev);
	_kernels->_predictKernel.setArg(index++, _predictionFromEWeights._weightsPrev);
	_kernels->_predictKernel.setArg(index++, _predictionFromIWeights._weightsPrev);
	_kernels->_predictKernel.setArg(index++, _prediction);
	
	_kernels->_predictKernel.setArg(index++, eFeedForwardDimsToEDims);
	_kernels->_predictKernel.setArg(index++, eFeedForwardDimsToIDims);
	_kernels->_predictKernel.setArg(index++, eDims);
	_kernels->_predictKernel.setArg(index++, iDims);
	_kernels->_predictKernel.setArg(index++, _predictionRadiusFromE);
	_kernels->_predictKernel.setArg(index++, _predictionRadiusFromI);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_predictKernel, cl::NullRange, cl::NDRange(_eiLayers.front().getConfig()._eFeedForwardWidth, _eiLayers.front().getConfig()._eFeedForwardHeight));
}

void HEInet::learn(sys::ComputeSystem &cs, const cl::Image2D &zeroImage,
	float eAlpha, float eBeta, float eDelta, float iAlpha, float iBeta, float iGamma, float iDelta,
	float sparsityE, float sparsityI)
{
	for (int li = 0; li < _eiLayers.size(); li++) {
		if (li == 0) {
			if (li == _eiLayers.size() - 1)
				_eiLayers[li].learn(cs, _inputSpikes, _inputSpikesPrev, zeroImage, zeroImage, eAlpha, eBeta, eDelta, iAlpha, iBeta, iGamma, iDelta, sparsityE, sparsityI);
			else
				_eiLayers[li].learn(cs, _inputSpikes, _inputSpikesPrev, _eiLayers[li + 1]._iLayer._statesHistory, _eiLayers[li + 1]._iLayer._statesHistoryPrev, eAlpha, eBeta, eDelta, iAlpha, iBeta, iGamma, iDelta, sparsityE, sparsityI);
		}
		else {
			if (li == _eiLayers.size() - 1)
				_eiLayers[li].learn(cs, _eiLayers[li - 1]._eLayer._statesHistory, _eiLayers[li - 1]._eLayer._statesHistoryPrev, zeroImage, zeroImage, eAlpha, eBeta, eDelta, iAlpha, iBeta, iGamma, iDelta, sparsityE, sparsityI);
			else
				_eiLayers[li].learn(cs, _eiLayers[li - 1]._eLayer._statesHistory, _eiLayers[li - 1]._eLayer._statesHistoryPrev, _eiLayers[li + 1]._iLayer._statesHistory, _eiLayers[li + 1]._iLayer._statesHistoryPrev, eAlpha, eBeta, eDelta, iAlpha, iBeta, iGamma, iDelta, sparsityE, sparsityI);
		}
	}
}

void HEInet::learnPrediction(sys::ComputeSystem &cs, const cl::Image2D &inputImage, float alpha) {
	cl_float2 eFeedForwardDimsToEDims = { static_cast<float>(_eiLayers.front().getConfig()._eWidth + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardWidth + 1), static_cast<float>(_eiLayers.front().getConfig()._eHeight + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardHeight + 1) };
	cl_float2 eFeedForwardDimsToIDims = { static_cast<float>(_eiLayers.front().getConfig()._iWidth + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardWidth + 1), static_cast<float>(_eiLayers.front().getConfig()._iHeight + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardHeight + 1) };

	cl_int2 eDims = { _eiLayers.front().getConfig()._eWidth, _eiLayers.front().getConfig()._eHeight };
	cl_int2 iDims = { _eiLayers.front().getConfig()._iWidth, _eiLayers.front().getConfig()._iHeight };

	int index = 0;

	_kernels->_predictionLearnKernel.setArg(index++, _eSpikeSumsIterPrev);
	_kernels->_predictionLearnKernel.setArg(index++, _iSpikeSumsIterPrev);
	_kernels->_predictionLearnKernel.setArg(index++, inputImage);
	_kernels->_predictionLearnKernel.setArg(index++, _predictionPrev);
	_kernels->_predictionLearnKernel.setArg(index++, _predictionFromEWeights._weightsPrev);
	_kernels->_predictionLearnKernel.setArg(index++, _predictionFromIWeights._weightsPrev);
	_kernels->_predictionLearnKernel.setArg(index++, _predictionFromEWeights._weights);
	_kernels->_predictionLearnKernel.setArg(index++, _predictionFromIWeights._weights);

	_kernels->_predictionLearnKernel.setArg(index++, eFeedForwardDimsToEDims);
	_kernels->_predictionLearnKernel.setArg(index++, eFeedForwardDimsToIDims);
	_kernels->_predictionLearnKernel.setArg(index++, eDims);
	_kernels->_predictionLearnKernel.setArg(index++, iDims);
	_kernels->_predictionLearnKernel.setArg(index++, _predictionRadiusFromE);
	_kernels->_predictionLearnKernel.setArg(index++, _predictionRadiusFromI);
	_kernels->_predictionLearnKernel.setArg(index++, alpha);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_predictionLearnKernel, cl::NullRange, cl::NDRange(_eiLayers.front().getConfig()._eFeedForwardWidth, _eiLayers.front().getConfig()._eFeedForwardHeight));
}

void HEInet::stepEnd(sys::ComputeSystem &cs) {
	std::swap(_inputSpikes, _inputSpikesPrev);
	std::swap(_inputSpikesHistory, _inputSpikesHistoryPrev);
	std::swap(_inputSpikeTimers, _inputSpikeTimersPrev);

	std::swap(_eSpikeSums, _eSpikeSumsPrev);
	std::swap(_iSpikeSums, _iSpikeSumsPrev);

	for (int li = 0; li < _eiLayers.size(); li++)
		_eiLayers[li].stepEnd();
}

void HEInet::predictionEnd() {
	std::swap(_eSpikeSumsPrev, _eSpikeSumsIterPrev);
	std::swap(_iSpikeSumsPrev, _iSpikeSumsIterPrev);
}

void ei::generateConfigsFromSizes(cl_int2 inputSize, const std::vector<cl_int2> &layerESizes, const std::vector<cl_int2> &layerISizes, std::vector<EIlayer::Configuration> &configs) {
	assert(layerESizes.size() == layerISizes.size());
	
	if (configs.size() != layerESizes.size())
		configs.resize(layerESizes.size());

	for (int li = 0; li < configs.size(); li++) {
		if (li == 0) {
			configs[li]._eFeedForwardWidth = inputSize.x;
			configs[li]._eFeedForwardHeight = inputSize.y;
		}
		else {
			configs[li]._eFeedForwardWidth = layerESizes[li - 1].x;
			configs[li]._eFeedForwardHeight = layerESizes[li - 1].y;
		}

		configs[li]._eWidth = layerESizes[li].x;
		configs[li]._eHeight = layerESizes[li].y;

		configs[li]._iWidth = layerISizes[li].x;
		configs[li]._iHeight = layerISizes[li].y;

		if (li == configs.size() - 1) {
			configs[li]._iFeedBackWidth = 1;
			configs[li]._iFeedBackHeight = 1;
		}
		else {
			configs[li]._iFeedBackWidth = layerISizes[li + 1].x;
			configs[li]._iFeedBackHeight = layerISizes[li + 1].y;
		}
	}
}