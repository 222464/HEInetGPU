#include "HEInet.h"

using namespace ei;

void HEInet::Kernels::loadFromProgram(sys::ComputeProgram &program) {
	// Create kernels
	_predictionInitializeKernel = cl::Kernel(program.getProgram(), "HEInet_predictionInitialize");

	_predictKernel = cl::Kernel(program.getProgram(), "HEInet_predict");

	_predictionLearnKernel = cl::Kernel(program.getProgram(), "HEInet_predictionLearn");
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

	_eShortAveragePrevIter = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eWidth, eilConfigs.front()._eHeight);
	_iShortAveragePrevIter = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._iWidth, eilConfigs.front()._iHeight);

	_inputLongAverages = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);
	_inputLongAveragesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), eilConfigs.front()._eFeedForwardWidth, eilConfigs.front()._eFeedForwardHeight);

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl_float4 inputLongAverageColor = { sparsityE, sparsityE, sparsityE, sparsityE };

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

	cs.getQueue().enqueueFillImage(_predictionPrev, zeroColor, zeroCoord, eFeedForwardDimsCoord);

	cs.getQueue().enqueueFillImage(_eShortAveragePrevIter, zeroColor, zeroCoord, eDims);
	cs.getQueue().enqueueFillImage(_iShortAveragePrevIter, zeroColor, zeroCoord, iDims);

	cs.getQueue().enqueueFillImage(_inputLongAverages, zeroColor, zeroCoord, eFeedForwardDimsCoord);
	cs.getQueue().enqueueFillImage(_inputLongAveragesPrev, zeroColor, zeroCoord, eFeedForwardDimsCoord);

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

void HEInet::update(sys::ComputeSystem &cs, const cl::Image2D &inputImage, const cl::Image2D &zeroImage, int iter, float eta) {
	const cl::Image2D* pLayerInput = &inputImage;

	float iterInv = 1.0f / iter;

	for (int li = 0; li < _eiLayers.size(); li++)
		_eiLayers[li].exStepBegin(cs);

	for (int i = 0; i < iter; i++) {
		// Feed forward
		for (int li = 0; li < _eiLayers.size(); li++) {
			_eiLayers[li].eActivate(cs, *pLayerInput, eta, iterInv);

			pLayerInput = &_eiLayers[li]._eLayer._statesPrev;
		}

		pLayerInput = &zeroImage;

		// Feed back
		for (int li = _eiLayers.size() - 1; li >= 0; li--) {
			_eiLayers[li].iActivate(cs, *pLayerInput, eta, iterInv);

			pLayerInput = &_eiLayers[li]._iLayer._statesPrev;
		}

		for (int li = 0; li < _eiLayers.size(); li++)
			_eiLayers[li].simStepEnd();
	}
}

void HEInet::updateLongAverages(sys::ComputeSystem &cs, const cl::Image2D &inputImage, float longAverageDecay) {
	for (int li = 0; li < _eiLayers.size(); li++)
		_eiLayers[li].updateLongAverages(cs, longAverageDecay);

	// Update long averages
	int index = 0;

	_eiLayers.front().getKernels()->_longAverageKernel.setArg(index++, inputImage);
	_eiLayers.front().getKernels()->_longAverageKernel.setArg(index++, _inputLongAveragesPrev);
	_eiLayers.front().getKernels()->_longAverageKernel.setArg(index++, _inputLongAverages);
	_eiLayers.front().getKernels()->_longAverageKernel.setArg(index++, longAverageDecay);

	cs.getQueue().enqueueNDRangeKernel(_eiLayers.front().getKernels()->_longAverageKernel, cl::NullRange, cl::NDRange(_eiLayers.front().getConfig()._eFeedForwardWidth, _eiLayers.front().getConfig()._eFeedForwardHeight));
}

void HEInet::predict(sys::ComputeSystem &cs) {
	cl_float2 eFeedForwardDimsToEDims = { static_cast<float>(_eiLayers.front().getConfig()._eWidth + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardWidth + 1), static_cast<float>(_eiLayers.front().getConfig()._eHeight + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardHeight + 1) };
	cl_float2 eFeedForwardDimsToIDims = { static_cast<float>(_eiLayers.front().getConfig()._iWidth + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardWidth + 1), static_cast<float>(_eiLayers.front().getConfig()._iHeight + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardHeight + 1) };

	cl_int2 eDims = { _eiLayers.front().getConfig()._eWidth, _eiLayers.front().getConfig()._eHeight };
	cl_int2 iDims = { _eiLayers.front().getConfig()._iWidth, _eiLayers.front().getConfig()._iHeight };

	int index = 0;

	_kernels->_predictKernel.setArg(index++, _eiLayers.front()._eLayer._shortAveragesPrev);
	_kernels->_predictKernel.setArg(index++, _eiLayers.front()._iLayer._shortAveragesPrev);
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

void HEInet::learn(sys::ComputeSystem &cs, const cl::Image2D &inputImage, const cl::Image2D &zeroImage,
	float eAlpha, float eBeta, float eDelta, float iAlpha, float iBeta, float iGamma, float iDelta,
	float sparsityE, float sparsityI)
{
	for (int li = 0; li < _eiLayers.size(); li++) {
		if (li == 0) {
			if (li == _eiLayers.size() - 1)
				_eiLayers[li].learn(cs, inputImage, _inputLongAveragesPrev, zeroImage, zeroImage, eAlpha, eBeta, eDelta, iAlpha, iBeta, iGamma, iDelta, sparsityE, sparsityI);
			else
				_eiLayers[li].learn(cs, inputImage, _inputLongAveragesPrev, _eiLayers[li + 1]._iLayer._shortAveragesPrev, _eiLayers[li + 1]._iLayer._longAveragesPrev, eAlpha, eBeta, eDelta, iAlpha, iBeta, iGamma, iDelta, sparsityE, sparsityI);
		}
		else {
			if (li == _eiLayers.size() - 1)
				_eiLayers[li].learn(cs, _eiLayers[li - 1]._eLayer._shortAveragesPrev, _eiLayers[li - 1]._eLayer._longAveragesPrev, zeroImage, zeroImage, eAlpha, eBeta, eDelta, iAlpha, iBeta, iGamma, iDelta, sparsityE, sparsityI);
			else
				_eiLayers[li].learn(cs, _eiLayers[li - 1]._eLayer._shortAveragesPrev, _eiLayers[li - 1]._eLayer._longAveragesPrev, _eiLayers[li + 1]._iLayer._shortAveragesPrev, _eiLayers[li + 1]._iLayer._longAveragesPrev, eAlpha, eBeta, eDelta, iAlpha, iBeta, iGamma, iDelta, sparsityE, sparsityI);
		}
	}
}

void HEInet::learnPrediction(sys::ComputeSystem &cs, const cl::Image2D &inputImage, float alpha) {
	cl_float2 eFeedForwardDimsToEDims = { static_cast<float>(_eiLayers.front().getConfig()._eWidth + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardWidth + 1), static_cast<float>(_eiLayers.front().getConfig()._eHeight + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardHeight + 1) };
	cl_float2 eFeedForwardDimsToIDims = { static_cast<float>(_eiLayers.front().getConfig()._iWidth + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardWidth + 1), static_cast<float>(_eiLayers.front().getConfig()._iHeight + 1) / static_cast<float>(_eiLayers.front().getConfig()._eFeedForwardHeight + 1) };

	cl_int2 eDims = { _eiLayers.front().getConfig()._eWidth, _eiLayers.front().getConfig()._eHeight };
	cl_int2 iDims = { _eiLayers.front().getConfig()._iWidth, _eiLayers.front().getConfig()._iHeight };

	int index = 0;

	_kernels->_predictionLearnKernel.setArg(index++, _eShortAveragePrevIter);
	_kernels->_predictionLearnKernel.setArg(index++, _iShortAveragePrevIter);
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

void HEInet::exStepEnd(sys::ComputeSystem &cs) {
	for (int li = 0; li < _eiLayers.size(); li++)
		_eiLayers[li].exStepEnd();

	// Clear spike sums
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

	cs.getQueue().enqueueCopyImage(_eiLayers.front()._eLayer._shortAveragesPrev, _eShortAveragePrevIter, zeroCoord, zeroCoord, eDims);
	cs.getQueue().enqueueCopyImage(_eiLayers.front()._iLayer._shortAveragesPrev, _iShortAveragePrevIter, zeroCoord, zeroCoord, iDims);

	std::swap(_inputLongAverages, _inputLongAveragesPrev);
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