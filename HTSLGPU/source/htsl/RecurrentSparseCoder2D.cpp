#include "RecurrentSparseCoder2D.h"

using namespace htsl;

void RecurrentSparseCoder2D::Kernels::loadFromProgram(sys::ComputeProgram &program) {
	_initializeKernel = cl::Kernel(program.getProgram(), "rscInitialize");
	_excitationKernel = cl::Kernel(program.getProgram(), "rscExcitation");
	_activateKernel = cl::Kernel(program.getProgram(), "rscActivate");
	_learnKernel = cl::Kernel(program.getProgram(), "rscLearn");
}

void RecurrentSparseCoder2D::createRandom(int inputWidth, int inputHeight, int width, int height,
	int receptiveRadius, int recurrentRadius, int inhibitionRadius,
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

	int receptiveSize = std::pow(_receptiveRadius * 2 + 1, 2);
	int recurrentSize = std::pow(_recurrentRadius * 2 + 1, 2);
	int inhibitionSize = std::pow(_inhibitionRadius * 2 + 1, 2);

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
	cl_float4 oneColor = { 1.0f, 1.0f, 1.0f, 1.0f };

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> dimsCoord;
	dimsCoord[0] = _width;
	dimsCoord[1] = _height;
	dimsCoord[2] = 1;

	cs.getQueue().enqueueFillImage(_activationsPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_spikesPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_spikesRecurrentPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_statesPrev, zeroColor, zeroCoord, dimsCoord);

	cs.getQueue().enqueueFillImage(_biasesPrev, oneColor, zeroCoord, dimsCoord);

	int index = 0;

	std::uniform_int_distribution<int> seedDist(0, 10000);

	cl_uint2 seed = { seedDist(generator), seedDist(generator) };

	_kernels->_initializeKernel.setArg(index++, _hiddenVisibleWeightsPrev);
	_kernels->_initializeKernel.setArg(index++, _hiddenHiddenPrevWeightsPrev);
	_kernels->_initializeKernel.setArg(index++, _hiddenHiddenWeightsPrev);
	_kernels->_initializeKernel.setArg(index++, receptiveSize);
	_kernels->_initializeKernel.setArg(index++, recurrentSize);
	_kernels->_initializeKernel.setArg(index++, inhibitionSize);
	_kernels->_initializeKernel.setArg(index++, seed);

	cs.getQueue().enqueueNDRangeKernel(_kernels->_initializeKernel, cl::NullRange, cl::NDRange(_width, _height));	
}

void RecurrentSparseCoder2D::update(sys::ComputeSystem &cs, const cl::Image2D &inputs, float dt, int iterations) {
	cl_int2 inputDims = { _inputWidth, _inputHeight };
	cl_int2 dims = { _width, _height };
	cl_float2 dimsToInputDims = { static_cast<float>(_inputWidth + 1) / static_cast<float>(_width + 1), static_cast<float>(_inputHeight + 1) / static_cast<float>(_height + 1) };
	cl_float2 inputDimsToDims = { static_cast<float>(_width + 1) / static_cast<float>(_inputWidth + 1), static_cast<float>(_height + 1) / static_cast<float>(_inputHeight + 1) };

	cl_int2 reverseReceptiveRadii = { std::ceil((_receptiveRadius + 0.5f) * inputDimsToDims.x + 0.5f), std::ceil((_receptiveRadius + 0.5f) * inputDimsToDims.y + 0.5f) };

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::size_t<3> zeroCoord;
	zeroCoord[0] = zeroCoord[1] = zeroCoord[2] = 0;

	cl::size_t<3> dimsCoord;
	dimsCoord[0] = _width;
	dimsCoord[1] = _height;
	dimsCoord[2] = 1;

	cs.getQueue().enqueueFillImage(_activationsPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_statesPrev, zeroColor, zeroCoord, dimsCoord);
	cs.getQueue().enqueueFillImage(_spikesPrev, zeroColor, zeroCoord, dimsCoord);

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

		cs.getQueue().enqueueNDRangeKernel(_kernels->_excitationKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	float iterationsInv = 1.0f / iterations;
	float inhibitionRadiusInv = 1.0f / _inhibitionRadius;

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
			_kernels->_activateKernel.setArg(index++, iterationsInv);

			cs.getQueue().enqueueNDRangeKernel(_kernels->_activateKernel, cl::NullRange, cl::NDRange(_width, _height));
		}

		if (i != iterations - 1) {
			std::swap(_activations, _activationsPrev);
			std::swap(_states, _statesPrev);
			std::swap(_spikes, _spikesPrev);
		}
	}
}

void RecurrentSparseCoder2D::learn(sys::ComputeSystem &cs, const cl::Image2D &inputs, float alpha, float beta, float gamma, float delta, float sparsity) {
	cl_int2 inputDims = { _inputWidth, _inputHeight };
	cl_int2 dims = { _width, _height };
	cl_float2 dimsToInputDims = { static_cast<float>(_inputWidth + 1) / static_cast<float>(_width + 1), static_cast<float>(_inputHeight + 1) / static_cast<float>(_height + 1) };
	cl_float4 learningRates = { alpha, beta, gamma, delta };

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
		_kernels->_learnKernel.setArg(index++, learningRates);
		_kernels->_learnKernel.setArg(index++, sparsity);
		_kernels->_learnKernel.setArg(index++, sparsity * sparsity);

		cs.getQueue().enqueueNDRangeKernel(_kernels->_learnKernel, cl::NullRange, cl::NDRange(_width, _height));
	}

	std::swap(_hiddenVisibleWeights, _hiddenVisibleWeightsPrev);
	std::swap(_hiddenHiddenPrevWeights, _hiddenHiddenPrevWeightsPrev);
	std::swap(_hiddenHiddenWeights, _hiddenHiddenWeightsPrev);
	std::swap(_biases, _biasesPrev);
}

void RecurrentSparseCoder2D::stepEnd() {
	std::swap(_spikes, _spikesRecurrentPrev);
}