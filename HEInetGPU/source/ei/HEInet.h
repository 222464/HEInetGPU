#pragma once

#include "EIlayer.h"

namespace ei {
	class HEInet {
	public:
		// Kernels this system uses
		struct Kernels {
			cl::Kernel _predictionInitializeKernel;
	
			cl::Kernel _predictKernel;
			cl::Kernel _predictionLearnKernel;

			cl::Kernel _updateInputSpikesKernel;

			cl::Kernel _sumSpikesKernel;

			// Load kernels from program
			void loadFromProgram(sys::ComputeProgram &program);
		};
	private:
		std::vector<EIlayer> _eiLayers;

		int _predictionRadiusFromE;
		int _predictionRadiusFromI;

		std::shared_ptr<Kernels> _kernels;

	public:
		cl::Image2D _prediction;
		cl::Image2D _predictionPrev;

		cl::Image2D _inputSpikes;
		cl::Image2D _inputSpikesPrev;

		cl::Image2D _inputSpikesHistory;
		cl::Image2D _inputSpikesHistoryPrev;

		cl::Image2D _inputSpikeTimers;
		cl::Image2D _inputSpikeTimersPrev;

		cl::Image2D _eSpikeSums;
		cl::Image2D _iSpikeSums;
		cl::Image2D _eSpikeSumsPrev;
		cl::Image2D _iSpikeSumsPrev;
		cl::Image2D _eSpikeSumsIterPrev;
		cl::Image2D _iSpikeSumsIterPrev;

		EIlayer::Weights2D _predictionFromEWeights;
		EIlayer::Weights2D _predictionFromIWeights;

		// Randomly initialized weights
		void createRandom(const std::vector<EIlayer::Configuration> &eilConfigs,
			int predictionRadiusFromE, int predictionRadiusFromI,
			float minInitEWeight, float maxInitEWeight,
			float minInitIWeight, float maxInitIWeight,
			float initEThreshold, float initIThreshold,
			float sparsityE, float sparsityI,
			sys::ComputeSystem &cs, const std::shared_ptr<EIlayer::Kernels> &eilKernels,
			const std::shared_ptr<Kernels> &heiKernels, std::mt19937 &generator);

		// Begin summation of spikes
		void spikeSumBegin(sys::ComputeSystem &cs);

		void sumSpikes(sys::ComputeSystem &cs, float scalar);

		// Run through an example step (multiple simulation steps)
		void update(sys::ComputeSystem &cs, const cl::Image2D &inputImage, const cl::Image2D &zeroImage, float eta, float shDecay);

		// Get prediction
		void predict(sys::ComputeSystem &cs);

		// Learn
		void learn(sys::ComputeSystem &cs, const cl::Image2D &zeroImage,
			float eAlpha, float eBeta, float eDelta, float iAlpha, float iBeta, float iGamma, float iDelta,
			float sparsityE, float sparsityI);

		// Learn prediction
		void learnPrediction(sys::ComputeSystem &cs, const cl::Image2D &inputImage, float alpha);

		void stepEnd(sys::ComputeSystem &cs);

		void predictionEnd();

		const std::vector<EIlayer> &getEIlayers() const {
			return _eiLayers;
		}

		int getPredictionRadiusFromE() const {
			return _predictionRadiusFromE;
		}

		int getPredictionRadiusFromI() const {
			return _predictionRadiusFromI;
		}
	};

	void generateConfigsFromSizes(cl_int2 inputSize, const std::vector<cl_int2> &layerESizes, const std::vector<cl_int2> &layerISizes, std::vector<EIlayer::Configuration> &configs);
}