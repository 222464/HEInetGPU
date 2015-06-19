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

			cl::Kernel _sumSpikesEKernel;
			cl::Kernel _sumSpikesIKernel;

			// Load kernels from program
			void loadFromProgram(sys::ComputeProgram &program);
		};
	private:
		std::vector<EIlayer> _rscLayers;

		int _predictionRadiusFromE;
		int _predictionRadiusFromI;

		std::shared_ptr<Kernels> _kernels;

	public:
		cl::Image2D _prediction;
		cl::Image2D _predictionPrev;
		cl::Image2D _spikeSumsE;
		cl::Image2D _spikeSumsEPrev;
		cl::Image2D _spikeSumsI;
		cl::Image2D _spikeSumsIPrev;

		EIlayer::Weights2D _predictionFromEWeights;
		EIlayer::Weights2D _predictionFromIWeights;

		// Randomly initialized weights
		void createRandom(const std::vector<EIlayer::Configuration> &rscConfigs,
			int predictionRadiusFromE, int predictionRadiusFromI,
			float minInitEWeight, float maxInitEWeight,
			float minInitIWeight, float maxInitIWeight,
			float initEThreshold, float initIThreshold,
			sys::ComputeSystem &cs, const std::shared_ptr<EIlayer::Kernels> &rscKernels,
			const std::shared_ptr<Kernels> &htslKernels, std::mt19937 &generator);

		// Run through a simulation step
		void update(sys::ComputeSystem &cs, const cl::Image2D &inputImage, const cl::Image2D &zeroImage, float eta, float homeoDecay, float sumSpikeScalar = 1.0f / 17.0f);

		// Get prediction
		void predict(sys::ComputeSystem &cs);

		// Learn (seperate from simulation step)
		void learn(sys::ComputeSystem &cs, const cl::Image2D &inputImage, const cl::Image2D &zeroImage,
			float eAlpha, float eBeta, float eDelta, float iAlpha, float iBeta, float iGamma, float iDelta,
			float sparsityE, float sparsityI);

		// Learn prediction
		void learnPrediction(sys::ComputeSystem &cs, const cl::Image2D &inputImage, float alpha);

		// End step (buffer swap)
		void stepEnd();

		// End prediction step (buffer swap)
		void predictionEnd(sys::ComputeSystem &cs);

		const std::vector<EIlayer> &getRSCLayers() const {
			return _rscLayers;
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