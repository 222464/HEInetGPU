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

// Sampler definitions
constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t normalizedClampedToEdgeNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t defaultNormalizedSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_NONE |
	CLK_FILTER_NEAREST;

constant sampler_t defaultUnnormalizedSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_NONE |
	CLK_FILTER_NEAREST;

// RNG
float randFloat(uint2* state) {
	const float invMaxInt = 1.0f / 4294967296.0f;
	uint x = (*state).x * 17 + (*state).y * 13123;
	(*state).x = (x << 13) ^ x;
	(*state).y ^= (x << 7);

	uint tmp = x * (x * x * 15731 + 74323) + 871483;

	return convert_float(tmp) * invMaxInt;
}

// Random weight initialization - excitatory
void kernel rsc_eInitialize(write_only image3d_t eFeedForwardWeights,
	write_only image3d_t eFeedBackWeights,
	int eFeedForwardSize, int eFeedBackSize,
	float minInitWeight, float maxInitWeight,
	uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1));

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	for (int wi = 0; wi < eFeedForwardSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitWeight - minInitWeight) + minInitWeight;

		write_imagef(eFeedForwardWeights, weightPosition, (float4)(weight));
	}

	for (int wi = 0; wi < eFeedBackSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitWeight - minInitWeight) + minInitWeight;

		write_imagef(eFeedBackWeights, weightPosition, (float4)(weight));
	}
}

// Random weight initialization - inhibitory
void kernel rsc_iInitialize(write_only image3d_t iFeedForwardWeights,
	write_only image3d_t iLateralWeights,
	write_only image3d_t iFeedBackWeights,
	int iFeedForwardSize, int iLateralSize, int iFeedBackSize,
	float minInitWeight, float maxInitWeight,
	uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0), get_global_id(1));

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	for (int wi = 0; wi < iFeedForwardSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitWeight - minInitWeight) + minInitWeight;

		write_imagef(iFeedForwardWeights, weightPosition, (float4)(weight));
	}

	for (int wi = 0; wi < iLateralSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitWeight - minInitWeight) + minInitWeight;

		write_imagef(iLateralWeights, weightPosition, (float4)(weight));
	}

	for (int wi = 0; wi < iFeedBackSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue) * (maxInitWeight - minInitWeight) + minInitWeight;

		write_imagef(iFeedBackWeights, weightPosition, (float4)(weight));
	}
}

void kernel rsc_eActivate(read_only image2d_t feedForwardInput,
	read_only image3d_t eFeedForwardWeightsPrev, read_only image3d_t eFeedBackWeightsPrev, read_only image2d_t eThresholds,
	read_only image2d_t eActivationsPrev, read_only image2d_t eStatesPrev,
	write_only image2d_t eActivations, write_only image2d_t eStates,
	int2 eFeedForwardDims, int2 eDims, int2 iDims,
	float2 eDimsToEFeedForwardDims, float2 eDimsToIDims,
	int eFeedForwardRadius, int eFeedBackRadius,
	float eta)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 feedForwardCenterPosition = (int2)(position.x * eDimsToEFeedForwardDims.x + 0.5f, position.y * eDimsToEFeedForwardDims.y + 0.5f);
	int2 feedBackCenterPosition = (int2)(position.x * eDimsToIDims.x + 0.5f, position.y * eDimsToIDims.y + 0.5f);

	int wi = 0;

	float excitation = 0.0f;

	// Feed forward (excitatory)
	for (int dx = -eFeedForwardRadius; dx <= eFeedForwardRadius; dx++)
		for (int dy = -eFeedForwardRadius; dy <= eFeedForwardRadius; dy++) {
			int2 feedForwardPosition = (int2)(feedForwardCenterPosition.x + dx, feedForwardCenterPosition.y + dy);

			if (feedForwardPosition.x >= 0 && feedForwardPosition.x < eFeedForwardDims.x && feedForwardPosition.y >= 0 && feedForwardPosition.y < eFeedForwardDims.y) {
				float input = read_imagef(inputs, defaultUnnormalizedSampler, feedForwardPosition).x;

				float weight = read_imagef(eFeedForwardWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				excitation += input * weight;
			}

			wi++;
		}

	float inhibition = 0.0f;

	// Feed back (inhibitory)
	for (int dx = -eFeedBackRadius; dx <= eFeedBackRadius; dx++)
		for (int dy = -eFeedBackRadius; dy <= eFeedBackRadius; dy++) {
			int2 feedBackPosition = (int2)(feedBackCenterPosition.x + dx, feedBackCenterPosition.y + dy);

			if (feedBackPosition.x >= 0 && feedBackPosition.x < iDims.x && feedBackPosition.y >= 0 && feedBackPosition.y < iDims.y) {
				float input = read_imagef(inputs, defaultUnnormalizedSampler, feedBackPosition).x;

				float weight = read_imagef(eFeedBackWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				inhibition += input * weight;
			}

			wi++;
		}

	float activationPrev = read_imagef(eActivationsPrev, position).x;

	float activation = (1.0f - eta) * activationPrev + eta * (excitation - inhibition);

	float threshold = read_imagef(eThresholds, position).x;

	float state = 0.0f;

	if (activation > threshold) {
		state = 1.0f;

		activation = 0.0f;
	}

	write_imagef(eActivations, position, (float4)(activation));
	write_imagef(eStates, position, (float4)(state));
}

// Iterated to solve for sparse codes
void kernel rscActivate(read_only image2d_t excitations, read_only image2d_t statesPrev,
	read_only image2d_t activationsPrev, read_only image2d_t spikesPrev,
	read_only image3d_t hiddenHiddenWeightsPrev, read_only image2d_t biasesPrev,
	write_only image2d_t activations, write_only image2d_t spikes, write_only image2d_t states,
	int2 dims,
	int inhibitionRadius, float inhibitionRadiusInv,
	float dt)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float excitation = read_imagef(excitations, defaultUnnormalizedSampler, position).x;

	float inhibition = 0.0f;

	int wi = 0;

	// Inhibit
	for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			if (dx != 0 || dy != 0) {
				int2 inputPosition = (int2)(position.x + dx, position.y + dy);

				if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {
					float input = read_imagef(statesPrev, defaultUnnormalizedSampler, inputPosition).x;

					float weight = read_imagef(hiddenHiddenWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

					float falloff = fmax(0.0f, 1.0f - sqrt((float)(dx * dx + dy * dy)) * inhibitionRadiusInv);

					inhibition += falloff * weight * input;
				}
			}

			wi++;
		}

	// Update activation
	float activationPrev = read_imagef(activationsPrev, defaultUnnormalizedSampler, position).x;

	float activation = (1.0f - dt) * activationPrev + dt * (excitation - inhibition);

	float biasPrev = read_imagef(biasesPrev, defaultUnnormalizedSampler, position).x;

	// Determine spiking
	float spike = 0.0f;

	if (activation > biasPrev) {
		spike = 1.0f;
		activation = 0.0f;
	}

	write_imagef(states, position, (float4)(spike));

	write_imagef(activations, position, (float4)(activation));

	// Accumulate spikes
	float spikeAccumPrev = read_imagef(spikesPrev, defaultUnnormalizedSampler, position).x;

	float spikeAccum = spikeAccumPrev + spike;

	write_imagef(spikes, position, (float4)(spikeAccum));
}

// Learn sparse codes
void kernel rscLearn(read_only image2d_t inputs, read_only image2d_t spikes, read_only image2d_t spikesRecurrentPrev,
	read_only image3d_t hiddenVisibleWeightsPrev, read_only image3d_t hiddenHiddenPrevWeightsPrev, read_only image3d_t hiddenHiddenWeightsPrev, read_only image2d_t biasesPrev,
	write_only image3d_t hiddenVisibleWeights, write_only image3d_t hiddenHiddenPrevWeights, write_only image3d_t hiddenHiddenWeights, write_only image2d_t biases,
	int2 inputDims, int2 dims, float2 dimsToInputDims,
	int receptiveRadius, int recurrentRadius, int inhibitionRadius, float inhibitionRadiusInv, float spikeNorm,
	float4 learningRates, float sparsity, float sparsitySquared)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 inputCenterPosition = (int2)(position.x * dimsToInputDims.x + 0.5f, position.y * dimsToInputDims.y + 0.5f);

	float spike = read_imagef(spikes, defaultUnnormalizedSampler, position).x;

	int wi = 0;

	// Receptive
	for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
		for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
			int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < inputDims.x && inputPosition.y >= 0 && inputPosition.y < inputDims.y) {
				float weightPrev = read_imagef(hiddenVisibleWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				float input = read_imagef(inputs, defaultUnnormalizedSampler, inputPosition).x;
	
				float weight = weightPrev + learningRates.x * spike * (input - spike * weightPrev);

				write_imagef(hiddenVisibleWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	wi = 0;

	// Recurrent
	for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
		for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
			int2 inputPosition = (int2)(position.x + dx, position.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {
				float weightPrev = read_imagef(hiddenHiddenPrevWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;

				float input = read_imagef(spikesRecurrentPrev, defaultUnnormalizedSampler, inputPosition).x * spikeNorm;
			
				float weight = weightPrev + learningRates.y * spike * (input - spike * weightPrev);

				write_imagef(hiddenHiddenPrevWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	wi = 0;

	// Inhibitory
	for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			int2 inputPosition = (int2)(position.x + dx, position.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {		
				float weightPrev = read_imagef(hiddenHiddenWeightsPrev, defaultUnnormalizedSampler, (int4)(position.x, position.y, wi, 0)).x;
				
				float input = read_imagef(spikes, defaultUnnormalizedSampler, inputPosition).x;

				float falloff = fmax(0.0f, 1.0f - sqrt((float)(dx * dx + dy * dy)) * inhibitionRadiusInv);

				float weight = fmax(0.0f, weightPrev + learningRates.z * (falloff * input * spike - sparsitySquared));

				write_imagef(hiddenHiddenWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	// Bias
	float biasPrev = read_imagef(biasesPrev, defaultUnnormalizedSampler, position).x;

	float bias = biasPrev + learningRates.w * (spike - sparsity);

	write_imagef(biases, position, (float4)(bias));
}